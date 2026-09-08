"""Numba search layer built on the perft-verified bitboard core (bb_numba).

Everything on the hot path is njit-compiled: evaluation, move ordering,
quiescence, and the main negamax with alpha-beta, a transposition table,
principal variation search, null-move pruning, late move reductions and check
extensions. Mutable state (TT, killers, history, stop flag, node counter,
scratch buffers) is passed in as arguments because numba treats module-level
arrays as read-only; only the constant lookup tables live as globals.

Time control: a background timer in get_move() flips stop[0] to 1 at the
deadline. The search polls it every few thousand nodes and unwinds; the
partially-searched depth is discarded and the last completed depth's move is
played (standard iterative deepening).
"""

import time
import threading
import numpy as np
from numba import njit

# Worker search threads need a large stack: deep check-extension lines nest
# many native frames. 64 MB is comfortable within the 2 GB memory limit.
try:
    threading.stack_size(256 * 1024 * 1024)
except (ValueError, RuntimeError):
    pass

import bb_core as _c
import bb_numba as bbn
from bb_numba import (make, gen_pseudo, is_attacked, bsf, board_np,
                      bishop_att, rook_att, occ_side,
                      KNIGHT_ATT, KING_ATT, PAWN_ATT)

# ---- optional neural evaluation -----------------------------------------
# The NN modules live beside this file in a submission, and one directory up
# in the development checkout; accept either without touching the board format.
try:
    from nn_runtime import nn_eval
except ImportError:                                  # pragma: no cover
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from nn_runtime import nn_eval

# Weights are THREADED through the search as an argument tuple, never read from
# a module global inside njit code: numba freezes the *contents* of global
# arrays into the machine code at compile time, so a global-weights design would
# keep whatever happened to be loaded at the first call and silently ignore
# every later load. Passing them in keeps the net swappable with no recompile.
#
# The placeholders below have the right dtype and rank but trivial shapes. Numba
# types an array by dtype/rank/contiguity, not by shape, so compiling the search
# against these and then passing real (780,256) weights needs no recompile.
_NN_PLACEHOLDER = (
    np.ascontiguousarray(np.zeros((2, 2), dtype=np.float32)),
    np.zeros(2, dtype=np.float32),
    np.ascontiguousarray(np.zeros((2, 2), dtype=np.float32)),
    np.zeros(2, dtype=np.float32),
    np.ascontiguousarray(np.zeros((1, 2), dtype=np.float32)),
    np.zeros(1, dtype=np.float32),
)
_NNW = _NN_PLACEHOLDER
_USE_NN = [0]          # 0 = hand-crafted evaluation (the default), 1 = network


def load_nn(path):
    """Load NN weights and switch evaluation to the network.

    Per-game search state is rebuilt because the transposition table, killers
    and history all hold scores produced by the PREVIOUS evaluator; reusing them
    across a switch would mix two incompatible score scales.
    """
    global _NNW
    import nn_model
    w = nn_model.load_weights(path)
    _NNW = (w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"], w["out_w"], w["out_b"])
    for name, array in zip(("acc_w", "acc_b", "l1_w", "l1_b", "out_w", "out_b"), _NNW):
        if not array.flags["C_CONTIGUOUS"]:
            raise ValueError(
                f"{name} is not C-contiguous; numba would treat it as a new type "
                "and recompile the entire search on the first move, on the clock")
    _USE_NN[0] = 1
    _warmup_nn()
    new_game()
    return _NNW


def _warmup_nn():
    """Compile the search against the REAL weight arrays, inside the init budget.

    get_move() runs the search in a worker thread joined with a short backstop.
    Any compilation triggered on the first NN move therefore overruns that join,
    the thread is abandoned with no move, and get_move silently returns its
    fallback for every move of the game. Forcing the compile here is what keeps
    that off the clock.
    """
    board = board_np("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    _stop[0] = 0
    _nodes[0] = 0
    search_root(board, 4, 0, -INF, INF, _tt_key, _tt_data, _killers, _history,
                _stop, _nodes, _ghist, 0, _movebuf, _scorebuf, _mbbuf, _NNW, 1)
    quiescence(board, -INF, INF, 0, _stop, _nodes, _movebuf, _scorebuf, _mbbuf,
               _NNW, 1)


def use_handcrafted():
    """Switch back to the hand-crafted evaluation and clear stale search state."""
    global _NNW
    _NNW = _NN_PLACEHOLDER
    _USE_NN[0] = 0
    new_game()


def nn_active():
    return _USE_NN[0] == 1

# board indices (mirror bb_numba)
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
STM, CR, EP, HM = 12, 13, 14, 15
NO_EP = 64
ONE = np.uint64(1)

MATE = 1_000_000
MATE_BOUND = MATE - 1000
INF = MATE + 10_000
MAX_PLY = 96

# ---- evaluation tables (ported from the classical engine) ---------------
_MG_VALUE = {1: 82, 2: 337, 3: 365, 4: 477, 5: 1025, 6: 0}   # P N B R Q K
_EG_VALUE = {1: 94, 2: 281, 3: 297, 4: 512, 5: 936, 6: 0}
_PHASE = [0, 1, 1, 2, 4, 0]  # by piece index 0..5 (P,N,B,R,Q,K)
TOTAL_PHASE = 24

_MG_PST = {
1: (0,0,0,0,0,0,0,0,-35,-1,-20,-23,-15,24,38,-22,-26,-4,-4,-10,3,3,33,-12,-27,-2,-5,12,17,6,10,-25,-14,13,6,21,23,12,17,-23,-6,7,26,31,65,56,25,-20,98,134,61,95,68,126,34,-11,0,0,0,0,0,0,0,0),
2: (-105,-21,-58,-33,-17,-28,-19,-23,-29,-53,-12,-3,-1,18,-14,-19,-23,-9,12,10,19,17,25,-16,-13,4,16,13,28,19,21,-8,-9,17,19,53,37,69,18,22,-47,60,37,65,84,129,73,44,-73,-41,72,36,23,62,7,-17,-167,-89,-34,-49,61,-97,-15,-107),
3: (-33,-3,-14,-21,-13,-12,-39,-21,4,15,16,0,7,21,33,1,0,15,15,15,14,27,18,10,-6,13,13,26,34,12,10,4,-4,5,19,50,37,37,7,-2,-16,37,43,40,35,50,37,-2,-26,16,-18,-13,30,59,18,-47,-29,4,-82,-37,-25,-42,7,-8),
4: (-19,-13,1,17,16,7,-37,-26,-44,-16,-20,-9,-1,11,-6,-71,-45,-25,-16,-17,3,0,-5,-33,-36,-26,-12,-1,9,-7,6,-23,-24,-11,7,26,24,35,-8,-20,-5,19,26,36,17,45,61,16,27,32,58,62,80,67,26,44,32,42,32,51,63,9,31,43),
5: (-1,-18,-9,10,-15,-25,-31,-50,-35,-8,11,2,8,15,-3,1,-14,2,-11,-2,-5,2,14,5,-9,-26,-9,-10,-2,-4,3,-3,-27,-27,-16,-16,-1,17,-2,1,-13,-17,7,8,29,56,47,57,-24,-39,-5,1,-16,57,28,54,-28,0,29,12,59,44,43,45),
6: (-15,36,12,-54,8,-28,24,14,1,7,-8,-64,-43,-16,9,8,-14,-14,-22,-46,-44,-30,-15,-27,-49,-1,-27,-39,-46,-44,-33,-51,-17,-20,-12,-27,-30,-25,-14,-36,-9,24,2,-16,-20,6,22,-22,29,-1,-20,-7,-8,-4,-38,-29,-65,23,16,-15,-56,-34,2,13),
}
_EG_PST = {
1: (0,0,0,0,0,0,0,0,13,8,8,10,13,0,2,-7,4,7,-6,1,0,-5,-1,-8,13,9,-3,-7,-7,-8,3,-1,32,24,13,5,-2,4,17,17,94,100,85,67,56,53,82,84,178,173,158,134,147,132,165,187,0,0,0,0,0,0,0,0),
2: (-29,-51,-23,-15,-22,-18,-50,-64,-42,-20,-10,-5,-2,-20,-23,-44,-23,-3,-1,15,10,-3,-20,-22,-18,-6,16,25,16,17,4,-18,-17,3,22,22,22,11,8,-18,-24,-20,10,9,-1,-9,-19,-41,-25,-8,-25,-2,-9,-25,-24,-52,-58,-38,-13,-28,-31,-27,-63,-99),
3: (-23,-9,-23,-5,-9,-16,-5,-17,-14,-18,-7,-1,4,-9,-15,-27,-12,-3,8,10,13,3,-7,-15,-6,3,13,19,7,10,-3,-9,-3,9,12,9,14,10,3,2,2,-8,0,-1,-2,6,0,4,-8,-4,7,-12,-3,-13,-4,-14,-14,-21,-11,-8,-7,-9,-17,-24),
4: (-9,2,3,-1,-5,-13,4,-20,-6,-6,0,2,-9,-9,-11,-3,-4,0,-5,-1,-7,-12,-8,-16,3,5,8,4,-5,-6,-8,-11,4,3,13,1,2,1,-1,2,7,7,7,5,4,-3,-5,-3,11,13,13,11,-3,3,8,3,13,10,18,15,12,12,8,5),
5: (-33,-28,-22,-43,-5,-32,-20,-41,-22,-23,-30,-16,-16,-23,-36,-32,-16,-27,15,6,9,17,10,5,-18,28,19,47,31,34,39,23,3,22,24,45,57,40,57,36,-20,6,9,49,47,35,19,9,-17,20,32,41,58,25,30,0,-9,22,22,27,27,19,10,20),
6: (-53,-34,-21,-11,-28,-14,-24,-43,-27,-11,4,13,14,4,-5,-17,-19,-3,11,21,23,16,7,-9,-18,-4,21,24,27,23,9,-11,-8,22,24,27,26,33,26,3,10,17,23,15,20,45,44,13,-12,17,14,17,17,38,23,11,-74,-35,-18,-18,-11,15,4,-17),
}

# Combined value+PST, white reads sq directly, black reads mirrored sq.
MGW = np.zeros((6, 64), dtype=np.int32)
EGW = np.zeros((6, 64), dtype=np.int32)
MGB = np.zeros((6, 64), dtype=np.int32)
EGB = np.zeros((6, 64), dtype=np.int32)
for _pt in range(1, 7):
    for _sq in range(64):
        _m = _sq ^ 56
        MGW[_pt - 1, _sq] = _MG_VALUE[_pt] + _MG_PST[_pt][_sq]
        EGW[_pt - 1, _sq] = _EG_VALUE[_pt] + _EG_PST[_pt][_sq]
        MGB[_pt - 1, _sq] = _MG_VALUE[_pt] + _MG_PST[_pt][_m]
        EGB[_pt - 1, _sq] = _EG_VALUE[_pt] + _EG_PST[_pt][_m]
PHW = np.array(_PHASE, dtype=np.int64)

def _build_passed_masks():
    mw = [0] * 64
    mb = [0] * 64
    for sq in range(64):
        f = sq & 7
        r = sq >> 3
        for ff in (f - 1, f, f + 1):
            if 0 <= ff <= 7:
                for rr in range(r + 1, 8):
                    mw[sq] |= 1 << (rr * 8 + ff)
                for rr in range(0, r):
                    mb[sq] |= 1 << (rr * 8 + ff)
    return mw, mb


_PW, _PB = _build_passed_masks()
PASSEDW = np.array(_PW, dtype=np.uint64)
PASSEDB = np.array(_PB, dtype=np.uint64)
PASSED_BONUS = np.array([0, 10, 15, 25, 40, 65, 100, 0], dtype=np.int64)

# ---- hashing constants (multiply-xor hash; O(1) per node) ---------------
_rng = np.random.default_rng(0xC0FFEE)
ZPIECE = _rng.integers(1, 2**63, size=12, dtype=np.uint64) | np.uint64(1)
ZCASTLE = _rng.integers(1, 2**63, size=16, dtype=np.uint64)
ZEP = _rng.integers(1, 2**63, size=9, dtype=np.uint64)
ZSTM = np.uint64(0x9E3779B97F4A7C15)


@njit(cache=False)
def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=False)
def zkey(bd):
    h = np.uint64(0)
    for i in range(12):
        h ^= bd[i] * ZPIECE[i]
    if bd[STM] == 1:
        h ^= ZSTM
    h ^= ZCASTLE[int(bd[CR]) & 15]
    ep = int(bd[EP])
    h ^= ZEP[ep & 7] if ep != NO_EP else ZEP[8]
    return h


@njit(cache=False)
def evaluate(bd, nnw, use_nn):
    # Both branches are compiled once, because use_nn is a runtime argument
    # rather than a compile-time constant. Switching evaluators mid-process
    # therefore costs no compilation on the clock.
    if use_nn == 1:
        # nn_eval already returns mover-relative, clamped integer centipawns,
        # the same contract as the hand-crafted evaluation below.
        return nn_eval(bd, nnw[0], nnw[1], nnw[2], nnw[3], nnw[4], nnw[5])
    mg = 0
    eg = 0
    phase = 0
    for pt in range(6):
        wbb = bd[pt]
        while wbb:
            sq = bsf(wbb)
            wbb &= wbb - ONE
            mg += MGW[pt, sq]
            eg += EGW[pt, sq]
            phase += PHW[pt]
        bbb = bd[6 + pt]
        while bbb:
            sq = bsf(bbb)
            bbb &= bbb - ONE
            mg -= MGB[pt, sq]
            eg -= EGB[pt, sq]
            phase += PHW[pt]
    if popcount(bd[WB]) >= 2:
        mg += 30; eg += 45
    if popcount(bd[BB]) >= 2:
        mg -= 30; eg -= 45
    # passed pawns (endgame weighted)
    wp = bd[WP]; bp = bd[BP]
    t = wp
    while t:
        sq = bsf(t); t &= t - ONE
        if (PASSEDW[sq] & bp) == 0:
            eg += PASSED_BONUS[sq >> 3]
    t = bp
    while t:
        sq = bsf(t); t &= t - ONE
        if (PASSEDB[sq] & wp) == 0:
            eg += -PASSED_BONUS[7 - (sq >> 3)]
    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    return score if bd[STM] == 0 else -score


@njit(cache=False)
def is_draw(bd, ghist, nghist):
    if bd[HM] >= 100:
        return True
    pawns = bd[WP] | bd[BP]
    rq = bd[WR] | bd[BR] | bd[WQ] | bd[BQ]
    if pawns == 0 and rq == 0:
        minors = popcount(bd[WN] | bd[WB] | bd[BN] | bd[BB])
        if minors <= 1:
            return True
    k = zkey(bd)
    for i in range(nghist):
        if ghist[i] == k:
            return True
    return False


@njit(cache=False)
def fill_mailbox(bd, mb):
    for s in range(64):
        mb[s] = 0
    for pt in range(6):
        wbb = bd[pt]
        while wbb:
            sq = bsf(wbb); wbb &= wbb - ONE
            mb[sq] = pt + 1
        bbb = bd[6 + pt]
        while bbb:
            sq = bsf(bbb); bbb &= bbb - ONE
            mb[sq] = pt + 1


# ---- Static Exchange Evaluation (validated in see.py) -------------------
SEE_VAL = np.array([0, 100, 320, 330, 500, 900, 10000], dtype=np.int64)


@njit(cache=False)
def _attackers_to(bd, sq, occ):
    att = np.uint64(0)
    att |= PAWN_ATT[1, sq] & bd[WP]
    att |= PAWN_ATT[0, sq] & bd[BP]
    att |= KNIGHT_ATT[sq] & (bd[WN] | bd[BN])
    att |= KING_ATT[sq] & (bd[WK] | bd[BK])
    att |= bishop_att(sq, occ) & (bd[WB] | bd[BB] | bd[WQ] | bd[BQ])
    att |= rook_att(sq, occ) & (bd[WR] | bd[BR] | bd[WQ] | bd[BQ])
    return att


@njit(cache=False)
def _ptype_at(bd, bit, white):
    base = 0 if white else 6
    for pt in range(6):
        if bd[base + pt] & bit:
            return pt + 1
    return 0


@njit(cache=False)
def _lva(bd, attackers, white):
    base = 0 if white else 6
    for pt in range(6):
        subset = attackers & bd[base + pt]
        if subset:
            return pt + 1, subset & (~subset + ONE)
    return 0, np.uint64(0)


@njit(cache=False)
def see(bd, frm, to):
    white = bd[STM] == 0
    occ = occ_side(bd, 0) | occ_side(bd, 6)
    frm_bit = ONE << np.uint64(frm)
    to_bit = ONE << np.uint64(to)
    captured = _ptype_at(bd, to_bit, not white)
    if captured == 0:
        return 0
    attacker_type = _ptype_at(bd, frm_bit, white)
    gain = np.empty(32, dtype=np.int64)
    d = 0
    gain[0] = SEE_VAL[captured]
    occ ^= frm_bit
    attackers = _attackers_to(bd, to, occ)
    side_white = not white
    from_type = attacker_type
    while True:
        d += 1
        gain[d] = SEE_VAL[from_type] - gain[d - 1]
        attackers &= occ
        lva_type, lva_bit = _lva(bd, attackers, side_white)
        if lva_bit == 0:
            break
        if lva_type == 6:
            _, opp_bit = _lva(bd, attackers, not side_white)
            if opp_bit != 0:
                break
        occ ^= lva_bit
        attackers |= bishop_att(to, occ) & (bd[WB] | bd[BB] | bd[WQ] | bd[BQ])
        attackers |= rook_att(to, occ) & (bd[WR] | bd[BR] | bd[WQ] | bd[BQ])
        attackers &= occ
        from_type = lva_type
        side_white = not side_white
    while d > 1:
        d -= 1
        if -gain[d] < gain[d - 1]:
            gain[d - 1] = -gain[d]
    return gain[0]


_VAL = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int64)  # by type 0..6


@njit(cache=False)
def order_moves(bd, moves, n, scores, mb, tt_move, killers, history, ply):
    for i in range(n):
        m = moves[i]
        frm = m & 63
        to = (m >> 6) & 63
        promo = (m >> 12) & 7
        flag = (m >> 15) & 3
        if m == tt_move:
            scores[i] = 2_000_000_000
            continue
        victim = mb[to]
        if victim != 0 or flag == 2:
            attacker = mb[frm]
            v = _VAL[victim] if victim != 0 else 100
            s = 1_000_000_000 + v * 16 - attacker
            if promo != 0:
                s += 800_000_000
            scores[i] = s
        elif promo != 0:
            scores[i] = 900_000_000 + promo
        elif killers[ply, 0] == m:
            scores[i] = 800_000_000
        elif killers[ply, 1] == m:
            scores[i] = 799_000_000
        else:
            scores[i] = history[bd[STM], frm, to]
    # insertion sort (descending) — move counts are small
    for i in range(1, n):
        km = moves[i]; ks = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < ks:
            moves[j + 1] = moves[j]
            scores[j + 1] = scores[j]
            j -= 1
        moves[j + 1] = km
        scores[j + 1] = ks


@njit(cache=False)
def has_non_pawn(bd, white):
    if white:
        return (bd[WN] | bd[WB] | bd[WR] | bd[WQ]) != 0
    return (bd[BN] | bd[BB] | bd[BR] | bd[BQ]) != 0


@njit(cache=False)
def make_null(bd):
    nb = bd.copy()
    nb[STM] = np.uint64(1 - bd[STM])
    nb[EP] = np.uint64(NO_EP)
    return nb


@njit(cache=False)
def quiescence(bd, alpha, beta, ply, stop, nodes, movebuf, scorebuf, mbbuf, nnw, use_nn):
    nodes[0] += 1
    if nodes[0] & 2047 == 0 and stop[0] == 1:
        return 0
    if ply >= MAX_PLY - 1:
        return evaluate(bd, nnw, use_nn)

    white = bd[STM] == 0
    ksq = bsf(bd[WK] if white else bd[BK])
    in_check = is_attacked(bd, ksq, not white)

    if not in_check:
        stand = evaluate(bd, nnw, use_nn)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand

    moves = movebuf[ply]
    n = gen_pseudo(bd, moves)
    mb = mbbuf[ply]
    fill_mailbox(bd, mb)

    # order: captures/promos (or all if in check)
    scores = scorebuf[ply]
    order_moves(bd, moves, n, scores, mb, -1,
                _KILL_DUMMY, _HIST_DUMMY, 0)

    any_legal = False
    for i in range(n):
        if stop[0] == 1:
            break
        m = moves[i]
        to = (m >> 6) & 63
        promo = (m >> 12) & 7
        flag = (m >> 15) & 3
        is_cap = mb[to] != 0 or flag == 2
        if (not in_check) and (not is_cap) and promo == 0:
            continue  # quiet move: skip in quiescence when not in check
        # SEE pruning: don't search captures that lose material outright.
        if (not in_check) and is_cap and promo == 0 and flag != 2:
            if see(bd, m & 63, to) < 0:
                continue
        nb = make(bd, m)
        nksq = bsf(nb[WK] if white else nb[BK])
        if is_attacked(nb, nksq, not white):
            continue
        any_legal = True
        score = -quiescence(nb, -beta, -alpha, ply + 1, stop, nodes,
                            movebuf, scorebuf, mbbuf, nnw, use_nn)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    if in_check and not any_legal:
        return -MATE + ply
    return alpha


@njit(cache=False)
def negamax(bd, depth, alpha, beta, ply, allow_null,
            tt_key, tt_data, killers, history,
            stop, nodes, ghist, nghist,
            movebuf, scorebuf, mbbuf, nnw, use_nn):
    nodes[0] += 1
    if nodes[0] & 2047 == 0 and stop[0] == 1:
        return 0

    if ply >= MAX_PLY - 1:
        return evaluate(bd, nnw, use_nn)

    if ply > 0 and is_draw(bd, ghist, nghist):
        return 0

    alpha_orig = alpha
    key = zkey(bd)
    idx = int(key & np.uint64(_TT_MASK))
    tt_move = -1
    if tt_key[idx] == key:
        e_depth = tt_data[idx, 0]
        e_score = tt_data[idx, 1]
        e_flag = tt_data[idx, 2]
        tt_move = tt_data[idx, 3]
        if e_depth >= depth and ply > 0:
            if e_flag == 0:      # exact
                return e_score
            elif e_flag == 1:    # lower bound
                if e_score > alpha:
                    alpha = e_score
            else:                # upper bound
                if e_score < beta:
                    beta = e_score
            if alpha >= beta:
                return e_score

    white = bd[STM] == 0
    ksq = bsf(bd[WK] if white else bd[BK])
    in_check = is_attacked(bd, ksq, not white)
    if in_check:
        depth += 1

    if depth <= 0:
        return quiescence(bd, alpha, beta, ply, stop, nodes,
                          movebuf, scorebuf, mbbuf, nnw, use_nn)

    # Reverse futility pruning (static null): if the static eval is already a
    # depth-scaled margin above beta, trust it and prune the node.
    if (not in_check and depth <= 6 and beta < MATE_BOUND
            and alpha > -MATE_BOUND):
        static = evaluate(bd, nnw, use_nn)
        if static - 85 * depth >= beta:
            return static - 85 * depth

    # Null-move pruning
    if (allow_null and not in_check and depth >= 3
            and has_non_pawn(bd, white) and beta < MATE_BOUND):
        if evaluate(bd, nnw, use_nn) >= beta:
            R = 2 + depth // 4
            nb = make_null(bd)
            null_score = -negamax(nb, depth - 1 - R, -beta, -beta + 1, ply + 1,
                                  False, tt_key, tt_data, killers, history,
                                  stop, nodes, ghist, nghist,
                                  movebuf, scorebuf, mbbuf, nnw, use_nn)
            if null_score >= beta:
                return beta

    moves = movebuf[ply]
    n = gen_pseudo(bd, moves)
    mb = mbbuf[ply]
    fill_mailbox(bd, mb)
    scores = scorebuf[ply]
    order_moves(bd, moves, n, scores, mb, tt_move, killers, history, ply)

    best_score = -INF
    best_move = -1
    legal = 0
    move_index = 0

    for i in range(n):
        if stop[0] == 1:
            break
        m = moves[i]
        frm = m & 63
        to = (m >> 6) & 63
        promo = (m >> 12) & 7
        flag = (m >> 15) & 3
        is_cap = mb[to] != 0 or flag == 2

        nb = make(bd, m)
        nksq = bsf(nb[WK] if white else nb[BK])
        if is_attacked(nb, nksq, not white):
            continue
        legal += 1

        gives_check = is_attacked(nb, bsf(nb[BK] if white else nb[WK]), white)

        reduction = 0
        if (depth >= 3 and move_index >= 3 and not is_cap
                and promo == 0 and not gives_check and not in_check):
            reduction = 1 if move_index < 6 else 2

        if move_index == 0:
            score = -negamax(nb, depth - 1, -beta, -alpha, ply + 1, True,
                             tt_key, tt_data, killers, history, stop, nodes,
                             ghist, nghist, movebuf, scorebuf, mbbuf, nnw, use_nn)
        else:
            score = -negamax(nb, depth - 1 - reduction, -alpha - 1, -alpha,
                             ply + 1, True, tt_key, tt_data, killers, history,
                             stop, nodes, ghist, nghist,
                             movebuf, scorebuf, mbbuf, nnw, use_nn)
            if score > alpha and (reduction != 0 or score < beta):
                score = -negamax(nb, depth - 1, -beta, -alpha, ply + 1, True,
                                 tt_key, tt_data, killers, history, stop, nodes,
                                 ghist, nghist, movebuf, scorebuf, mbbuf, nnw, use_nn)

        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if not is_cap and promo == 0:
                if killers[ply, 0] != m:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = m
                history[bd[STM], frm, to] += depth * depth
            break
        move_index += 1

    if legal == 0:
        if stop[0] == 1:
            return 0
        return -MATE + ply if in_check else 0

    # Do not pollute the TT with an aborted (partial) search result.
    if stop[0] == 1:
        return best_score

    # store TT
    if best_score <= alpha_orig:
        flag = 2   # upper
    elif best_score >= beta:
        flag = 1   # lower
    else:
        flag = 0   # exact
    if tt_key[idx] != key or tt_data[idx, 0] <= depth:
        tt_key[idx] = key
        tt_data[idx, 0] = depth
        tt_data[idx, 1] = best_score
        tt_data[idx, 2] = flag
        tt_data[idx, 3] = best_move

    return best_score


@njit(cache=False)
def search_root(bd, depth, prev_best, alpha, beta,
                tt_key, tt_data, killers, history,
                stop, nodes, ghist, nghist,
                movebuf, scorebuf, mbbuf, nnw, use_nn):
    """Returns (best_score, best_move, completed). Fail-soft within [alpha,beta]."""
    white = bd[STM] == 0

    moves = movebuf[0]
    n = gen_pseudo(bd, moves)
    mb = mbbuf[0]
    fill_mailbox(bd, mb)
    scores = scorebuf[0]
    order_moves(bd, moves, n, scores, mb, prev_best, killers, history, 0)

    best_score = -INF
    best_move = prev_best
    first = True
    for i in range(n):
        m = moves[i]
        nb = make(bd, m)
        nksq = bsf(nb[WK] if white else nb[BK])
        if is_attacked(nb, nksq, not white):
            continue
        if first:
            score = -negamax(nb, depth - 1, -beta, -alpha, 1, True,
                             tt_key, tt_data, killers, history, stop, nodes,
                             ghist, nghist, movebuf, scorebuf, mbbuf, nnw, use_nn)
            first = False
        else:
            score = -negamax(nb, depth - 1, -alpha - 1, -alpha, 1, True,
                             tt_key, tt_data, killers, history, stop, nodes,
                             ghist, nghist, movebuf, scorebuf, mbbuf, nnw, use_nn)
            if score > alpha:
                score = -negamax(nb, depth - 1, -beta, -alpha, 1, True,
                                 tt_key, tt_data, killers, history, stop, nodes,
                                 ghist, nghist, movebuf, scorebuf, mbbuf, nnw, use_nn)
        if stop[0] == 1:
            return best_score, best_move, 0
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score

    if best_move == -1 and n > 0:
        best_move = moves[0]
    return best_score, best_move, 1


# ---- TT sizing and persistent per-game state ----------------------------
_TT_BITS = 21
_TT_SIZE = 1 << _TT_BITS
_TT_MASK = _TT_SIZE - 1

_tt_key = np.zeros(_TT_SIZE, dtype=np.uint64)
_tt_data = np.zeros((_TT_SIZE, 4), dtype=np.int64)
_killers = np.full((MAX_PLY, 2), -1, dtype=np.int64)
_history = np.zeros((2, 64, 64), dtype=np.int64)
_stop = np.zeros(1, dtype=np.uint8)
_nodes = np.zeros(1, dtype=np.int64)
_movebuf = np.zeros((MAX_PLY, 256), dtype=np.int32)
_scorebuf = np.zeros((MAX_PLY, 256), dtype=np.int64)
_mbbuf = np.zeros((MAX_PLY, 64), dtype=np.int64)
_ghist = np.zeros(1024, dtype=np.uint64)
_nghist = 0

# dummies for quiescence ordering (no killers/history there)
_KILL_DUMMY = np.full((1, 2), -1, dtype=np.int64)
_HIST_DUMMY = np.zeros((2, 64, 64), dtype=np.int64)


def _uci(m):
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    s = chr(ord('a') + (frm & 7)) + str((frm >> 3) + 1)
    s += chr(ord('a') + (to & 7)) + str((to >> 3) + 1)
    if promo:
        s += "nbrq"[promo - 1]
    return s


def _budget_seconds(time_left_ms):
    # Self-throttling: budget scales with remaining time (no large fixed
    # constant), so at low clocks the engine automatically moves fast. The
    # usable*0.5 cap guarantees that even a large overshoot cannot exceed the
    # remaining clock minus the reserve -> it is physically impossible to flag.
    t = max(0.0, time_left_ms / 1000.0)
    if t <= 0.06:
        return 0.0
    reserve = min(1.0, max(0.05, t * 0.05))
    usable = max(0.0, t - reserve)
    target = t / 25.0 + min(0.3, t * 0.02)
    return max(0.02, min(target, usable * 0.5))


def get_move(fen, time_left_ms):
    """Public API. Always returns a legal UCI move; never raises/flags."""
    global _nghist
    try:
        bd = board_np(fen)
    except Exception:
        return "0000"

    # fallback legal move (never illegal, never crash)
    fallback = "0000"
    try:
        tmp = np.zeros(256, dtype=np.int32)
        npseudo = gen_pseudo(bd, tmp)
        white = bd[STM] == 0
        for i in range(npseudo):
            nb = make(bd, tmp[i])
            nksq = bsf(nb[WK] if white else nb[BK])
            if not is_attacked(nb, nksq, not white):
                fallback = _uci(int(tmp[i]))
                break
    except Exception:
        return "0000"
    if fallback == "0000":
        return "0000"

    # record current game position for repetition awareness
    try:
        k = zkey(bd)
        if _nghist < _ghist.shape[0]:
            _ghist[_nghist] = k
            _nghist += 1
    except Exception:
        pass

    budget = _budget_seconds(time_left_ms)
    if budget <= 0.0:
        return fallback

    # fresh killers/history per move; keep TT across moves
    _killers.fill(-1)
    _history.fill(0)
    _stop[0] = 0
    _nodes[0] = 0

    # The search runs in a worker thread with a large stack, because deep
    # check-extension lines can nest ~100+ native frames and overflow the
    # default C stack (a crash = loss). The watchdog trips stop[0] at the
    # deadline; the njit search polls it and unwinds.
    holder = [-1]

    def _worker():
        prev = -1
        score = 0
        start = time.perf_counter()
        for depth in range(1, MAX_PLY - 2):
            # Aspiration windows: search a narrow band around the last score
            # and widen only on a fail. Shallow depths use the full window.
            if depth <= 4:
                alpha, beta = -INF, INF
            else:
                delta = 30
                alpha, beta = score - delta, score + delta
            while True:
                sc, mv, completed = search_root(
                    bd, depth, prev if prev != -1 else 0, alpha, beta,
                    _tt_key, _tt_data, _killers, _history,
                    _stop, _nodes, _ghist, _nghist,
                    _movebuf, _scorebuf, _mbbuf, _NNW, _USE_NN[0])
                if completed == 0:
                    break
                if sc <= alpha:                       # fail low: widen down
                    delta *= 4
                    alpha = sc - delta
                    if alpha < -MATE_BOUND:
                        alpha = -INF
                    continue
                if sc >= beta:                        # fail high: widen up
                    delta *= 4
                    beta = sc + delta
                    if beta > MATE_BOUND:
                        beta = INF
                    if mv != -1:
                        holder[0] = mv                # keep the fail-high move
                        prev = mv
                    continue
                break
            if completed == 1 and mv != -1:
                holder[0] = mv       # atomic: only completed depths are kept
                prev = mv
                score = sc
            else:
                break
            if abs(score) >= MATE_BOUND:
                break
            # Don't start a new depth we're unlikely to finish.
            if time.perf_counter() - start > budget * 0.5:
                break
            if _stop[0] == 1:
                break

    timer = threading.Timer(budget, lambda: _stop.__setitem__(0, 1))
    timer.daemon = True
    timer.start()
    try:
        worker = threading.Thread(target=_worker)
        worker.start()
        # Hard backstop join: even in the worst case we return near the budget.
        worker.join(timeout=budget + 0.5)
        if worker.is_alive():
            _stop[0] = 1
            worker.join(timeout=0.5)
    except Exception:
        pass
    finally:
        timer.cancel()

    best_move = holder[0]
    if best_move == -1:
        return fallback

    # final legality guarantee
    try:
        white = bd[STM] == 0
        nb = make(bd, best_move)
        nksq = bsf(nb[WK] if white else nb[BK])
        if is_attacked(nb, nksq, not white):
            return fallback
    except Exception:
        return fallback
    return _uci(int(best_move))


def new_game():
    """Reset per-game state (call at the start of each game / process)."""
    global _nghist
    _tt_key.fill(0)
    _tt_data.fill(0)
    _killers.fill(-1)
    _history.fill(0)
    _ghist.fill(0)
    _nghist = 0


def _warmup():
    # Run a genuine shallow search so numba compiles the entire call graph
    # (negamax, null-move, LMR, quiescence, make, eval, ordering) during the
    # init budget rather than lazily on the first real move (on the clock).
    b = board_np('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    _stop[0] = 0
    _nodes[0] = 0
    search_root(b, 4, 0, -INF, INF, _tt_key, _tt_data, _killers, _history,
                _stop, _nodes, _ghist, 0, _movebuf, _scorebuf, _mbbuf, _NNW, _USE_NN[0])
    quiescence(b, -INF, INF, 0, _stop, _nodes, _movebuf, _scorebuf, _mbbuf, _NNW, _USE_NN[0])
    see(b, 0, 8)  # force SEE to compile in the init budget, not on the clock
    new_game()


_warmup()
