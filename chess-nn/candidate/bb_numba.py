"""Numba-JIT bitboard move generator.

Tables are built from the already-perft-verified pure-Python core, then the
hot functions are njit-compiled. All bitboard math is uint64. Compiling is
triggered once at import (warm-up), which in the real container is paid inside
the 90s init budget, off the game clock.
"""

import numpy as np
from numba import njit, uint64, int32, int64

import bb_core as _c

ONE = np.uint64(1)
U64 = np.uint64
NO_EP = 64

# ---- tables as numpy arrays (uint64) ------------------------------------
KNIGHT_ATT = np.array(_c.KNIGHT_ATTACKS, dtype=np.uint64)
KING_ATT = np.array(_c.KING_ATTACKS, dtype=np.uint64)
PAWN_ATT = np.array(_c.PAWN_ATTACKS, dtype=np.uint64)          # shape (2,64)
RAYS = np.array(_c.RAYS, dtype=np.uint64)                       # shape (8,64)

DEBRUIJN = np.uint64(0x03f79d71b4cb0a89)
INDEX64 = np.array([
    0, 47, 1, 56, 48, 27, 2, 60, 57, 49, 41, 37, 28, 16, 3, 61,
    54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11, 4, 62,
    46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
    25, 39, 14, 33, 19, 30, 9, 24, 13, 18, 8, 12, 7, 6, 5, 63,
], dtype=np.int64)

# board indices
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
STM, CR, EP, HM = 12, 13, 14, 15
C_WK, C_WQ, C_BK, C_BQ = 1, 2, 4, 8
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=False)
def bsf(bb):  # index of least-significant set bit (bb != 0)
    return INDEX64[(((bb ^ (bb - ONE)) * DEBRUIJN) >> np.uint64(58))]


@njit(cache=False)
def bsr(bb):  # index of most-significant set bit (bb != 0)
    bb |= bb >> np.uint64(1)
    bb |= bb >> np.uint64(2)
    bb |= bb >> np.uint64(4)
    bb |= bb >> np.uint64(8)
    bb |= bb >> np.uint64(16)
    bb |= bb >> np.uint64(32)
    return INDEX64[((bb * DEBRUIJN) >> np.uint64(58))]


@njit(cache=False)
def ray_pos(sq, occ, d):
    attacks = RAYS[d, sq]
    blockers = attacks & occ
    if blockers != 0:
        attacks ^= RAYS[d, bsf(blockers)]
    return attacks


@njit(cache=False)
def ray_neg(sq, occ, d):
    attacks = RAYS[d, sq]
    blockers = attacks & occ
    if blockers != 0:
        attacks ^= RAYS[d, bsr(blockers)]
    return attacks


@njit(cache=False)
def bishop_att(sq, occ):
    return (ray_pos(sq, occ, 2) | ray_pos(sq, occ, 3)
            | ray_neg(sq, occ, 6) | ray_neg(sq, occ, 7))


@njit(cache=False)
def rook_att(sq, occ):
    return (ray_pos(sq, occ, 0) | ray_pos(sq, occ, 1)
            | ray_neg(sq, occ, 4) | ray_neg(sq, occ, 5))


@njit(cache=False)
def occ_side(bd, base):
    return (bd[base] | bd[base + 1] | bd[base + 2]
            | bd[base + 3] | bd[base + 4] | bd[base + 5])


@njit(cache=False)
def is_attacked(bd, sq, by_white):
    occ = occ_side(bd, 0) | occ_side(bd, 6)
    if by_white:
        if PAWN_ATT[1, sq] & bd[WP]:
            return True
        if KNIGHT_ATT[sq] & bd[WN]:
            return True
        if KING_ATT[sq] & bd[WK]:
            return True
        if bishop_att(sq, occ) & (bd[WB] | bd[WQ]):
            return True
        if rook_att(sq, occ) & (bd[WR] | bd[WQ]):
            return True
    else:
        if PAWN_ATT[0, sq] & bd[BP]:
            return True
        if KNIGHT_ATT[sq] & bd[BN]:
            return True
        if KING_ATT[sq] & bd[BK]:
            return True
        if bishop_att(sq, occ) & (bd[BB] | bd[BQ]):
            return True
        if rook_att(sq, occ) & (bd[BR] | bd[BQ]):
            return True
    return False


@njit(cache=False)
def _enc(frm, to, promo, flag):
    return int32(frm | (to << 6) | (promo << 12) | (flag << 15))


@njit(cache=False)
def gen_pseudo(bd, out):
    """Fill `out` (int32[:]) with pseudo-legal moves; return the count."""
    n = 0
    white = bd[STM] == 0
    if white:
        base = 0
        oppbase = 6
        push = 8
        start_mask = np.uint64(0x000000000000FF00)
        promo_mask = np.uint64(0xFF00000000000000)
        pcolor = 0
    else:
        base = 6
        oppbase = 0
        push = -8
        start_mask = np.uint64(0x00FF000000000000)
        promo_mask = np.uint64(0x00000000000000FF)
        pcolor = 1

    own = occ_side(bd, base)
    opp = occ_side(bd, oppbase)
    occ = own | opp
    empty = ~occ
    ep = int(bd[EP])

    # Pawns
    pawns = bd[base + 0]
    while pawns:
        frm = bsf(pawns)
        pawns &= pawns - ONE
        to = frm + push
        to_bit = ONE << np.uint64(to)
        if empty & to_bit:
            if to_bit & promo_mask:
                out[n] = _enc(frm, to, 4, 0); n += 1
                out[n] = _enc(frm, to, 3, 0); n += 1
                out[n] = _enc(frm, to, 2, 0); n += 1
                out[n] = _enc(frm, to, 1, 0); n += 1
            else:
                out[n] = _enc(frm, to, 0, 0); n += 1
                if (ONE << np.uint64(frm)) & start_mask:
                    to2 = frm + 2 * push
                    if empty & (ONE << np.uint64(to2)):
                        out[n] = _enc(frm, to2, 0, 1); n += 1
        caps = PAWN_ATT[pcolor, frm] & opp
        while caps:
            c = bsf(caps)
            caps &= caps - ONE
            cbit = ONE << np.uint64(c)
            if cbit & promo_mask:
                out[n] = _enc(frm, c, 4, 0); n += 1
                out[n] = _enc(frm, c, 3, 0); n += 1
                out[n] = _enc(frm, c, 2, 0); n += 1
                out[n] = _enc(frm, c, 1, 0); n += 1
            else:
                out[n] = _enc(frm, c, 0, 0); n += 1
        if ep != NO_EP:
            if PAWN_ATT[pcolor, frm] & (ONE << np.uint64(ep)):
                out[n] = _enc(frm, ep, 0, 2); n += 1

    # Knights
    kn = bd[base + 1]
    while kn:
        frm = bsf(kn)
        kn &= kn - ONE
        t = KNIGHT_ATT[frm] & ~own
        while t:
            to = bsf(t)
            t &= t - ONE
            out[n] = _enc(frm, to, 0, 0); n += 1

    # Bishops
    bi = bd[base + 2]
    while bi:
        frm = bsf(bi)
        bi &= bi - ONE
        t = bishop_att(frm, occ) & ~own
        while t:
            to = bsf(t)
            t &= t - ONE
            out[n] = _enc(frm, to, 0, 0); n += 1

    # Rooks
    ro = bd[base + 3]
    while ro:
        frm = bsf(ro)
        ro &= ro - ONE
        t = rook_att(frm, occ) & ~own
        while t:
            to = bsf(t)
            t &= t - ONE
            out[n] = _enc(frm, to, 0, 0); n += 1

    # Queens
    qu = bd[base + 4]
    while qu:
        frm = bsf(qu)
        qu &= qu - ONE
        t = (bishop_att(frm, occ) | rook_att(frm, occ)) & ~own
        while t:
            to = bsf(t)
            t &= t - ONE
            out[n] = _enc(frm, to, 0, 0); n += 1

    # King
    ksq = bsf(bd[base + 5])
    t = KING_ATT[ksq] & ~own
    while t:
        to = bsf(t)
        t &= t - ONE
        out[n] = _enc(ksq, to, 0, 0); n += 1

    # Castling
    cr = int(bd[CR])
    if white:
        if (cr & C_WK) and (occ & ((ONE << np.uint64(5)) | (ONE << np.uint64(6)))) == 0:
            if (not is_attacked(bd, 4, False) and not is_attacked(bd, 5, False)
                    and not is_attacked(bd, 6, False)):
                out[n] = _enc(4, 6, 0, 3); n += 1
        if (cr & C_WQ) and (occ & ((ONE << np.uint64(1)) | (ONE << np.uint64(2)) | (ONE << np.uint64(3)))) == 0:
            if (not is_attacked(bd, 4, False) and not is_attacked(bd, 3, False)
                    and not is_attacked(bd, 2, False)):
                out[n] = _enc(4, 2, 0, 3); n += 1
    else:
        if (cr & C_BK) and (occ & ((ONE << np.uint64(61)) | (ONE << np.uint64(62)))) == 0:
            if (not is_attacked(bd, 60, True) and not is_attacked(bd, 61, True)
                    and not is_attacked(bd, 62, True)):
                out[n] = _enc(60, 62, 0, 3); n += 1
        if (cr & C_BQ) and (occ & ((ONE << np.uint64(57)) | (ONE << np.uint64(58)) | (ONE << np.uint64(59)))) == 0:
            if (not is_attacked(bd, 60, True) and not is_attacked(bd, 59, True)
                    and not is_attacked(bd, 58, True)):
                out[n] = _enc(60, 58, 0, 3); n += 1

    return n


@njit(cache=False)
def make(bd, m):
    nb = bd.copy()
    white = bd[STM] == 0
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    flag = (m >> 15) & 3
    frm_bit = ONE << np.uint64(frm)
    to_bit = ONE << np.uint64(to)

    if white:
        base = 0
        oppbase = 6
    else:
        base = 6
        oppbase = 0

    # find moved piece
    moved = base
    for pi in range(base, base + 6):
        if nb[pi] & frm_bit:
            moved = pi
            break

    is_pawn = moved == base  # pawn is first in each side's block
    opp_occ_before = occ_side(bd, oppbase)
    is_capture = (opp_occ_before & to_bit) != 0 or flag == 2

    nb[moved] ^= frm_bit

    # normal capture
    if flag != 2 and (opp_occ_before & to_bit):
        for pi in range(oppbase, oppbase + 6):
            if nb[pi] & to_bit:
                nb[pi] ^= to_bit
                break

    # en passant capture
    if flag == 2:
        if white:
            cap_sq = to - 8
            nb[BP] ^= (ONE << np.uint64(cap_sq))
        else:
            cap_sq = to + 8
            nb[WP] ^= (ONE << np.uint64(cap_sq))

    # place piece / promotion
    if promo != 0:
        # promo codes 1..4 -> N,B,R,Q  => piece index base+1..base+4
        promo_idx = base + promo
        nb[promo_idx] |= to_bit
    else:
        nb[moved] |= to_bit

    # castling rook move
    if flag == 3:
        if to == 6:
            nb[WR] ^= (ONE << np.uint64(7)); nb[WR] |= (ONE << np.uint64(5))
        elif to == 2:
            nb[WR] ^= (ONE << np.uint64(0)); nb[WR] |= (ONE << np.uint64(3))
        elif to == 62:
            nb[BR] ^= (ONE << np.uint64(63)); nb[BR] |= (ONE << np.uint64(61))
        elif to == 58:
            nb[BR] ^= (ONE << np.uint64(56)); nb[BR] |= (ONE << np.uint64(59))

    # castling rights
    cr = nb[CR]
    if moved == WK:
        cr &= np.uint64(~(C_WK | C_WQ) & 0xFF)
    elif moved == BK:
        cr &= np.uint64(~(C_BK | C_BQ) & 0xFF)
    if frm == 0 or to == 0:
        cr &= np.uint64(~C_WQ & 0xFF)
    if frm == 7 or to == 7:
        cr &= np.uint64(~C_WK & 0xFF)
    if frm == 56 or to == 56:
        cr &= np.uint64(~C_BQ & 0xFF)
    if frm == 63 or to == 63:
        cr &= np.uint64(~C_BK & 0xFF)
    nb[CR] = cr

    if flag == 1:
        nb[EP] = np.uint64(frm + (8 if white else -8))
    else:
        nb[EP] = np.uint64(NO_EP)

    if is_pawn or is_capture:
        nb[HM] = np.uint64(0)
    else:
        nb[HM] = bd[HM] + ONE

    nb[STM] = np.uint64(1 - bd[STM])
    return nb


@njit(cache=False)
def gen_legal(bd, out):
    """Fill out with legal moves, return count."""
    tmp = np.empty(256, dtype=np.int32)
    npseudo = gen_pseudo(bd, tmp)
    white = bd[STM] == 0
    n = 0
    for i in range(npseudo):
        nb = make(bd, tmp[i])
        ksq = bsf(nb[WK] if white else nb[BK])
        if not is_attacked(nb, ksq, not white):
            out[n] = tmp[i]
            n += 1
    return n


@njit(cache=False)
def perft(bd, depth):
    if depth == 0:
        return 1
    out = np.empty(256, dtype=np.int32)
    n = gen_legal(bd, out)
    if depth == 1:
        return n
    total = 0
    for i in range(n):
        total += perft(make(bd, out[i]), depth - 1)
    return total


def board_np(fen):
    return np.array(_c.board_from_fen(fen), dtype=np.uint64)


# Warm up the JIT at import (paid in the init budget in the real container).
def _warmup():
    b = board_np('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    perft(b, 1)


_warmup()
