"""Shared feature encoding for the NNUE-style value net  (AI Chessathon).

================================  THE CONTRACT  ================================
This file is the SINGLE SOURCE OF TRUTH for turning a chess position into the
780 network inputs. Data preparation, training, and the in-engine runtime MUST
all use it. If two of them encode positions even slightly differently, the net
trains on one thing and plays on another and silently gets much weaker. So:
nobody re-implements the encoding anywhere else. Extend it here, and the parity
test (test_nn.py) must stay green.

--------------------------------  ENCODING  -----------------------------------
768 piece inputs = 2 (own / opponent) x 6 (piece types) x 64 (squares),
plus 4 castling-right and 8 en-passant-file inputs.

  * "Side-to-move relative": the position is always seen from the mover's side.
    - White to move: squares as-is (a1 = 0 ... h8 = 63).
    - Black to move: the board is mirrored vertically (square ^ 56) and the
      colours swap, so the mover's pieces are always the "own" set on the
      near ranks. This is why one net handles both colours.

  * Feature index for a piece:
        index = own_or_opp * 384  +  piece_type * 64  +  rel_square
      own_or_opp : 0 = side-to-move's piece, 1 = opponent's
      piece_type : 0=P 1=N 2=B 3=R 4=Q 5=K   (matches the engine's bitboard order)
      rel_square : square from the mover's perspective (see mirror rule above)

  * The network output is ALSO side-to-move relative: positive = good for the
    side to move, in centipawns. That is exactly what the engine's search wants
    (its evaluate() already returns a mover-relative score), so no sign flip is
    needed when the net replaces evaluate().

--------------------------------  ENGINE BOARD  -------------------------------
The runtime encoder reads the engine's board array `bd` (length 16):
    bd[0..5]   white  P,N,B,R,Q,K  bitboards (uint64, bit s = square s)
    bd[6..11]  black  P,N,B,R,Q,K  bitboards
    bd[12]     side to move (0 = white, 1 = black)
    bd[13]     castling rights (1=WK, 2=WQ, 4=BK, 8=BQ)
    bd[14]     en-passant square (0..63), or 64 for none
    bd[15]     halfmove clock (not used by this encoding)
===============================================================================
"""

import numpy as np
from numba import njit

# ---- dimensions (import these everywhere; never hard-code the numbers) --------
N_SQ = 64
N_PT = 6            # piece types P,N,B,R,Q,K
N_COLOR = 2         # own / opponent
N_PIECE_FEATURES = N_COLOR * N_PT * N_SQ     # = 768 piece-placement inputs
N_EXTRA = 12       # 4 castling-rights + 8 en-passant-file inputs
N_FEATURES = N_PIECE_FEATURES + N_EXTRA      # = 780
MAX_PIECES = 32    # at most 32 pieces on the board
MAX_ACTIVE = 40    # buffer size: up to 32 pieces + 4 castling + 1 ep active

# Extra-feature index layout (all mover-relative), based at N_PIECE_FEATURES:
#   +0 own kingside castling      +1 own queenside castling
#   +2 opp kingside castling      +3 opp queenside castling
#   +4 .. +11  en-passant file a..h  (file is unchanged by the vertical flip)
_EX = N_PIECE_FEATURES   # 768

# ---- self-contained least-significant-bit index (De Bruijn, O(1)) -------------
# Kept local so this module has NO engine dependency and stays importable on the
# training machine with just numpy+numba. The table is GENERATED from the
# constant (not hand-typed) so it is correct by construction; the parity test
# double-checks it against python-chess.
_DEBRUIJN = np.uint64(0x03f79d71b4cb0a89)
_MASK64 = (1 << 64) - 1


def _build_debruijn_table(deb):
    tbl = [0] * 64
    for i in range(64):
        iso = 1 << i                                   # isolated bit i
        tbl[((iso * deb) & _MASK64) >> 58] = i
    return tbl


_DBTABLE = np.array(_build_debruijn_table(int(_DEBRUIJN)), dtype=np.int64)
_ONE = np.uint64(1)
_S58 = np.uint64(58)


@njit(cache=False, inline='always')
def _lsb(b):
    # b must be non-zero; returns the index of its least-significant set bit.
    iso = b & (~b + _ONE)                     # isolate lowest set bit
    return _DBTABLE[(iso * _DEBRUIJN) >> _S58]


@njit(cache=False)
def encode_board_np(bd, out_idx):
    """RUNTIME encoder. Fills out_idx (int32[MAX_ACTIVE]) with active feature
    indices for position `bd` and returns how many there are. Allocation-free so
    it is cheap to call from inside the search."""
    stm = bd[12]
    n = 0
    for pt in range(6):
        # white pieces (engine indices 0..5)
        wbb = bd[pt]
        while wbb:
            sq = _lsb(wbb)
            wbb &= wbb - _ONE
            if stm == 0:
                oo = 0; rs = sq          # white to move: white = own, no mirror
            else:
                oo = 1; rs = sq ^ 56     # black to move: white = opponent, mirror
            out_idx[n] = oo * 384 + pt * 64 + rs
            n += 1
        # black pieces (engine indices 6..11)
        bbb = bd[6 + pt]
        while bbb:
            sq = _lsb(bbb)
            bbb &= bbb - _ONE
            if stm == 0:
                oo = 1; rs = sq          # white to move: black = opponent
            else:
                oo = 0; rs = sq ^ 56     # black to move: black = own, mirror
            out_idx[n] = oo * 384 + pt * 64 + rs
            n += 1

    # ---- extra state: castling rights + en-passant file (mover-relative) ----
    cr = bd[13]                          # bits: 1=WK 2=WQ 4=BK 8=BQ
    if stm == 0:
        okc, oqc, pkc, pqc = cr & 1, cr & 2, cr & 4, cr & 8   # own=white
    else:
        okc, oqc, pkc, pqc = cr & 4, cr & 8, cr & 1, cr & 2   # own=black
    if okc:
        out_idx[n] = _EX + 0; n += 1
    if oqc:
        out_idx[n] = _EX + 1; n += 1
    if pkc:
        out_idx[n] = _EX + 2; n += 1
    if pqc:
        out_idx[n] = _EX + 3; n += 1
    ep = bd[14]                          # en-passant square, 64 = none
    if ep < 64:
        out_idx[n] = _EX + 4 + (ep & 7); n += 1
    return n


# ---- data-prep encoder (reads a python-chess Board; pure python) --------------
# piece_type from python-chess is 1..6 (PAWN..KING); map to our 0..5.
_PT_FROM_PYCHESS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def encode_pychess(board):
    """DATA-PREP encoder. Returns a python list of active feature indices for a
    `chess.Board`. Must produce the SAME index set as encode_board_np for the
    same position (guaranteed by the parity test)."""
    import chess
    stm_white = board.turn  # chess.WHITE is True
    idx = []
    for sq, piece in board.piece_map().items():
        pt = _PT_FROM_PYCHESS[piece.piece_type]
        is_white = piece.color  # True = white
        if stm_white:
            oo = 0 if is_white else 1
            rs = sq
        else:
            oo = 0 if (not is_white) else 1
            rs = sq ^ 56
        idx.append(oo * 384 + pt * 64 + rs)

    # ---- extra state: castling rights + en-passant file (mover-relative) ----
    wk = board.has_kingside_castling_rights(chess.WHITE)
    wq = board.has_queenside_castling_rights(chess.WHITE)
    bk = board.has_kingside_castling_rights(chess.BLACK)
    bq = board.has_queenside_castling_rights(chess.BLACK)
    if stm_white:
        okc, oqc, pkc, pqc = wk, wq, bk, bq
    else:
        okc, oqc, pkc, pqc = bk, bq, wk, wq
    if okc: idx.append(_EX + 0)
    if oqc: idx.append(_EX + 1)
    if pkc: idx.append(_EX + 2)
    if pqc: idx.append(_EX + 3)
    ep = board.ep_square           # square or None; file is flip-invariant
    if ep is not None:
        idx.append(_EX + 4 + (ep & 7))
    return idx


def encode_dense_pychess(board, dtype=np.float32):
    """Dense 780-vector for a python-chess Board (reference / training input if
    you prefer dense). Sparse indices are cheaper; both are provided."""
    v = np.zeros(N_FEATURES, dtype=dtype)
    for i in encode_pychess(board):
        v[i] = 1.0
    return v
