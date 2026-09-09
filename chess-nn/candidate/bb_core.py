"""Bitboard chess core for the Chessathon numba engine.

Correctness-first implementation. Written in plain Python so it can be
perft-verified against python-chess, then JIT-compiled with numba (the only
compiled-speed path the rules allow) once the logic is proven.

Square numbering matches python-chess: 0 = a1, 1 = b1, ..., 63 = h8.
file = sq & 7, rank = sq >> 3.

Board state is a length-16 array of unsigned 64-bit ints:
  [0..5]   White P,N,B,R,Q,K bitboards
  [6..11]  Black P,N,B,R,Q,K bitboards
  [12]     side to move (0 = white, 1 = black)
  [13]     castling rights bitmask: 1=WK 2=WQ 4=BK 8=BQ
  [14]     en-passant target square (0..63), or 64 if none
  [15]     halfmove clock
"""

import numpy as np

# Piece indices into the board array.
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
STM, CR, EP, HM = 12, 13, 14, 15
NO_EP = 64

# Castling right bits.
C_WK, C_WQ, C_BK, C_BQ = 1, 2, 4, 8

MASK64 = (1 << 64) - 1

FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
RANK_1 = 0x00000000000000FF
RANK_8 = 0xFF00000000000000


def _sq(file, rank):
    return rank * 8 + file


# ---- precomputed leaper attack tables -----------------------------------

KNIGHT_ATTACKS = [0] * 64
KING_ATTACKS = [0] * 64
# Pawn attacks indexed [color][square]; color 0 = white, 1 = black.
PAWN_ATTACKS = [[0] * 64, [0] * 64]

_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2),
                  (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1),
                (-1, 0), (-1, -1), (0, -1), (1, -1)]

for s in range(64):
    f, r = s & 7, s >> 3
    kb = 0
    for df, dr in _KNIGHT_DELTAS:
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            kb |= 1 << _sq(nf, nr)
    KNIGHT_ATTACKS[s] = kb
    gb = 0
    for df, dr in _KING_DELTAS:
        nf, nr = f + df, r + dr
        if 0 <= nf < 8 and 0 <= nr < 8:
            gb |= 1 << _sq(nf, nr)
    KING_ATTACKS[s] = gb
    # White pawn attacks up-left / up-right.
    wb = 0
    if f > 0 and r < 7:
        wb |= 1 << _sq(f - 1, r + 1)
    if f < 7 and r < 7:
        wb |= 1 << _sq(f + 1, r + 1)
    PAWN_ATTACKS[0][s] = wb
    # Black pawn attacks down-left / down-right.
    bb = 0
    if f > 0 and r > 0:
        bb |= 1 << _sq(f - 1, r - 1)
    if f < 7 and r > 0:
        bb |= 1 << _sq(f + 1, r - 1)
    PAWN_ATTACKS[1][s] = bb


# ---- ray tables for classical sliding-attack generation -----------------
# 8 directions. Positive rays (toward higher square index) use LSB scan of
# blockers; negative rays use MSB scan.
# dir order: 0=N(+8) 1=E(+1) 2=NE(+9) 3=NW(+7) | 4=S(-8) 5=W(-1) 6=SE(-7) 7=SW(-9)
RAYS = [[0] * 64 for _ in range(8)]
_DIR_DF = [0, 1, 1, -1, 0, -1, 1, -1]
_DIR_DR = [1, 0, 1, 1, -1, 0, -1, -1]

for d in range(8):
    df, dr = _DIR_DF[d], _DIR_DR[d]
    for s in range(64):
        f, r = s & 7, s >> 3
        bits = 0
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            bits |= 1 << _sq(nf, nr)
            nf += df
            nr += dr
        RAYS[d][s] = bits

POSITIVE_DIRS = (0, 1, 2, 3)
NEGATIVE_DIRS = (4, 5, 6, 7)
ROOK_DIRS = (0, 1, 4, 5)
BISHOP_DIRS = (2, 3, 6, 7)


# ---- bit helpers (plain Python versions) --------------------------------

def popcount(x):
    return bin(x).count("1")


def lsb(x):
    return (x & -x).bit_length() - 1


def msb(x):
    return x.bit_length() - 1


def pop_lsb(x):
    b = x & -x
    return b.bit_length() - 1, x ^ b


# ---- sliding attacks via classical ray method ---------------------------

def ray_attacks_dir(sq, occ, d):
    """Attacks along a single direction from sq, blocked by occ."""
    attacks = RAYS[d][sq]
    blockers = attacks & occ
    if blockers:
        if d in POSITIVE_DIRS:
            b = lsb(blockers)
        else:
            b = msb(blockers)
        attacks ^= RAYS[d][b]
    return attacks


def bishop_attacks(sq, occ):
    return (ray_attacks_dir(sq, occ, 2) | ray_attacks_dir(sq, occ, 3)
            | ray_attacks_dir(sq, occ, 6) | ray_attacks_dir(sq, occ, 7))


def rook_attacks(sq, occ):
    return (ray_attacks_dir(sq, occ, 0) | ray_attacks_dir(sq, occ, 1)
            | ray_attacks_dir(sq, occ, 4) | ray_attacks_dir(sq, occ, 5))


def queen_attacks(sq, occ):
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ)


# ---- occupancy helpers --------------------------------------------------

def white_occ(bd):
    return bd[WP] | bd[WN] | bd[WB] | bd[WR] | bd[WQ] | bd[WK]


def black_occ(bd):
    return bd[BP] | bd[BN] | bd[BB] | bd[BR] | bd[BQ] | bd[BK]


def all_occ(bd):
    return white_occ(bd) | black_occ(bd)


def is_square_attacked(bd, sq, by_white):
    """True if `sq` is attacked by the given side."""
    occ = all_occ(bd)
    if by_white:
        if PAWN_ATTACKS[1][sq] & bd[WP]:   # black-pawn-attack pattern hits WP
            return True
        if KNIGHT_ATTACKS[sq] & bd[WN]:
            return True
        if KING_ATTACKS[sq] & bd[WK]:
            return True
        if bishop_attacks(sq, occ) & (bd[WB] | bd[WQ]):
            return True
        if rook_attacks(sq, occ) & (bd[WR] | bd[WQ]):
            return True
    else:
        if PAWN_ATTACKS[0][sq] & bd[BP]:
            return True
        if KNIGHT_ATTACKS[sq] & bd[BN]:
            return True
        if KING_ATTACKS[sq] & bd[BK]:
            return True
        if bishop_attacks(sq, occ) & (bd[BB] | bd[BQ]):
            return True
        if rook_attacks(sq, occ) & (bd[BR] | bd[BQ]):
            return True
    return False


# ---- FEN -> board array (for testing/interfacing) -----------------------

_PIECE_TO_IDX = {
    'P': WP, 'N': WN, 'B': WB, 'R': WR, 'Q': WQ, 'K': WK,
    'p': BP, 'n': BN, 'b': BB, 'r': BR, 'q': BQ, 'k': BK,
}


def board_from_fen(fen):
    bd = [0] * 16
    parts = fen.split()
    rows = parts[0].split("/")
    for i, row in enumerate(rows):
        rank = 7 - i
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                bd[_PIECE_TO_IDX[ch]] |= 1 << _sq(file, rank)
                file += 1
    bd[STM] = 0 if parts[1] == 'w' else 1
    cr = 0
    castle = parts[2]
    if 'K' in castle:
        cr |= C_WK
    if 'Q' in castle:
        cr |= C_WQ
    if 'k' in castle:
        cr |= C_BK
    if 'q' in castle:
        cr |= C_BQ
    bd[CR] = cr
    if parts[3] == '-':
        bd[EP] = NO_EP
    else:
        ef = ord(parts[3][0]) - ord('a')
        er = int(parts[3][1]) - 1
        bd[EP] = _sq(ef, er)
    bd[HM] = int(parts[4]) if len(parts) > 4 else 0
    return bd
