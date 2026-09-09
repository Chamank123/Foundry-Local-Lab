"""Versioned NN input policy; never changes the engine's legal-move state.

legal-ep-v1 suppresses the EP input unless the mover has a LEGAL EP capture.
The original nn_encode module remains available for reproducing legacy results.
"""
import numpy as np
from numba import njit
from bb_numba import make, is_attacked, bsf, PAWN_ATT
from nn_encode import encode_board_np, MAX_ACTIVE
from nn_runtime import nn_forward


@njit(cache=False)
def has_legal_ep(bd):
    ep = int(bd[14])
    if ep == 64:
        return False
    white = bd[12] == 0
    if ep // 8 != (5 if white else 2):
        return False
    base = 0 if white else 6
    opponent = 6 if white else 0
    captured = ep - 8 if white else ep + 8
    bit = np.uint64(1) << np.uint64(ep)
    occupied = np.uint64(0)
    for i in range(12):
        occupied |= bd[i]
    if occupied & bit or not (bd[opponent] & (np.uint64(1) << np.uint64(captured))):
        return False
    pawns = bd[base]
    while pawns:
        frm = bsf(pawns)
        pawns &= pawns - np.uint64(1)
        if PAWN_ATT[0 if white else 1, frm] & bit:
            move = frm | (ep << 6) | (2 << 15)
            child = make(bd, move)
            if not is_attacked(child, bsf(child[base + 5]), not white):
                return True
    return False


@njit(cache=False)
def encode_legal_ep(bd, out):
    n = encode_board_np(bd, out)
    # The existing encoder appends the EP file feature last.
    if int(bd[14]) != 64 and not has_legal_ep(bd):
        if n > 0 and out[n - 1] >= 772:
            n -= 1
    return n


@njit(cache=False)
def score_nn(bd, weights, legal_ep=True):
    ids = np.empty(MAX_ACTIVE, np.int32)
    n = encode_legal_ep(bd, ids) if legal_ep else encode_board_np(bd, ids)
    return nn_forward(ids, n, weights[0], weights[1], weights[2],
                      weights[3], weights[4], weights[5])
