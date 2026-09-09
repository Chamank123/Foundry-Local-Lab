"""Reconstruct mover-normalised boards from the existing sparse feature contract."""
import numpy as np
from numba import njit


@njit(cache=False)
def board_from_features(ids, count):
    bd = np.zeros(16, np.uint64)
    bd[14] = 64
    for j in range(count):
        i = int(ids[j])
        if i < 768:
            side = i // 384
            piece = (i % 384) // 64
            sq = i % 64
            bd[side * 6 + piece] |= np.uint64(1) << np.uint64(sq)
        elif i < 772:
            bd[13] |= np.uint64(1) << np.uint64(i - 768)
        else:
            bd[14] = np.uint64(40 + i - 772)
    return bd


def baseline_values(idx, cnt):
    from search_numba import evaluate_handcrafted

    @njit(cache=False)
    def batch(x, n):
        result = np.empty(len(n), np.float32)
        for i in range(len(n)):
            result[i] = evaluate_handcrafted(board_from_features(x[i], n[i]))
        return result

    return batch(idx, cnt)
