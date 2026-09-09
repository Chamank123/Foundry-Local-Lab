"""In-engine NN runtime  (AI Chessathon)  --  step 2.

Runs the trained network inside the engine to score a position, returning
centipawns from the side-to-move's perspective (drop-in for the hand-crafted
evaluate()). This is the NON-INCREMENTAL version: it rebuilds the accumulator
from scratch every call. It is correct and simple, which is what you want first
-- it gives a real in-engine test harness before anyone touches the fast
incremental path (that is a later, gated, skippable optimisation).

Weights are passed as ARGUMENTS to the forward pass on purpose: numba freezes
the *contents* of module-global arrays at compile time, so a global-weights
design would silently keep whatever was loaded at the first call. Passing them
in keeps the net swappable and fully testable. See README_NN.md for the two
ways to wire this into search_numba.
"""

import numpy as np
from numba import njit

from nn_encode import encode_board_np, MAX_ACTIVE, N_FEATURES

# Output is clamped below the search's mate scores so a wild untrained net can
# never be confused with a forced mate.
# The training labels are clamped to this range. Scores outside it carry no
# additional learned meaning and must remain comfortably below mate scores.
EVAL_CLAMP = 6000


@njit(cache=False)
def nn_forward(idx, cnt, acc_w, acc_b, l1_w, l1_b, out_w, out_b):
    """Sparse forward pass. idx[:cnt] are the active feature indices.
    Returns centipawns (float), side-to-move relative."""
    h0 = acc_b.shape[0]
    h1 = l1_b.shape[0]

    # accumulator: bias + sum of the active feature columns (rows of acc_w)
    acc = acc_b.copy()
    for k in range(cnt):
        row = acc_w[idx[k]]              # contiguous length-h0 row
        for j in range(h0):
            acc[j] += row[j]
    for j in range(h0):                  # clipped ReLU
        v = acc[j]
        acc[j] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # hidden layer
    hid = l1_b.copy()
    for j in range(h1):
        wr = l1_w[j]
        s = np.float32(0.0)
        for k in range(h0):
            s += wr[k] * acc[k]
        v = hid[j] + s
        hid[j] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    # output
    o = np.float32(out_b[0])
    orow = out_w[0]
    for j in range(h1):
        o += orow[j] * hid[j]
    return o


@njit(cache=False)
def nn_eval(bd, acc_w, acc_b, l1_w, l1_b, out_w, out_b):
    """Engine-facing evaluation: encode `bd`, run the net, clamp to int cp.
    Signature-compatible with a weights-passing search integration."""
    idx = np.empty(MAX_ACTIVE, dtype=np.int32)
    cnt = encode_board_np(bd, idx)
    cp = nn_forward(idx, cnt, acc_w, acc_b, l1_w, l1_b, out_w, out_b)
    if cp > EVAL_CLAMP:
        cp = EVAL_CLAMP
    elif cp < -EVAL_CLAMP:
        cp = -EVAL_CLAMP
    return int(cp)


# ---- numpy reference (dense) -- parity target, NOT for runtime ---------------
def nn_forward_numpy(idx, cnt, w):
    acc = w["acc_b"].astype(np.float64).copy()
    for k in range(cnt):
        acc += w["acc_w"][idx[k]]
    acc = np.clip(acc, 0.0, 1.0)
    hid = np.clip(w["l1_b"].astype(np.float64) + w["l1_w"].astype(np.float64) @ acc,
                  0.0, 1.0)
    return float(w["out_b"][0] + w["out_w"][0].astype(np.float64) @ hid)


# ---- convenience: a loaded net you can call as nn_eval_bd(bd) ----------------
# Holds weights in a dict (python side) and passes them into the njit forward on
# each call, so swapping weights at runtime Just Works (no recompile, no freeze).
_WEIGHTS = None


def load(path):
    """Load a weight .npz for use via nn_eval_bd()."""
    global _WEIGHTS
    with np.load(path, allow_pickle=False) as data:
        if "meta_evaluator_kind" in data and str(data["meta_evaluator_kind"].item()) != "full":
            raise ValueError("This checkpoint predicts a correction, not a full evaluation. Use search_numba.load_nn().")
    import nn_model
    _WEIGHTS = nn_model.load_weights(path)
    return _WEIGHTS


def nn_eval_bd(bd):
    """Python-side evaluation using the loaded weights. Handy for tests and for
    a simple (non-search-embedded) integration. Returns int centipawns."""
    if _WEIGHTS is None:
        raise RuntimeError("call load(path) first")
    w = _WEIGHTS
    return nn_eval(bd, w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"],
                   w["out_w"], w["out_b"])
