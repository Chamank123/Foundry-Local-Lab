"""Rescale a checkpoint's output by a constant, folded into out_w/out_b.

In pure minimax a positive constant on the evaluation changes nothing: move
ordering is invariant. This search is not pure minimax. Its pruning margins are
denominated in centipawns -- reverse futility prunes on `static - 85*depth >=
beta`, null-move and aspiration windows likewise -- and those constants were
tuned against an evaluation that returns true-ish centipawns.

The 8M nets are compressed: a linear fit of predicted on true centipawns over
the non-saturated band gives a slope near 0.59, so the net says ~59 cp where the
label says 100. Every fixed margin is therefore effectively ~1.7x wider than
intended, which prunes far more aggressively than the search was tuned for.
Undoing that compression is a pure output rescale: the function class, the
export layout and the runtime socket are untouched.

    python rescale_net.py in.npz out.npz 1.69
"""
import sys

import numpy as np

import nn_model


def rescale(src, dst, factor):
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["out_w"] = (arrays["out_w"].astype(np.float64) * factor).astype(np.float32)
    arrays["out_b"] = (arrays["out_b"].astype(np.float64) * factor).astype(np.float32)
    nn_model._atomic_savez(dst, arrays)
    # Reload through the validating loader so a bad file fails here, not in the engine.
    w = nn_model.load_weights(dst)
    bound = float(w["out_b"][0]) + float(np.maximum(w["out_w"], 0).sum())
    print(f"{src} x{factor} -> {dst}   upper output bound now {bound:+.1f} cp")


if __name__ == "__main__":
    rescale(sys.argv[1], sys.argv[2], float(sys.argv[3]))
