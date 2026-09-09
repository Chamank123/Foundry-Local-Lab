"""Network architecture + weight I/O  (AI Chessathon).

This file FREEZES the architecture so training (this file) and the in-engine
runtime (nn_runtime.py) can never drift apart. The numba forward pass mirrors
`NNUE.forward` exactly; the parity test enforces it.

Architecture (small on purpose — one CPU core at run time):
    780  --Linear-->  256   (the "accumulator")
                      clipped ReLU  (clamp to [0, 1])
         --Linear-->   32
                      clipped ReLU
         --Linear-->    1    -> centipawns, side-to-move relative

Weight file (.npz) is stored in RUNTIME-READY layout:
    acc_w : (780, 256) float32   accumulator weights, TRANSPOSED so the runtime
                                  adds contiguous rows per active feature
    acc_b : (256,)     float32
    l1_w  : (32, 256)  float32
    l1_b  : (32,)      float32
    out_w : (1, 32)    float32
    out_b : (1,)       float32
"""

import os
import tempfile

import numpy as np

from nn_encode import N_FEATURES

H0 = 256          # accumulator width
H1 = 32           # hidden width
FORMAT_VERSION = 2
# --- if you change H0/H1, nothing else needs editing: the runtime reads the
#     shapes from the weight file. Keep them small for CPU speed.

WEIGHT_KEYS = ("acc_w", "acc_b", "l1_w", "l1_b", "out_w", "out_b")


# ----------------------------------------------------------------------------
# PyTorch model (used on the training machine: Colab / Kaggle T4).
# Imported lazily so this module is usable for weight I/O without torch.
# ----------------------------------------------------------------------------
def build_model(out_scale=1.0, h0=None, h1=None):
    """Build the network. `out_scale` is a FIXED (non-trainable) multiplier on
    the output head.

    With out_scale=1 the head must itself grow to tens of centipawns per unit to
    express a real evaluation, while the training gradient reaching it is divided
    by the loss scale K=400. The net's cheapest route to magnitude is then to
    saturate its clipped [0,1] units, which is what the pilot checkpoints do.
    Factoring a fixed constant out of the head keeps the trainable weights O(1)
    and leaves the FUNCTION CLASS UNCHANGED: export folds the constant into
    out_w/out_b, so the exported file and the runtime socket are byte-identical
    in layout and the model still outputs raw centipawns.
    """
    import torch
    import torch.nn as nn

    h0 = H0 if h0 is None else int(h0)
    h1 = H1 if h1 is None else int(h1)

    class NNUE(nn.Module):
        def __init__(self, n_features=N_FEATURES, h0=H0, h1=H1, out_scale=1.0):
            super().__init__()
            self.acc = nn.Linear(n_features, h0)   # accumulator
            self.l1 = nn.Linear(h0, h1)
            self.out = nn.Linear(h1, 1)
            self.out_scale = float(out_scale)

        def forward(self, x):
            # x: (batch, 780) dense 0/1 features
            a = torch.clamp(self.acc(x), 0.0, 1.0)   # clipped ReLU
            h = torch.clamp(self.l1(a), 0.0, 1.0)
            # centipawns, mover-relative
            return self.out(h).squeeze(-1) * self.out_scale

    return NNUE(h0=h0, h1=h1, out_scale=out_scale)


def _atomic_savez(path, arrays):
    """Write an npz in one atomic same-directory replacement."""
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".nn_weights_", suffix=".npz",
                                     dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def export_weights(model, path, metadata=None):
    """Save a trained torch model to a runtime-ready .npz."""
    sd = model.state_dict()
    # Fold the fixed output multiplier into the exported head so the runtime
    # stays a plain linear layer and keeps emitting raw centipawns.
    out_scale = float(getattr(model, "out_scale", 1.0))
    arrays = {
        # ascontiguousarray is REQUIRED, not cosmetic: .T yields an F-ordered
        # view, np.savez records that order, and np.load hands it back F-ordered.
        # nn_forward indexes acc_w[idx[k]] expecting a contiguous row, and numba
        # types an F-ordered array differently from a C-ordered one -- so an
        # F-ordered acc_w both defeats the row-access design and forces a fresh
        # compilation of the whole search on first use, on the clock.
        "acc_w": np.ascontiguousarray(
            sd["acc.weight"].detach().cpu().numpy().T, dtype=np.float32),  # (780,256)
        "acc_b": sd["acc.bias"].detach().cpu().numpy().astype(np.float32),
        "l1_w":  sd["l1.weight"].detach().cpu().numpy().astype(np.float32),      # (32,256)
        "l1_b":  sd["l1.bias"].detach().cpu().numpy().astype(np.float32),
        "out_w": (sd["out.weight"].detach().cpu().numpy() * out_scale).astype(np.float32),
        "out_b": (sd["out.bias"].detach().cpu().numpy() * out_scale).astype(np.float32),
    }
    arrays["format_version"] = np.array(FORMAT_VERSION, np.int32)
    arrays["n_features"] = np.array(N_FEATURES, np.int32)
    if metadata:
        for key, value in metadata.items():
            if key in arrays:
                raise ValueError(f"metadata key collides with a weight key: {key}")
            arrays[f"meta_{key}"] = np.asarray(value)
    _atomic_savez(path, arrays)
    return path


def random_weights(path, seed=0, scale=0.05):
    """Create a small RANDOM weight file with numpy only (no torch needed).
    Lets the whole runtime + engine pipeline be exercised before any training.
    An untrained net plays weak-but-legal chess — that is the point: it proves
    the plumbing end to end."""
    rng = np.random.default_rng(seed)
    arrays = {
        "acc_w": np.ascontiguousarray(
            rng.standard_normal((N_FEATURES, H0)) * scale, dtype=np.float32),
        "acc_b": np.zeros(H0, dtype=np.float32),
        "l1_w":  (rng.standard_normal((H1, H0)) * scale).astype(np.float32),
        "l1_b":  np.zeros(H1, dtype=np.float32),
        "out_w": (rng.standard_normal((1, H1)) * scale * 20).astype(np.float32),
        "out_b": np.zeros(1, dtype=np.float32),
    }
    arrays["format_version"] = np.array(FORMAT_VERSION, np.int32)
    arrays["n_features"] = np.array(N_FEATURES, np.int32)
    _atomic_savez(path, arrays)
    return path


def load_weights(path):
    """Load and validate an .npz into runtime-ready float32 arrays."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in WEIGHT_KEYS if k not in z]
        if missing:
            raise ValueError(f"weight file is missing arrays: {missing}")
        # Force C order on load as well, so weight files written before the
        # export fix above still give the runtime contiguous rows and one
        # stable numba type.
        w = {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in WEIGHT_KEYS}
        if "format_version" in z and int(z["format_version"]) != FORMAT_VERSION:
            raise ValueError(
                f"weight format {int(z['format_version'])} is unsupported; "
                f"expected {FORMAT_VERSION}")
        if "n_features" in z and int(z["n_features"]) != N_FEATURES:
            raise ValueError(f"weight file has {int(z['n_features'])} features; expected {N_FEATURES}")

    if w["acc_w"].ndim != 2 or w["acc_w"].shape[0] != N_FEATURES:
        raise ValueError(f"bad acc_w shape: {w['acc_w'].shape}")
    h0 = w["acc_w"].shape[1]
    if w["acc_b"].shape != (h0,):
        raise ValueError(f"bad acc_b shape: {w['acc_b'].shape}")
    if w["l1_w"].ndim != 2 or w["l1_w"].shape[1] != h0:
        raise ValueError(f"bad l1_w shape: {w['l1_w'].shape}")
    h1 = w["l1_w"].shape[0]
    expected = {"l1_b": (h1,), "out_w": (1, h1), "out_b": (1,)}
    for key, shape in expected.items():
        if w[key].shape != shape:
            raise ValueError(f"bad {key} shape: {w[key].shape}; expected {shape}")
    for key, value in w.items():
        if not np.isfinite(value).all():
            raise ValueError(f"weight file contains non-finite values in {key}")
    return w
