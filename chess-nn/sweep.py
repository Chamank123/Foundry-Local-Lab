"""Validation-only hyperparameter sweep for the pilot net.

The handoff calls 30 epochs "a starting setting, not a guaranteed optimum", and
section 6 forbids using the test split to choose between repeated experiments.
train() always prints a test metric at the end of a run, so this driver captures
each run's stdout and reports ONLY the validation numbers. The test split stays
unread until a single configuration has been chosen on validation alone.

Architecture, encoding, scale and loss are left exactly as the handoff freezes
them; only optimisation settings (epochs, batch size, learning rate) vary.
"""
import contextlib
import io
import itertools
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

import train_nnue


def run(epochs, bs, lr, seed=0, shards="pilot_shards", lr_schedule="none"):
    """Run one configuration; return (best_val_rmse, best_epoch, seconds)."""
    buffer = io.StringIO()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sweep_") as temporary:
        out = os.path.join(temporary, "net.npz")
        with contextlib.redirect_stdout(buffer):
            train_nnue.train(shards, out, epochs=epochs, bs=bs, lr=lr, seed=seed,
                             lr_schedule=lr_schedule)
    elapsed = time.perf_counter() - started
    text = buffer.getvalue()
    values = [float(v) for v in re.findall(r"validation expected-score RMSE=([0-9.]+)", text)]
    best = min(values)
    return best, values.index(best) + 1, elapsed


def constant_baseline(shards="pilot_shards"):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        idx, cnt, y, wdl, keys, groups = train_nnue._load_shards(shards)
    tr, val, _test = train_nnue.split_indices(keys)
    target = 1.0 / (1.0 + np.exp(-y / train_nnue.DEFAULT_SCALE))
    return float(np.sqrt(np.mean((target[val] - target[tr].mean()) ** 2)))


if __name__ == "__main__":
    grid = [
        # (epochs, batch size, learning rate)
        (30,   8192, 1e-3),    # the handoff's prescribed baseline
        (30,   1024, 1e-3),
        (100,  1024, 1e-3),
        (100,  1024, 3e-3),
        (100,   256, 1e-3),
        (300,  8192, 1e-2),
        (200,  1024, 3e-3),
    ]
    reference = constant_baseline()
    print(f"constant-predictor validation RMSE (predict train mean): {reference:.5f}")
    print(f"{'epochs':>7} {'batch':>6} {'lr':>7} | {'best val RMSE':>13} {'@epoch':>7} "
          f"{'vs constant':>12} {'secs':>6}")
    results = []
    for epochs, bs, lr in grid:
        best, at, secs = run(epochs, bs, lr)
        gain = (reference - best) / reference * 100.0
        results.append({"epochs": epochs, "batch_size": bs, "learning_rate": lr,
                        "best_val_rmse": best, "best_epoch": at, "seconds": secs})
        print(f"{epochs:7d} {bs:6d} {lr:7.4f} | {best:13.5f} {at:7d} "
              f"{gain:11.1f}% {secs:6.0f}")
        sys.stdout.flush()
    results.sort(key=lambda r: r["best_val_rmse"])
    print("\nbest on validation:", json.dumps(results[0], indent=2))
    with open("sweep_results.json", "w", encoding="utf-8") as fh:
        json.dump({"constant_baseline_val_rmse": reference, "results": results},
                  fh, indent=2)
