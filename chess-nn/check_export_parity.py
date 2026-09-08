"""Export/runtime parity for an ACTUAL TRAINED checkpoint (handoff section 8).

The shipped tests only exercise random weights. This compares a trained
checkpoint's PyTorch predictions with its exported NumPy and numba predictions
on HELD-OUT pilot positions (the test split), and checks the raw float output
and the engine-facing clamped-integer output separately.

    python check_export_parity.py pilot_net.npz --shards pilot_shards
"""
import argparse
import sys

import numpy as np

import nn_model
import nn_runtime as rt
from nn_encode import MAX_ACTIVE, N_FEATURES
from train_nnue import _load_shards, split_indices


def torch_model_from_npz(weights):
    """Rebuild the PyTorch model from the exported runtime layout.

    export_weights() only casts to float32 and transposes acc_w, so this is the
    exact inverse: the reconstructed module holds the trained parameters
    bit-for-bit. Any disagreement below is a real export/runtime defect.
    """
    import torch
    model = nn_model.build_model()
    with torch.no_grad():
        sd = model.state_dict()
        sd["acc.weight"].copy_(torch.from_numpy(weights["acc_w"].T.copy()))
        sd["acc.bias"].copy_(torch.from_numpy(weights["acc_b"]))
        sd["l1.weight"].copy_(torch.from_numpy(weights["l1_w"]))
        sd["l1.bias"].copy_(torch.from_numpy(weights["l1_b"]))
        sd["out.weight"].copy_(torch.from_numpy(weights["out_w"]))
        sd["out.bias"].copy_(torch.from_numpy(weights["out_b"]))
    model.eval()
    return model


def engine_int(cp):
    """Mirror nn_eval's tail: clamp to +-EVAL_CLAMP, then truncate to int."""
    cp = min(rt.EVAL_CLAMP, max(-rt.EVAL_CLAMP, float(cp)))
    return int(cp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("--shards", default="pilot_shards")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    args = parser.parse_args()

    import torch

    w = nn_model.load_weights(args.weights)
    model = torch_model_from_npz(w)

    idx, cnt, y, _wdl, keys, groups = _load_shards(args.shards)
    split_keys = np.where(groups != 0, groups, keys).astype(np.uint64, copy=False)
    _tr, _val, test_rows = split_indices(split_keys)
    rows = test_rows[:args.limit]
    print(f"parity on {len(rows)} HELD-OUT (test-split) pilot positions")

    # dense batch for torch, on CPU, from the same sparse rows the runtime uses
    dense = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
    for r, row in enumerate(rows):
        n = int(cnt[row])
        dense[r, idx[row, :n].astype(np.int64)] = 1.0
    with torch.no_grad():
        torch_cp = model(torch.from_numpy(dense)).numpy().astype(np.float64)

    buf = np.empty(MAX_ACTIVE, dtype=np.int32)
    numba_cp = np.empty(len(rows), dtype=np.float64)
    numpy_cp = np.empty(len(rows), dtype=np.float64)
    for r, row in enumerate(rows):
        n = int(cnt[row])
        buf[:n] = idx[row, :n]
        numba_cp[r] = rt.nn_forward(buf, n, w["acc_w"], w["acc_b"], w["l1_w"],
                                    w["l1_b"], w["out_w"], w["out_b"])
        numpy_cp[r] = rt.nn_forward_numpy(buf, n, w)

    failures = []
    print("\nRAW FLOAT output (centipawns):")
    for name, a, b in (("numba  vs numpy ", numba_cp, numpy_cp),
                       ("numba  vs torch ", numba_cp, torch_cp),
                       ("numpy  vs torch ", numpy_cp, torch_cp)):
        worst = float(np.max(np.abs(a - b)))
        ok = worst < args.tolerance
        print(f"  {name}: max |diff| = {worst:.3e} cp  [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            failures.append(name.strip())

    print("\nENGINE-FACING clamped int output:")
    ints = {name: np.array([engine_int(v) for v in arr])
            for name, arr in (("numba", numba_cp), ("numpy", numpy_cp), ("torch", torch_cp))}
    for a, b in (("numba", "numpy"), ("numba", "torch"), ("numpy", "torch")):
        diff = np.abs(ints[a] - ints[b])
        mismatched = int((diff != 0).sum())
        worst = int(diff.max()) if len(diff) else 0
        # A +-1 cp disagreement is truncation landing either side of an integer
        # boundary, not an export defect; anything larger is.
        ok = worst <= 1
        print(f"  {a} vs {b}: {mismatched}/{len(diff)} rows differ, "
              f"max |diff| = {worst} cp  [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            failures.append(f"int {a} vs {b}")

    print(f"\npredicted cp on held-out rows: min={numba_cp.min():.1f} "
          f"max={numba_cp.max():.1f} mean={numba_cp.mean():.1f}")
    print(f"label cp on the same rows:     min={y[rows].min():.1f} "
          f"max={y[rows].max():.1f} mean={y[rows].mean():.1f}")
    saturated = int((np.abs(numba_cp) >= rt.EVAL_CLAMP).sum())
    print(f"rows hitting the +-{rt.EVAL_CLAMP} cp clamp: {saturated}")

    if failures:
        print("\nPARITY FAILED: " + ", ".join(failures))
        return 1
    print("\nEXPORT/RUNTIME PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
