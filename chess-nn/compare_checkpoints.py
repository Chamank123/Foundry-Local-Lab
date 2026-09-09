"""Compare checkpoints on ONE fixed validation distribution, plus material.

Validation RMSE printed by train() is computed on that run's own dataset split,
so numbers from the 100k pilot and the 8M snapshot are NOT comparable -- the
pilot's validation set is a 1,031-row subset of the 8M validation set. This
evaluates every checkpoint on the SAME rows, and pairs that with the controlled
material probe, because REPORT2 section 3 found the two move independently.

    python compare_checkpoints.py --shards overnight_shards_8m a.npz b.npz
"""
import argparse

import numpy as np

import nn_model
import nn_runtime as rt
from nn_encode import MAX_ACTIVE, N_FEATURES
from train_nnue import _load_shards, split_indices, DEFAULT_SCALE


def predict(w, idx, cnt, rows, chunk=8192):
    """Dense forward in numpy, chunked so 80k rows do not build a huge matrix."""
    acc_w = w["acc_w"].astype(np.float32)
    out = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), chunk):
        batch = rows[start:start + chunk]
        dense = np.zeros((len(batch), N_FEATURES), dtype=np.float32)
        for r, row in enumerate(batch):
            n = int(cnt[row])
            dense[r, idx[row, :n].astype(np.int64)] = 1.0
        a = np.clip(dense @ acc_w + w["acc_b"], 0, 1)
        h = np.clip(a @ w["l1_w"].T + w["l1_b"], 0, 1)
        out[start:start + len(batch)] = h @ w["out_w"].ravel() + float(w["out_b"][0])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--shards", default="overnight_shards_8m")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()

    idx, cnt, y, _wdl, keys, groups = _load_shards(args.shards)
    split_keys = np.where(groups != 0, groups, keys).astype(np.uint64, copy=False)
    train, val, test = split_indices(split_keys)
    rows = (val if args.split == "val" else test)[:args.limit]
    label = y[rows].astype(np.float64)
    to_score = lambda v: 1.0 / (1.0 + np.exp(-v / DEFAULT_SCALE))
    target = to_score(label)
    print(f"common yardstick: {len(rows):,} rows from the {args.split} split of "
          f"{args.shards}")
    print(f"trivial baseline (predict the mean target): "
          f"RMSE {np.sqrt(np.mean((target - target.mean()) ** 2)):.5f}\n")

    print(f"{'checkpoint':32} {'RMSE':>8} {'Pearson':>8} {'Spearman':>9} "
          f"{'sign':>6} {'medAE':>7} {'min cp':>8} {'max cp':>8}")
    for path in args.checkpoints:
        w = nn_model.load_weights(path)
        pred = predict(w, idx, cnt, rows)
        rmse = float(np.sqrt(np.mean((to_score(pred) - target) ** 2)))
        pearson = float(np.corrcoef(pred, label)[0, 1])
        spearman = float(np.corrcoef(pred.argsort().argsort(),
                                     label.argsort().argsort())[0, 1])
        decisive = np.abs(label) >= 50
        sign = float((np.sign(pred[decisive]) == np.sign(label[decisive])).mean())
        med = float(np.median(np.abs(pred - label)))
        print(f"{path:32} {rmse:8.5f} {pearson:8.4f} {spearman:9.4f} "
              f"{sign * 100:5.1f}% {med:7.1f} {pred.min():8.1f} {pred.max():8.1f}")


if __name__ == "__main__":
    main()
