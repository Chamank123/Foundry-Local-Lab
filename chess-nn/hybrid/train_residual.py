"""Train a compact correction to the preserved handcrafted evaluator.

The LOSS compares sigmoid((HC + correction) / 400) to sigmoid(teacher / 400).
The NPZ contains only the correction and is explicitly marked as residual.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))
import numpy as np
import torch
from torch.nn import functional as F
import nn_model
from train_nnue import _load_shards, _dense_batch, split_indices
from residual_data import baseline_values, board_from_features
from nn_adapter import has_legal_ep


def main(a):
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    idx, cnt, y, wdl, keys, groups = _load_shards(a.shards)
    if np.any(wdl != -1) or np.any(groups != 0):
        raise ValueError("This experiment requires the supplied eval-only snapshot")
    if len(np.unique(keys)) != len(keys):
        raise ValueError("Duplicate position keys")
    # Check the alleged training EP convention rather than infer it from frequency.
    ep_rows = np.flatnonzero(np.any(idx >= 772, axis=1))
    bad = [int(i) for i in ep_rows if not has_legal_ep(board_from_features(idx[i], cnt[i]))]
    print(f"EP audit: {len(ep_rows):,} flagged rows, {len(bad)} without a legal capture", flush=True)
    if bad:
        raise ValueError(f"Training EP policy differs; do not silently change keys/splits: {bad[:10]}")
    started = time.perf_counter()
    baseline = baseline_values(idx, cnt)
    print(f"Computed {len(y):,} handcrafted baselines in {time.perf_counter()-started:.1f}s", flush=True)
    tr, va, te = split_indices(keys)
    print(f"Split: train {len(tr):,}, validation {len(va):,}, test {len(te):,}", flush=True)
    model = nn_model.build_model(out_scale=400, h0=a.h0, h1=a.h1)
    torch.nn.init.zeros_(model.out.weight)
    torch.nn.init.zeros_(model.out.bias)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    device = torch.device("cpu")
    targets = torch.sigmoid(torch.from_numpy(y) / 400)
    hc = torch.from_numpy(baseline)

    def measure(rows, active_model=True):
        model.eval()
        squared = 0.
        with torch.no_grad():
            for start in range(0, len(rows), a.batch):
                r = rows[start:start+a.batch]
                total = hc[r]
                if active_model:
                    total = total + model(_dense_batch(idx, cnt, r, device))
                squared += F.mse_loss(torch.sigmoid(total / 400), targets[r], reduction="sum").item()
        return (squared / len(rows)) ** .5

    hc_rmse = measure(va, False)
    print(f"Handcrafted validation RMSE: {hc_rmse:.6f}; zero-initialised correction matches it", flush=True)
    best, best_epoch, history = hc_rmse, 0, []
    source_hash = hashlib.sha256((ROOT / "engine/search_numba.py").read_bytes()).hexdigest()

    def export(epoch, val):
        nn_model.export_weights(model, a.out, metadata={
            "evaluator_kind": np.asarray("hc_residual_v1"),
            "ep_policy": np.asarray("legal-ep-v1"), "output_scale": np.float32(400),
            "scale": np.float32(400), "lambda": np.float32(1),
            "baseline_source_sha256": np.asarray(source_hash),
            "training_rows": np.int64(len(tr)), "best_epoch": np.int32(epoch),
            "validation_rmse": np.float32(val), "baseline_validation_rmse": np.float32(hc_rmse),
            "h0": np.int32(a.h0), "h1": np.int32(a.h1), "seed": np.int32(a.seed),
            "batch_size": np.int32(a.batch), "epochs": np.int32(a.epochs),
            "learning_rate": np.float32(a.lr), "lr_schedule": np.asarray("cosine"),
            "baseline_rounding_note": np.asarray("Canonical white orientation differs from Black HC by at most 1cp")})

    export(0, hc_rmse)
    for epoch in range(1, a.epochs+1):
        model.train()
        rng.shuffle(tr)
        started = time.perf_counter()
        for start in range(0, len(tr), a.batch):
            r = tr[start:start+a.batch]
            correction = model(_dense_batch(idx, cnt, r, device))
            loss = F.mse_loss(torch.sigmoid((hc[r] + correction) / 400), targets[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = measure(va)
        elapsed = time.perf_counter()-started
        if not np.isfinite(val):
            raise ValueError("Non-finite validation metric")
        if val < best:
            best, best_epoch = val, epoch
            export(epoch, val)
        history.append({"epoch":epoch, "validation_rmse":val, "seconds":elapsed})
        Path(a.out + ".training.json").write_text(json.dumps({"baseline_rmse":hc_rmse,
            "best_epoch":best_epoch, "history":history, "arguments":vars(a)}, indent=2))
        print(f"Epoch {epoch}/{a.epochs}: val={val:.6f}, best={best:.6f}, {elapsed:.1f}s", flush=True)
        schedule.step()
    # Report held-out performance of the selected model once, after selection.
    weights = nn_model.load_weights(a.out)
    with torch.no_grad():
        model.acc.weight.copy_(torch.from_numpy(weights["acc_w"].T.copy()))
        model.acc.bias.copy_(torch.from_numpy(weights["acc_b"]))
        model.l1.weight.copy_(torch.from_numpy(weights["l1_w"]))
        model.l1.bias.copy_(torch.from_numpy(weights["l1_b"]))
        model.out.weight.copy_(torch.from_numpy(weights["out_w"] / 400))
        model.out.bias.copy_(torch.from_numpy(weights["out_b"] / 400))
    result = {"best_epoch":best_epoch, "validation_rmse":best,
              "test_rmse":measure(te), "baseline_test_rmse":measure(te,False),
              "data_rows":len(y), "ep_flagged_rows":len(ep_rows), "illegal_ep_rows":len(bad),
              "history":history, "arguments":vars(a)}
    Path(a.out + ".training.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("history","arguments")}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards",required=True)
    p.add_argument("--out",default="residual_64x16.npz")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--lr",type=float,default=.003)
    p.add_argument("--h0",type=int,default=64)
    p.add_argument("--h1",type=int,default=16)
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=7)
    a=p.parse_args()
    if Path(a.out).exists():
        p.error("Choose a fresh --out filename; an existing experiment will not be overwritten")
    if min(a.epochs,a.batch,a.h0,a.h1,a.threads) <= 0 or not np.isfinite(a.lr) or a.lr <= 0:
        p.error("Invalid training parameters")
    main(a)
