"""Independent verification of a received shard snapshot.

Checks each shard on its own before anything is concatenated, so a bad shard is
named rather than surfacing as a confusing error 8 million rows later. Then
verifies the cross-shard invariants that only hold globally: key uniqueness and
non-empty, disjoint splits.

    python verify_snapshot.py overnight_shards_8m
"""
import glob
import json
import os
import sys

import numpy as np

from nn_encode import MAX_ACTIVE, N_FEATURES
from train_nnue import _inspect_shard, split_indices, CP_CLAMP, WDL_UNKNOWN


def main(shards_dir):
    manifest_path = os.path.join(shards_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    files = sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz")))
    print(f"manifest: kept={manifest['kept']:,} shards={manifest['shards']} "
          f"shard_size={manifest['shard_size']:,} min_depth={manifest['min_depth']} "
          f"schema_version={manifest.get('schema_version')} "
          f"stop_reason={manifest.get('stop_reason')}")
    print(f"files on disk: {len(files)}")
    if len(files) != manifest["shards"]:
        print(f"  *** shard count mismatch: manifest says {manifest['shards']}")

    total = 0
    all_keys = []
    y_min, y_max = np.inf, -np.inf
    y_sum = 0.0
    clamped = 0
    cnt_min, cnt_max = 10**9, -1
    ep_rows = 0
    king_bad = 0
    bad = []

    for path in files:
        try:
            n, keys = _inspect_shard(path, include_keys=True)   # schema gate
        except ValueError as exc:
            bad.append(f"{os.path.basename(path)}: {exc}")
            continue
        with np.load(path, allow_pickle=False) as z:
            idx, cnt, y, wdl, group = z["idx"], z["cnt"], z["y"], z["wdl"], z["group"]
        if not np.all(wdl == WDL_UNKNOWN):
            bad.append(f"{os.path.basename(path)}: wdl is not all {WDL_UNKNOWN}")
        if not np.all(group == 0):
            bad.append(f"{os.path.basename(path)}: group is not all 0")
        if np.any(np.abs(y) > CP_CLAMP):
            bad.append(f"{os.path.basename(path)}: |y| exceeds {CP_CLAMP}")

        active = np.arange(MAX_ACTIVE)[None, :] < cnt[:, None]
        kings = (((idx >= 320) & (idx < 384)) | ((idx >= 704) & (idx < 768))) & active
        king_bad += int((kings.sum(1) != 2).sum())
        ep_rows += int(((((idx >= 772) & (idx < 780)) & active).sum(1) > 0).sum())

        total += n
        all_keys.append(keys)
        y_min = min(y_min, float(y.min()))
        y_max = max(y_max, float(y.max()))
        y_sum += float(y.sum())
        clamped += int((np.abs(y) == CP_CLAMP).sum())
        cnt_min = min(cnt_min, int(cnt.min()))
        cnt_max = max(cnt_max, int(cnt.max()))

    print(f"\nrows summed independently from shards: {total:,}")
    print(f"  manifest 'kept' agrees: {total == manifest['kept']}")
    print(f"  every row has exactly 2 kings: {king_bad == 0}"
          + (f" ({king_bad:,} bad rows)" if king_bad else ""))
    print(f"  active-feature count range: {cnt_min}..{cnt_max} (capacity {MAX_ACTIVE})")
    print(f"  label range: {y_min:.1f}..{y_max:.1f}  mean {y_sum / total:+.2f}")
    print(f"  labels on the +-{CP_CLAMP} clamp: {clamped:,} ({clamped / total * 100:.2f}%)")
    print(f"  rows carrying an en-passant feature: {ep_rows:,} "
          f"({ep_rows / total * 100:.3f}%)")

    keys = np.concatenate(all_keys)
    del all_keys
    unique = len(np.unique(keys))
    print(f"\n  canonical keys unique across ALL shards: {unique == len(keys)} "
          f"({len(keys) - unique:,} collisions)")

    train, val, test = split_indices(keys)
    print(f"  split train/validation/test: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"    all non-empty: {bool(len(train) and len(val) and len(test))}")
    print(f"    sum equals total: {len(train) + len(val) + len(test) == total}")

    if bad:
        print("\nPROBLEMS:")
        for line in bad:
            print("  " + line)
        return 1
    print("\nSNAPSHOT VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "overnight_shards_8m"))
