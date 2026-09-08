"""Audited data preparation and training for the AI Chessathon value net.

Commands:
  python3 train_nnue.py smoke
  python3 train_nnue.py prep --db lichess_db_eval.jsonl.zst --out shards_pilot \
      --limit 100000 --min-depth 16
  python3 train_nnue.py prep --db lichess_db_eval.jsonl.zst --out shards_pilot \
      --limit 1000000 --min-depth 16 --resume
  python3 train_nnue.py train --shards shards_pilot --out net.npz

Shards keep sparse active indices, mover-relative centipawn labels, a reserved
completed-game result field, a canonical position key, and a reserved game-group
key used for deterministic 98/1/1 train/validation/test assignment.
"""

import argparse
import glob
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time

import numpy as np

import nn_model
from nn_encode import MAX_ACTIVE, N_FEATURES, encode_pychess


CP_CLAMP = 6000
DEFAULT_SCALE = 400.0
WDL_UNKNOWN = -1.0
SHARD_SCHEMA_VERSION = 2
SPLIT_BUCKETS = 1000
DEFAULT_VAL_PERMILLE = 10
DEFAULT_TEST_PERMILLE = 10


def _open_records(path):
    """Stream JSONL, optionally zstd-compressed, without permanent expansion."""
    if path.endswith(".zst"):
        try:
            import zstandard
            fh = open(path, "rb")
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            return io.TextIOWrapper(reader, encoding="utf-8")
        except ImportError:
            process = subprocess.Popen(["zstdcat", path], stdout=subprocess.PIPE)
            if process.stdout is None:
                raise RuntimeError("zstdcat did not provide stdout")
            return io.TextIOWrapper(process.stdout, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _norm_fen(fen):
    """Return a six-field FEN, filling only the clocks omitted by Lichess."""
    parts = fen.split()
    if len(parts) < 4:
        raise ValueError("FEN must contain board, mover, castling, and ep fields")
    if len(parts) == 4:
        parts.extend(("0", "1"))
    elif len(parts) == 5:
        parts.append("1")
    return " ".join(parts[:6])


def _pick_eval(rec, min_depth):
    """Return the White-relative score from the deepest usable first PV."""
    best = None
    for ev in rec.get("evals", []):
        if not isinstance(ev, dict):
            continue
        depth = ev.get("depth", 0)
        pvs = ev.get("pvs")
        if not isinstance(depth, (int, float)) or depth < min_depth:
            continue
        if not isinstance(pvs, list) or not pvs or not isinstance(pvs[0], dict):
            continue
        pv = pvs[0]
        if "cp" not in pv and "mate" not in pv:
            continue
        if best is None or depth > best[0]:
            best = (depth, pv)
    if best is None:
        return None
    pv = best[1]
    if "cp" in pv:
        try:
            cp = float(pv["cp"])
        except (TypeError, ValueError):
            return None
        return cp if np.isfinite(cp) else None
    try:
        mate = int(pv["mate"])
    except (TypeError, ValueError):
        return None
    if mate == 0:
        return None
    return float(CP_CLAMP if mate > 0 else -CP_CLAMP)


def _mover_relative_cp(white_cp, white_to_move):
    cp = float(white_cp if white_to_move else -white_cp)
    return max(-CP_CLAMP, min(CP_CLAMP, cp))


def _canonical_key(ids):
    """Stable uint64 key for the complete mover-relative 780-feature state."""
    ordered = np.asarray(sorted(int(i) for i in ids), dtype="<i2")
    digest = hashlib.blake2b(ordered.tobytes(), digest_size=8,
                             person=b"nnue-pos").digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _source_sample_key(fen):
    digest = hashlib.blake2b(fen.encode("utf-8"), digest_size=8,
                             person=b"nnue-src").digest()
    return int.from_bytes(digest, "little")


def split_indices(keys, val_permille=DEFAULT_VAL_PERMILLE,
                  test_permille=DEFAULT_TEST_PERMILLE):
    """Deterministically split canonical keys; identical positions stay together."""
    if val_permille < 0 or test_permille < 0:
        raise ValueError("split sizes must be non-negative")
    if val_permille + test_permille >= SPLIT_BUCKETS:
        raise ValueError("validation + test must leave a non-empty training split")
    buckets = np.asarray(keys, dtype=np.uint64) % np.uint64(SPLIT_BUCKETS)
    test = np.flatnonzero(buckets < test_permille)
    val = np.flatnonzero((buckets >= test_permille) &
                         (buckets < test_permille + val_permille))
    train = np.flatnonzero(buckets >= test_permille + val_permille)
    return train, val, test


def _atomic_savez(path, **arrays):
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".shard_", suffix=".npz",
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


def _atomic_json(path, value):
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest_", suffix=".json",
                                     dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _shard_number(path):
    stem = os.path.basename(path)
    try:
        return int(stem[len("shard_"):-len(".npz")])
    except ValueError as exc:
        raise ValueError(f"invalid shard filename: {path}") from exc


def _inspect_shard(path, include_keys=False):
    with np.load(path, allow_pickle=False) as z:
        required = ("idx", "cnt", "y", "wdl", "key", "group")
        missing = [name for name in required if name not in z]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        idx = z["idx"]
        cnt = z["cnt"]
        y = z["y"]
        wdl = z["wdl"]
        keys = z["key"]
        groups = z["group"]
        n = len(y)
        if idx.shape != (n, MAX_ACTIVE):
            raise ValueError(f"{path}: bad idx shape {idx.shape}")
        if (cnt.shape != (n,) or wdl.shape != (n,) or keys.shape != (n,) or
                groups.shape != (n,)):
            raise ValueError(f"{path}: inconsistent row counts")
        if np.any(cnt < 0) or np.any(cnt > MAX_ACTIVE):
            raise ValueError(f"{path}: invalid active-feature count")
        active = np.arange(MAX_ACTIVE)[None, :] < cnt[:, None]
        if np.any(idx[active] < 0) or np.any(idx[active] >= N_FEATURES):
            raise ValueError(f"{path}: active feature index outside 0..{N_FEATURES - 1}")
        if np.any(idx[~active] != -1):
            raise ValueError(f"{path}: padding must be -1")
        if not np.isfinite(y).all() or not np.isfinite(wdl).all():
            raise ValueError(f"{path}: non-finite labels")
        return n, keys.astype(np.uint64).copy() if include_keys else None


def _prep_config(db_path, min_depth, shard_size, dedup,
                 sample_modulus, sample_remainder):
    stat = os.stat(db_path)
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "source_path": os.path.abspath(db_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "min_depth": int(min_depth),
        "shard_size": int(shard_size),
        "dedup": bool(dedup),
        "sample_modulus": int(sample_modulus),
        "sample_remainder": int(sample_remainder),
    }


def _prep_locked(db_path, out_dir, limit=None, min_depth=12, shard_size=500_000,
                 dedup=True, resume=False, sample_modulus=1, sample_remainder=0,
                 audit_size=200):
    """Stream, validate, encode, split-key, and atomically shard evaluations."""
    import chess

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if min_depth < 0 or shard_size <= 0 or audit_size < 0:
        raise ValueError("min-depth and audit-size must be non-negative; shard-size must be positive")
    if sample_modulus <= 0 or not 0 <= sample_remainder < sample_modulus:
        raise ValueError("require 0 <= sample-remainder < sample-modulus")

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    audit_path = os.path.join(out_dir, "audit_sample.json")
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    config = _prep_config(db_path, min_depth, shard_size, dedup,
                          sample_modulus, sample_remainder)

    seen = set()
    rejected = {}
    scanned = 0
    kept_total = 0
    next_shard = 0
    audit = []

    if shards or os.path.exists(manifest_path):
        if not resume:
            raise FileExistsError(
                f"{out_dir} already contains prepared data; use --resume only to "
                "continue the exact same configuration, or choose a fresh folder")
        if not os.path.exists(manifest_path):
            raise ValueError("cannot resume: manifest.json is missing")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        for key, value in config.items():
            if old.get(key) != value:
                raise ValueError(f"cannot resume: {key} changed from {old.get(key)!r} to {value!r}")
        scanned = int(old.get("scanned", 0))
        rejected = dict(old.get("rejected", {}))
        for path in shards:
            n, keys = _inspect_shard(path, include_keys=dedup)
            kept_total += n
            next_shard = max(next_shard, _shard_number(path) + 1)
            if keys is not None:
                seen.update(int(key) for key in keys)
        if kept_total != int(old.get("kept", -1)):
            raise ValueError("cannot resume: manifest row count does not match shards")
        if os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8") as fh:
                audit = json.load(fh)
        print(f"resuming after {scanned:,} source records and {kept_total:,} saved positions")
    else:
        print("starting a fresh preparation run")

    def reject(reason):
        rejected[reason] = int(rejected.get(reason, 0)) + 1

    idx_buf, cnt_buf, y_buf, key_buf = [], [], [], []
    started = time.perf_counter()

    def write_manifest(status, stop_reason=None):
        value = dict(config)
        value.update({
            "status": status,
            "stop_reason": stop_reason,
            "scanned": scanned,
            "kept": kept_total,
            "shards": next_shard,
            "rejected": rejected,
        })
        _atomic_json(manifest_path, value)
        if audit:
            _atomic_json(audit_path, audit)

    def flush():
        nonlocal next_shard, kept_total, idx_buf, cnt_buf, y_buf, key_buf
        if not y_buf:
            return
        path = os.path.join(out_dir, f"shard_{next_shard:05d}.npz")
        if os.path.exists(path):
            raise FileExistsError(f"refusing to overwrite existing shard: {path}")
        n = len(y_buf)
        _atomic_savez(
            path,
            idx=np.stack(idx_buf).astype(np.int16, copy=False),
            cnt=np.asarray(cnt_buf, dtype=np.int16),
            y=np.asarray(y_buf, dtype=np.float32),
            wdl=np.full(n, WDL_UNKNOWN, dtype=np.float32),
            key=np.asarray(key_buf, dtype=np.uint64),
            group=np.zeros(n, dtype=np.uint64),
        )
        kept_total += n
        next_shard += 1
        idx_buf, cnt_buf, y_buf, key_buf = [], [], [], []
        write_manifest("running")
        print(f"  wrote {path} ({n:,} rows; {kept_total:,} total; "
              f"{time.perf_counter() - started:.0f}s)")

    stopped = "eof"
    records = _open_records(db_path)
    try:
        for line_number, line in enumerate(records, start=1):
            if line_number <= scanned:
                continue
            if limit is not None and kept_total + len(y_buf) >= limit:
                stopped = "limit"
                break
            scanned = line_number
            try:
                rec = json.loads(line)
                fen = _norm_fen(rec["fen"])
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                reject("malformed_record")
                continue

            if _source_sample_key(fen) % sample_modulus != sample_remainder:
                reject("sampled_out")
                continue
            white_cp = _pick_eval(rec, min_depth)
            if white_cp is None:
                reject("no_usable_eval")
                continue

            try:
                board = chess.Board(fen)
            except ValueError:
                reject("invalid_fen")
                continue
            if not board.is_valid():
                reject("invalid_position")
                continue
            if board.is_game_over(claim_draw=False):
                reject("terminal_position")
                continue

            ids = encode_pychess(board)
            if len(ids) > MAX_ACTIVE or len(ids) != len(set(ids)):
                reject("invalid_encoding")
                continue
            if any(i < 0 or i >= N_FEATURES for i in ids):
                reject("invalid_encoding")
                continue

            key = int(_canonical_key(ids))
            if dedup and key in seen:
                reject("duplicate")
                continue
            if dedup:
                seen.add(key)

            mover_cp = _mover_relative_cp(white_cp, board.turn)
            row = np.full(MAX_ACTIVE, -1, dtype=np.int16)
            row[:len(ids)] = ids
            idx_buf.append(row)
            cnt_buf.append(len(ids))
            y_buf.append(mover_cp)
            key_buf.append(key)

            if len(audit) < audit_size:
                audit.append({
                    "fen": fen,
                    "selected_white_cp": white_cp,
                    "mover_cp": mover_cp,
                    "active": sorted(int(i) for i in ids),
                    "canonical_key": key,
                })
            if len(y_buf) >= shard_size:
                flush()
    except KeyboardInterrupt:
        flush()
        write_manifest("interrupted", "keyboard_interrupt")
        raise
    finally:
        records.close()

    flush()
    write_manifest("complete" if stopped == "eof" else "stopped", stopped)
    print(f"done: scanned {scanned:,}, kept {kept_total:,}, "
          f"wrote {next_shard} shard(s) -> {out_dir} ({stopped})")


def prep(db_path, out_dir, limit=None, min_depth=12, shard_size=500_000,
         dedup=True, resume=False, sample_modulus=1, sample_remainder=0,
         audit_size=200):
    """Run preparation while holding an exclusive lock on the output folder."""
    os.makedirs(out_dir, exist_ok=True)
    lock_path = os.path.join(out_dir, ".prep.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another prep process may own {out_dir}; if it crashed, verify no "
            f"worker is running before removing {lock_path}") from exc
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(lock_fd)
        lock_fd = None
        return _prep_locked(db_path, out_dir, limit, min_depth, shard_size,
                            dedup, resume, sample_modulus, sample_remainder,
                            audit_size)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _load_shards(shards_dir):
    files = sorted(glob.glob(os.path.join(shards_dir, "shard_*.npz")))
    if not files and shards_dir.endswith(".npz"):
        files = [shards_dir]
    if not files:
        raise FileNotFoundError(f"no shards in {shards_dir}")

    parts = {name: [] for name in ("idx", "cnt", "y", "wdl", "key", "group")}
    for path in files:
        _inspect_shard(path)
        with np.load(path, allow_pickle=False) as z:
            for name in parts:
                parts[name].append(z[name])
    idx = np.concatenate(parts["idx"]).astype(np.int16, copy=False)
    cnt = np.concatenate(parts["cnt"]).astype(np.int16, copy=False)
    y = np.concatenate(parts["y"]).astype(np.float32, copy=False)
    wdl = np.concatenate(parts["wdl"]).astype(np.float32, copy=False)
    keys = np.concatenate(parts["key"]).astype(np.uint64, copy=False)
    groups = np.concatenate(parts["group"]).astype(np.uint64, copy=False)
    print(f"loaded {len(y):,} positions from {len(files)} shard(s)")
    return idx, cnt, y, wdl, keys, groups


def _dense_batch(idx, cnt, rows, device):
    """Vectorised sparse-to-dense expansion directly on the training device."""
    import torch

    selected_idx = torch.as_tensor(idx[rows], dtype=torch.long, device=device)
    selected_cnt = torch.as_tensor(cnt[rows], dtype=torch.long, device=device)
    width = selected_idx.shape[1]
    valid = torch.arange(width, device=device).unsqueeze(0) < selected_cnt.unsqueeze(1)
    batch_rows = torch.arange(len(rows), device=device).unsqueeze(1).expand_as(selected_idx)
    x = torch.zeros((len(rows), N_FEATURES), dtype=torch.float32, device=device)
    x[batch_rows[valid], selected_idx[valid]] = 1.0
    return x


def train(shards_dir, out_path, epochs=30, bs=8192, lr=1e-3,
          scale=DEFAULT_SCALE, lam=1.0, loss="mse", seed=0,
          val_permille=DEFAULT_VAL_PERMILLE,
          test_permille=DEFAULT_TEST_PERMILLE, lr_schedule="none",
          out_scale=1.0):
    import torch
    import torch.nn.functional as F

    if epochs <= 0 or bs <= 0 or lr <= 0 or scale <= 0:
        raise ValueError("epochs, batch size, learning rate, and scale must be positive")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    if lr_schedule not in ("none", "cosine"):
        raise ValueError("lr-schedule must be 'none' or 'cosine'")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | K={scale} | lambda(eval)={lam} | loss={loss} | "
          f"seed={seed} | lr={lr} | batch={bs} | epochs={epochs} | "
          f"lr-schedule={lr_schedule} | out-scale={out_scale}")

    idx, cnt, y, wdl, keys, groups = _load_shards(shards_dir)
    allowed_wdl = (wdl == WDL_UNKNOWN) | (wdl == 0.0) | (wdl == 0.5) | (wdl == 1.0)
    if not np.all(allowed_wdl):
        raise ValueError("wdl must be -1, 0, 0.5, or 1")
    # Locally labelled game data can set a shared non-zero group key so every
    # position from one game stays in one split. Eval-DB rows use group=0.
    split_keys = np.where(groups != 0, groups, keys).astype(np.uint64, copy=False)
    tr, val, test = split_indices(split_keys, val_permille, test_permille)
    if len(tr) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("a split is empty; use more data or adjust split permille values")
    print(f"split: train={len(tr):,} validation={len(val):,} test={len(test):,}")

    rng = np.random.default_rng(seed)
    y_t = torch.from_numpy(y)
    wdl_t = torch.from_numpy(wdl)

    def targets(rows):
        cp = y_t[rows].to(device)
        result = wdl_t[rows].to(device)
        eval_score = torch.sigmoid(cp / scale)
        blended = lam * eval_score + (1.0 - lam) * result
        return torch.where(result >= 0.0, blended, eval_score)

    def training_loss(predicted, target):
        if loss == "huber":
            return F.huber_loss(predicted, target, delta=0.1)
        return F.mse_loss(predicted, target)

    model = nn_model.build_model(out_scale=out_scale).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    # Optional cosine decay, stepped once per epoch. Defaults to "none" so the
    # handoff's prescribed baseline command reproduces exactly.
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
                 if lr_schedule == "cosine" else None)

    def evaluate(rows):
        model.eval()
        squared_error = 0.0
        objective = 0.0
        with torch.no_grad():
            for start in range(0, len(rows), bs):
                batch = rows[start:start + bs]
                predicted = torch.sigmoid(model(_dense_batch(idx, cnt, batch, device)) / scale)
                target = targets(batch)
                squared_error += F.mse_loss(predicted, target, reduction="sum").item()
                objective += training_loss(predicted, target).item() * len(batch)
        return (squared_error / len(rows)) ** 0.5, objective / len(rows)

    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(tr)
        started = time.perf_counter()
        for start in range(0, len(tr), bs):
            batch = tr[start:start + bs]
            predicted = torch.sigmoid(model(_dense_batch(idx, cnt, batch, device)) / scale)
            objective = training_loss(predicted, targets(batch))
            optimiser.zero_grad(set_to_none=True)
            objective.backward()
            optimiser.step()
        if scheduler is not None:
            scheduler.step()
        val_rmse, val_objective = evaluate(val)
        if not np.isfinite(val_rmse):
            raise FloatingPointError("validation metric became non-finite")
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:2}/{epochs} validation expected-score RMSE={val_rmse:.5f} "
              f"objective={val_objective:.6f} ({time.perf_counter() - started:.0f}s)")

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_rmse, test_objective = evaluate(test)
    metadata = {
        "scale": np.float32(scale),
        "lambda": np.float32(lam),
        "seed": np.int64(seed),
        "best_epoch": np.int32(best_epoch),
        "validation_rmse": np.float32(best_rmse),
        "test_rmse": np.float32(test_rmse),
        "loss": np.array(loss),
        "epochs": np.int32(epochs),
        "batch_size": np.int32(bs),
        "learning_rate": np.float32(lr),
        "lr_schedule": np.array(lr_schedule),
        "out_scale": np.float32(out_scale),
    }
    nn_model.export_weights(model, out_path, metadata=metadata)
    print(f"best epoch {best_epoch}: validation RMSE={best_rmse:.5f}; "
          f"untouched test RMSE={test_rmse:.5f} objective={test_objective:.6f}")
    print(f"exported runtime weights -> {out_path}")


def smoke():
    """Exercise synthetic sharding, deterministic splits, training, and runtime."""
    import nn_runtime as runtime

    rng = np.random.default_rng(0)
    n = 20_000
    cnt = rng.integers(6, 24, n).astype(np.int16)
    idx = np.full((n, MAX_ACTIVE), -1, np.int16)
    secret = rng.standard_normal(N_FEATURES).astype(np.float32)
    y = np.zeros(n, np.float32)
    for row in range(n):
        features = rng.choice(N_FEATURES, cnt[row], replace=False)
        idx[row, :cnt[row]] = features
        y[row] = np.clip(secret[features].sum() * 120.0, -CP_CLAMP, CP_CLAMP)
    keys = np.arange(n, dtype=np.uint64)

    with tempfile.TemporaryDirectory(prefix="nnue_smoke_") as temporary:
        shard_dir = os.path.join(temporary, "shards")
        os.makedirs(shard_dir)
        shard = os.path.join(shard_dir, "shard_00000.npz")
        _atomic_savez(shard, idx=idx, cnt=cnt, y=y,
                      wdl=np.full(n, WDL_UNKNOWN, np.float32), key=keys,
                      group=np.zeros(n, np.uint64))
        output = os.path.join(temporary, "smoke_net.npz")
        train(shard_dir, output, epochs=5, bs=2048, seed=0)
        weights = nn_model.load_weights(output)
        value = runtime.nn_forward(idx[0].astype(np.int32), int(cnt[0]),
                                   weights["acc_w"], weights["acc_b"],
                                   weights["l1_w"], weights["l1_b"],
                                   weights["out_w"], weights["out_b"])
        if not np.isfinite(value):
            raise AssertionError("runtime returned a non-finite score")
        print(f"smoke runtime sample={value:.1f} cp target={y[0]:.1f} cp")
    print("smoke OK")


def _build_cli():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="cmd", required=True)
    commands.add_parser("smoke")

    prep_parser = commands.add_parser("prep")
    prep_parser.add_argument("--db", required=True)
    prep_parser.add_argument("--out", required=True)
    prep_parser.add_argument("--limit", type=int)
    prep_parser.add_argument("--min-depth", type=int, default=12)
    prep_parser.add_argument("--shard-size", type=int, default=500_000)
    prep_parser.add_argument("--no-dedup", action="store_true")
    prep_parser.add_argument("--resume", action="store_true")
    prep_parser.add_argument("--sample-modulus", type=int, default=1)
    prep_parser.add_argument("--sample-remainder", type=int, default=0)
    prep_parser.add_argument("--audit-size", type=int, default=200)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--shards", required=True)
    train_parser.add_argument("--out", required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=8192)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    train_parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    train_parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--val-permille", type=int, default=DEFAULT_VAL_PERMILLE)
    train_parser.add_argument("--test-permille", type=int, default=DEFAULT_TEST_PERMILLE)
    train_parser.add_argument("--lr-schedule", choices=("none", "cosine"), default="none")
    train_parser.add_argument("--output-scale", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    if args.cmd == "smoke":
        smoke()
    elif args.cmd == "prep":
        prep(args.db, args.out, args.limit, args.min_depth, args.shard_size,
             not args.no_dedup, args.resume, args.sample_modulus,
             args.sample_remainder, args.audit_size)
    elif args.cmd == "train":
        train(args.shards, args.out, args.epochs, args.batch_size,
              args.learning_rate, args.scale, args.lam, args.loss, args.seed,
              args.val_permille, args.test_permille, args.lr_schedule,
              args.output_scale)
