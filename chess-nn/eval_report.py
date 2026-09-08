"""Chess-sanity and cold-start report for a trained checkpoint.

Beyond loss numbers: does the net rank positions sensibly, does its sign agree
with the label, and how long does the numba JIT take against the team's
90-second initialisation budget?

    python eval_report.py pilot_net_tuned.npz --shards pilot_shards
"""
import argparse
import time

import numpy as np
import chess

import nn_model
import nn_runtime as rt
from nn_encode import MAX_ACTIVE, encode_pychess
from train_nnue import _load_shards, split_indices


def make_eval(w):
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)

    def ev_board(board):
        ids = encode_pychess(board)
        buf[:len(ids)] = ids
        return float(rt.nn_forward(buf, len(ids), w["acc_w"], w["acc_b"],
                                   w["l1_w"], w["l1_b"], w["out_w"], w["out_b"]))
    return ev_board


def held_out_quality(w, shards):
    idx, cnt, y, _wdl, keys, groups = _load_shards(shards)
    split_keys = np.where(groups != 0, groups, keys).astype(np.uint64, copy=False)
    _tr, _val, rows = split_indices(split_keys)
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)
    pred = np.empty(len(rows))
    for r, row in enumerate(rows):
        n = int(cnt[row])
        buf[:n] = idx[row, :n]
        pred[r] = rt.nn_forward(buf, n, w["acc_w"], w["acc_b"], w["l1_w"],
                                w["l1_b"], w["out_w"], w["out_b"])
    label = y[rows].astype(np.float64)
    print(f"\nheld-out test split: {len(rows)} positions")
    print(f"  Pearson  r (cp vs cp)          : {np.corrcoef(pred, label)[0, 1]:.4f}")
    order_p = pred.argsort().argsort()
    order_l = label.argsort().argsort()
    print(f"  Spearman r (rank agreement)    : {np.corrcoef(order_p, order_l)[0, 1]:.4f}")
    print(f"  mean absolute error            : {np.abs(pred - label).mean():.1f} cp")
    print(f"  median absolute error          : {np.median(np.abs(pred - label)):.1f} cp")
    decisive = np.abs(label) >= 50
    sign_ok = (np.sign(pred[decisive]) == np.sign(label[decisive])).mean()
    print(f"  sign agreement (|label|>=50cp) : {sign_ok * 100:.1f}%  "
          f"({int(decisive.sum())} positions)")
    # Expected-score RMSE, the metric train() reports, recomputed from the export.
    to_score = lambda v: 1.0 / (1.0 + np.exp(-v / 400.0))
    rmse = float(np.sqrt(np.mean((to_score(pred) - to_score(label)) ** 2)))
    print(f"  expected-score RMSE (exported) : {rmse:.5f}")


def material_sanity(ev):
    """The net should not be blind to plain material and simple mates."""
    cases = [
        ("start position (expect near 0)",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("white to move, a queen up (expect large +)",
         "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("white to move, a queen down (expect large -)",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
        ("black to move, a queen up (expect large +)",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"),
        ("white to move, a rook up (expect +)",
         "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"),
        ("white to move, a knight up (expect +)",
         "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("white to move, a pawn up (expect small +)",
         "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("K+Q vs K, white to move (expect large +)", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"),
        ("K+Q vs K, black to move (expect large -)", "4k3/8/8/8/8/8/8/3QK3 b - - 0 1"),
    ]
    print("\nmaterial sanity (mover-relative centipawns):")
    for label, fen in cases:
        print(f"  {ev(chess.Board(fen)):+9.1f}  {label}")

    print("\nmirror symmetry (mover-relative encoding must make these identical):")
    for fen in ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "8/5k2/4p3/3pP3/3P1K2/8/8/8 w - - 0 1"):
        board = chess.Board(fen)
        a, b = ev(board), ev(board.mirror())
        flag = "OK" if abs(a - b) < 1e-3 else "MISMATCH"
        print(f"  {a:+9.1f} vs mirrored {b:+9.1f}  [{flag}]  {fen}")


def cold_start(w):
    """numba compile time, against the team's 90-second initialisation budget.

    Only nn_forward is measured. encode_board_np compiles against the engine's
    board array, so its compile cost must be measured in the engine checkout.
    """
    from numba import njit
    buf = np.zeros(MAX_ACTIVE, dtype=np.int32)
    buf[:5] = [0, 100, 300, 768, 772]
    started = time.perf_counter()
    rt.nn_forward(buf, 5, w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"],
                  w["out_w"], w["out_b"])
    compile_s = time.perf_counter() - started
    args = (w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"], w["out_w"], w["out_b"])
    iters = 20000
    started = time.perf_counter()
    for _ in range(iters):
        rt.nn_forward(buf, 5, *args)
    nps = iters / (time.perf_counter() - started)
    print(f"\ncold start: nn_forward numba compile = {compile_s:.1f}s "
          f"(budget 90s; encode_board_np NOT included - needs the engine)")
    print(f"throughput: {nps:,.0f} forward passes/sec on one core "
          f"(non-incremental, ~{1e6 / nps:.1f} us each)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights")
    parser.add_argument("--shards", default="pilot_shards")
    args = parser.parse_args()

    w = nn_model.load_weights(args.weights)
    with np.load(args.weights, allow_pickle=False) as z:
        meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
    print(f"checkpoint: {args.weights}")
    for k in sorted(meta):
        print(f"  meta {k:16}= {meta[k]}")
    ev = make_eval(w)
    material_sanity(ev)
    held_out_quality(w, args.shards)
    cold_start(w)
