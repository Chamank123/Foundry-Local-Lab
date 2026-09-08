"""Foundation tests. Run:  python3 test_nn.py
Green here means the encoding contract holds and the runtime matches PyTorch, so
the team can build on it with confidence.
"""
import glob
import json
import os
import random
import sys
import tempfile
import time
sys.path.insert(0, os.path.dirname(__file__))                 # nn/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) or "..")  # build/ (engine)

import numpy as np
import chess

from nn_encode import encode_board_np, encode_pychess, encode_dense_pychess, MAX_ACTIVE, N_FEATURES
import nn_model
import nn_runtime as rt
import train_nnue as trainer
from bb_numba import board_np   # engine board array from FEN


def random_positions(n, max_plies=40, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(0, max_plies)):
            ms = list(b.legal_moves)
            if not ms or b.is_game_over():
                break
            b.push(rng.choice(ms))
        out.append(b)
    return out


def test_encoding_parity(boards):
    idxbuf = np.empty(MAX_ACTIVE, dtype=np.int32)
    for b in boards:
        fen = b.fen()
        b2 = chess.Board(fen)               # normalise via FEN = the engine's view
        cnt = encode_board_np(board_np(fen), idxbuf)
        eng = sorted(int(x) for x in idxbuf[:cnt])
        pyc = sorted(encode_pychess(b2))
        assert eng == pyc, f"ENCODING MISMATCH\n{fen}\nengine={eng}\npychess={pyc}"
        assert all(0 <= i < N_FEATURES for i in eng)
    return len(boards)


def test_forward_parity(boards, w):
    idxbuf = np.empty(MAX_ACTIVE, dtype=np.int32)
    worst = 0.0
    for b in boards:
        cnt = encode_board_np(board_np(b.fen()), idxbuf)
        a = rt.nn_forward(idxbuf, cnt, w["acc_w"], w["acc_b"], w["l1_w"],
                          w["l1_b"], w["out_w"], w["out_b"])
        ref = rt.nn_forward_numpy(idxbuf, cnt, w)
        worst = max(worst, abs(a - ref))
    assert worst < 1e-2, f"numba vs numpy mismatch: {worst}"
    return worst


def test_torch_parity(boards, model, w):
    import torch
    model.eval()
    idxbuf = np.empty(MAX_ACTIVE, dtype=np.int32)
    worst = 0.0
    with torch.no_grad():
        for b in boards:
            fen = b.fen(); b2 = chess.Board(fen)          # same view for both sides
            dense = encode_dense_pychess(b2)              # (780,)
            t = float(model(torch.from_numpy(dense[None, :]))[0])
            cnt = encode_board_np(board_np(fen), idxbuf)
            a = rt.nn_forward(idxbuf, cnt, w["acc_w"], w["acc_b"], w["l1_w"],
                              w["l1_b"], w["out_w"], w["out_b"])
            worst = max(worst, abs(a - t))
    assert worst < 1e-2, f"torch vs numba mismatch: {worst}"
    return worst


def test_engine_can_use_net(boards, w):
    """Prove the socket: pick the move minimising the opponent's NN eval of the
    resulting position. Every move returned must be legal."""
    idxbuf = np.empty(MAX_ACTIVE, dtype=np.int32)
    def ev(bd):
        cnt = encode_board_np(bd, idxbuf)
        return rt.nn_forward(idxbuf, cnt, w["acc_w"], w["acc_b"], w["l1_w"],
                             w["l1_b"], w["out_w"], w["out_b"])
    picked = 0
    for b in boards:
        if b.is_game_over():
            continue
        best, bestmv = 1e18, None
        for m in b.legal_moves:
            b.push(m)
            s = ev(board_np(b.fen()))   # opponent-to-move eval; lower = better for us
            b.pop()
            if s < best:
                best, bestmv = s, m
        assert bestmv in b.legal_moves
        picked += 1
    return picked


def test_data_contract():
    """Score selection, mover sign, canonical key, and deterministic split."""
    rec = {"evals": [
        {"depth": 12, "pvs": [{"cp": 20}]},
        {"depth": 20, "pvs": [{"cp": 75}, {"cp": -10}]},
        {"depth": 18, "pvs": [{"cp": 50}]},
    ]}
    assert trainer._pick_eval(rec, 12) == 75
    assert trainer._pick_eval({"evals": [{"depth": 20, "pvs": []}]}, 12) is None
    assert trainer._pick_eval({"evals": [{"depth": 20, "pvs": [{"mate": 3}]}]}, 12) == 6000
    assert trainer._pick_eval({"evals": [{"depth": 20, "pvs": [{"mate": -3}]}]}, 12) == -6000
    assert trainer._mover_relative_cp(123, True) == 123
    assert trainer._mover_relative_cp(123, False) == -123
    assert trainer._mover_relative_cp(99999, True) == 6000

    board = chess.Board()
    mirrored = board.mirror()
    key_a = trainer._canonical_key(encode_pychess(board))
    key_b = trainer._canonical_key(encode_pychess(mirrored))
    assert key_a == key_b, "mover-relative mirror must have one canonical key"

    keys = np.array([key_a, key_a, key_a + np.uint64(1), key_a + np.uint64(2)])
    splits = trainer.split_indices(keys)
    membership = {}
    for split_no, rows in enumerate(splits):
        for row in rows:
            membership[int(row)] = split_no
    assert membership[0] == membership[1], "duplicate canonical keys crossed splits"


def test_prep_resume():
    """A partial shard resumes exactly, without duplicates or overwrites."""
    records = [
        {"fen": chess.STARTING_FEN, "evals": [{"depth": 18, "pvs": [{"cp": 10}]}]},
        {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
         "evals": [{"depth": 18, "pvs": [{"cp": 20}]}]},
        "not a record",
        {"fen": chess.STARTING_FEN, "evals": [{"depth": 18, "pvs": []}]},
        {"fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
         "evals": [{"depth": 18, "pvs": [{"cp": 30}]}]},
        {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
         "evals": [{"depth": 18, "pvs": [{"cp": -40}]}]},
    ]
    with tempfile.TemporaryDirectory(prefix="nnue_prep_test_") as temporary:
        db = os.path.join(temporary, "eval.jsonl")
        out = os.path.join(temporary, "shards")
        with open(db, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

        trainer.prep(db, out, limit=2, min_depth=12, shard_size=2, audit_size=2)
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as fh:
            first = json.load(fh)
        assert first["kept"] == 2 and first["scanned"] == 2

        trainer.prep(db, out, limit=4, min_depth=12, shard_size=2,
                     resume=True, audit_size=2)
        files = sorted(glob.glob(os.path.join(out, "shard_*.npz")))
        assert len(files) == 2
        keys = []
        labels = []
        for path in files:
            with np.load(path, allow_pickle=False) as shard:
                keys.extend(int(k) for k in shard["key"])
                labels.extend(float(y) for y in shard["y"])
                assert np.all(shard["group"] == 0)
        assert len(keys) == len(set(keys)) == 4
        assert labels[:2] == [10.0, -20.0], "Lichess White scores were not made mover-relative"
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as fh:
            final = json.load(fh)
        assert final["kept"] == 4 and final["scanned"] == len(records)
        assert final["rejected"]["malformed_record"] == 1
        assert final["rejected"]["no_usable_eval"] == 1

        try:
            trainer.prep(db, out, limit=4, min_depth=12, shard_size=2)
        except FileExistsError:
            pass
        else:
            raise AssertionError("prep accepted a non-empty output folder without --resume")

        lock = os.path.join(out, ".prep.lock")
        with open(lock, "w", encoding="ascii") as fh:
            fh.write("test lock\n")
        try:
            trainer.prep(db, out, limit=4, min_depth=12, shard_size=2,
                         resume=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("prep did not reject a concurrent output-folder writer")
        finally:
            os.unlink(lock)


def test_weight_validation(temporary):
    good = os.path.join(temporary, "weights_random.npz")
    nn_model.random_weights(good, seed=0)
    w = nn_model.load_weights(good)
    assert w["acc_w"].shape[0] == N_FEATURES

    bad = os.path.join(temporary, "weights_bad.npz")
    np.savez(bad, acc_w=np.zeros((1, 1), np.float32))
    try:
        nn_model.load_weights(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid weights were accepted")
    return good, w


def test_speed(w, iters=20000):
    b = board_np("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    idxbuf = np.empty(MAX_ACTIVE, dtype=np.int32)
    # warm
    rt.nn_eval(b, w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"], w["out_w"], w["out_b"])
    t0 = time.perf_counter()
    for _ in range(iters):
        rt.nn_eval(b, w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"], w["out_w"], w["out_b"])
    dt = time.perf_counter() - t0
    return iters / dt


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="nnue_test_") as temporary:
        print("checking data contract + resumable preparation ...")
        test_data_contract()
        test_prep_resume()
        print("  [PASS] score signs, canonical split, atomic shards, and exact resume")

        print("building validated random weights + model ...")
        _, w = test_weight_validation(temporary)
        model = nn_model.build_model()
        # Load the SAME random weights into PyTorch for forward parity.
        import torch
        sd = model.state_dict()
        sd["acc.weight"].copy_(torch.from_numpy(w["acc_w"].T.copy()))
        sd["acc.bias"].copy_(torch.from_numpy(w["acc_b"]))
        sd["l1.weight"].copy_(torch.from_numpy(w["l1_w"]))
        sd["l1.bias"].copy_(torch.from_numpy(w["l1_b"]))
        sd["out.weight"].copy_(torch.from_numpy(w["out_w"]))
        sd["out.bias"].copy_(torch.from_numpy(w["out_b"]))

        boards = random_positions(300)
        extra_fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
            "4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            # FEN-standard ep square even though no capture is currently legal.
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ]
        boards += [chess.Board(fen) for fen in extra_fens]
        print("compiling engine + encoder (one-time) ...")
        n = test_encoding_parity(boards)
        print(f"  [PASS] encoding parity: engine == python-chess on {n} positions")
        difference = test_forward_parity(boards, w)
        print(f"  [PASS] forward parity: numba == numpy (max diff {difference:.2e} cp)")
        difference = test_torch_parity(boards, model, w)
        print(f"  [PASS] forward parity: numba == PyTorch (max diff {difference:.2e} cp)")
        picked = test_engine_can_use_net(boards, w)
        print(f"  [PASS] move-selection demo returned legal moves on {picked} positions")
        nps = test_speed(w)
        print(f"  [info] non-incremental nn_eval speed: {nps:,.0f} evals/sec (one core)")
    print("ALL GREEN")
