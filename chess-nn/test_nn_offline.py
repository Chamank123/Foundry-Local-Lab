"""Offline subset of test_nn.py for a checkout WITHOUT the engine.

The shipped test_nn.py imports `bb_numba.board_np` from the alpha-beta engine,
which ships in neither NN ZIP. The handoff (section 5.6) forbids stubbing it and
calling the result an engine-parity pass, so this runner deliberately SKIPS the
two checks that need the engine board array:

    * encoding parity  encode_board_np(bd) == encode_pychess(board)
    * the numba runtime driven from a real engine `bd`

Everything reachable without the engine IS checked here, using indices produced
by encode_pychess as the input to the numba forward pass:

    * data contract (score selection, mover sign, canonical key, split)
    * resumable/atomic preparation
    * weight file validation
    * numba nn_forward  ==  numpy reference
    * numba nn_forward  ==  PyTorch nn_model.build_model()
    * legal-move selection demo driven by the net
    * single-core forward-pass throughput

Run test_nn.py itself in the engine checkout. This file does not replace it.
"""
import glob
import json
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import chess

from nn_encode import encode_pychess, encode_dense_pychess, MAX_ACTIVE, N_FEATURES
import nn_model
import nn_runtime as rt
import train_nnue as trainer


def _load_engine_independent_tests():
    """Import the engine-independent helpers from the REAL test_nn.py.

    test_nn.py does `from bb_numba import board_np` at module scope, so a plain
    import fails without the engine. Rather than duplicate its code (which would
    drift) or stub bb_numba (which the handoff forbids), we execute the real
    source with only that one import line removed. `board_np` therefore stays
    UNDEFINED: any test that touches the engine board raises NameError instead
    of quietly appearing to pass. Only the four helpers below are reused, and
    none of them references board_np.
    """
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "test_nn.py"), encoding="utf-8").read()
    stripped, count = re.subn(r"^from bb_numba import .*$", "", source,
                              count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("test_nn.py no longer has the expected bb_numba import; "
                           "re-check this loader before trusting its results")
    namespace = {"__name__": "test_nn_engine_independent", "__file__": "test_nn.py"}
    exec(compile(stripped, "test_nn.py", "exec"), namespace)
    return (namespace["random_positions"], namespace["test_data_contract"],
            namespace["test_prep_resume"], namespace["test_weight_validation"])


(random_positions, test_data_contract, test_prep_resume,
 test_weight_validation) = _load_engine_independent_tests()


def idx_from_board(board, buf):
    """Active indices for a python-chess board, in the runtime's buffer layout."""
    ids = encode_pychess(board)
    assert len(ids) <= MAX_ACTIVE
    buf[:len(ids)] = ids
    return len(ids)


def test_forward_parity_pychess(boards, w):
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)
    worst = 0.0
    for b in boards:
        cnt = idx_from_board(b, buf)
        a = rt.nn_forward(buf, cnt, w["acc_w"], w["acc_b"], w["l1_w"],
                          w["l1_b"], w["out_w"], w["out_b"])
        ref = rt.nn_forward_numpy(buf, cnt, w)
        worst = max(worst, abs(a - ref))
    assert worst < 1e-2, f"numba vs numpy mismatch: {worst}"
    return worst


def test_torch_parity_pychess(boards, model, w):
    import torch
    model.eval()
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)
    worst = 0.0
    with torch.no_grad():
        for b in boards:
            dense = encode_dense_pychess(b)
            t = float(model(torch.from_numpy(dense[None, :]))[0])
            cnt = idx_from_board(b, buf)
            a = rt.nn_forward(buf, cnt, w["acc_w"], w["acc_b"], w["l1_w"],
                              w["l1_b"], w["out_w"], w["out_b"])
            worst = max(worst, abs(a - t))
    assert worst < 1e-2, f"torch vs numba mismatch: {worst}"
    return worst


def test_move_selection(boards, w):
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)

    def ev(board):
        cnt = idx_from_board(board, buf)
        return rt.nn_forward(buf, cnt, w["acc_w"], w["acc_b"], w["l1_w"],
                             w["l1_b"], w["out_w"], w["out_b"])

    picked = 0
    for b in boards:
        if b.is_game_over():
            continue
        best, best_move = 1e18, None
        for m in b.legal_moves:
            b.push(m)
            score = ev(b)          # opponent to move: lower is better for us
            b.pop()
            if score < best:
                best, best_move = score, m
        assert best_move in b.legal_moves
        picked += 1
    return picked


def test_speed_pychess(w, iters=20000):
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)
    cnt = idx_from_board(b, buf)
    args = (w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"], w["out_w"], w["out_b"])
    rt.nn_forward(buf, cnt, *args)                     # warm the JIT
    started = time.perf_counter()
    for _ in range(iters):
        rt.nn_forward(buf, cnt, *args)
    return iters / (time.perf_counter() - started)


if __name__ == "__main__":
    print("SKIPPED (needs the engine's bb_numba.py): encode_board_np parity,")
    print("         runtime driven from an engine board array, nn_eval speed.")
    with tempfile.TemporaryDirectory(prefix="nnue_offline_") as temporary:
        print("checking data contract + resumable preparation ...")
        test_data_contract()
        test_prep_resume()
        print("  [PASS] score signs, canonical split, atomic shards, and exact resume")

        print("building validated random weights + model ...")
        _, w = test_weight_validation(temporary)
        model = nn_model.build_model()
        import torch
        sd = model.state_dict()
        sd["acc.weight"].copy_(torch.from_numpy(w["acc_w"].T.copy()))
        sd["acc.bias"].copy_(torch.from_numpy(w["acc_b"]))
        sd["l1.weight"].copy_(torch.from_numpy(w["l1_w"]))
        sd["l1.bias"].copy_(torch.from_numpy(w["l1_b"]))
        sd["out.weight"].copy_(torch.from_numpy(w["out_w"]))
        sd["out.bias"].copy_(torch.from_numpy(w["out_b"]))

        boards = random_positions(300)
        boards += [chess.Board(fen) for fen in (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
            "4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )]
        print("compiling encoder + runtime (one-time) ...")
        difference = test_forward_parity_pychess(boards, w)
        print(f"  [PASS] forward parity: numba == numpy (max diff {difference:.2e} cp)")
        difference = test_torch_parity_pychess(boards, model, w)
        print(f"  [PASS] forward parity: numba == PyTorch (max diff {difference:.2e} cp)")
        picked = test_move_selection(boards, w)
        print(f"  [PASS] move-selection demo returned legal moves on {picked} positions")
        nps = test_speed_pychess(w)
        print(f"  [info] non-incremental nn_forward speed: {nps:,.0f} evals/sec (one core)")
    print("OFFLINE SUBSET GREEN (engine parity NOT established)")
