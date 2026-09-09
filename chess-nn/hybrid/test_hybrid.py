"""Real-dependency checks for EP semantics, blends, and baseline preservation."""
import importlib.util
import random
from pathlib import Path
import sys
import json
import subprocess
import unittest
import numpy as np
import chess
sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
import search_numba as engine
from nn_adapter import encode_legal_ep, has_legal_ep, score_nn
from nn_encode import MAX_ACTIVE, encode_pychess
from residual_data import board_from_features
import nn_model
from nn_runtime import nn_forward_numpy


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = nn_model.load_weights("net_8m_scale400_cos24.npz")
        cls.w = tuple(cls.weights[k] for k in nn_model.WEIGHT_KEYS)
        rng = random.Random(72)
        cls.boards = []
        for _ in range(12):
            b = chess.Board()
            for _ in range(80):
                if b.is_game_over():
                    break
                cls.boards.append(b.copy())
                b.push(rng.choice(list(b.legal_moves)))
        for fen in ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                    "k7/8/8/r4pPK/8/8/8/8 w - f6 0 1"):
            b = chess.Board(fen)
            cls.boards.extend((b, b.mirror()))

    def test_legal_ep_and_encoding_without_board_mutation(self):
        ids = np.empty(MAX_ACTIVE, np.int32)
        for b in self.boards:
            bd = engine.board_np(b.fen(en_passant="fen"))
            original = bd.copy()
            self.assertEqual(has_legal_ep(bd), b.has_legal_en_passant())
            n = encode_legal_ep(bd, ids)
            expected = encode_pychess(chess.Board(b.fen(en_passant="legal")))
            self.assertEqual(set(ids[:n]), set(expected))
            np.testing.assert_array_equal(bd, original)

    def test_irrelevant_ep_changes_neither_canonical_features_nor_score(self):
        b = chess.Board()
        b.push_uci("e2e4")
        raw = engine.board_np(b.fen(en_passant="fen"))
        canonical = engine.board_np(b.fen())
        self.assertEqual(score_nn(raw, self.w), score_nn(canonical, self.w))
        self.assertNotEqual(raw[14], canonical[14])

    def test_runtime_numpy_and_blend_equations(self):
        for b in self.boards[::17]:
            bd = engine.board_np(b.fen(en_passant="fen"))
            ids = np.asarray(encode_pychess(chess.Board(b.fen())), np.int32)
            raw = score_nn(bd, self.w)
            self.assertAlmostEqual(raw, nn_forward_numpy(ids, len(ids), self.weights), delta=.01)
            hc = engine.evaluate_handcrafted(bd)
            self.assertEqual(engine.evaluate(bd, self.w, 0), hc)
            for mode, expected in ((2, raw), (1100, .9 * hc + .1 * raw),
                                   (1250, .75 * hc + .25 * raw), (4000, hc + raw)):
                self.assertEqual(engine.evaluate(bd, self.w, mode), int(max(-6000, min(6000, expected))))

    def test_reconstructed_training_baseline(self):
        # Sparse features omit original colour; HC has a <=1cp floor-rounding
        # asymmetry. It is bounded and documented, not hidden as exact parity.
        errors = []
        for b in self.boards:
            bd = engine.board_np(b.fen())
            ids = np.asarray(encode_pychess(b), np.int16)
            rebuilt = board_from_features(ids, len(ids))
            errors.append(abs(engine.evaluate_handcrafted(bd) - engine.evaluate_handcrafted(rebuilt)))
        self.assertLessEqual(max(errors), 1)

    def test_original_handcrafted_function_preserved(self):
        # Separate process avoids recursive numba symbol collisions between
        # two different search modules loaded into the same LLVM namespace.
        code = ("import sys,json; import search_numba as e; "
                "fens=json.load(sys.stdin); "
                "print(json.dumps([int(e.evaluate(e.board_np(f))) for f in fens]))")
        reply = subprocess.run([sys.executable, "-c", code], cwd="engine_pristine",
                               input=json.dumps([b.fen(en_passant="fen") for b in self.boards]),
                               capture_output=True, text=True, timeout=90, check=True)
        expected = json.loads(reply.stdout.strip().splitlines()[-1])
        for b, score in zip(self.boards, expected):
            bd = engine.board_np(b.fen(en_passant="fen"))
            self.assertEqual(engine.evaluate_handcrafted(bd), score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
