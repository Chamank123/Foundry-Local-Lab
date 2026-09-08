"""Controlled material-sensitivity probe.

REPORT.md's first version of this test was too crude: it removed a piece from
any random-play position, including tactical and already-decided ones, where a
teacher would not move by the piece's nominal value either. NEXT_STEPS section 3
asks to keep only legal comparisons, control side-to-move, and separate quiet
roughly-balanced positions. This does that: positions are quiet (mover not in
check, no capture available to either side) and roughly balanced (nominal
material within +-200 cp) before a piece is removed.

    PYTHONPATH=engine python test_material.py pilot_net_tuned.npz [more.npz ...]
"""
import random
import sys

import numpy as np
import chess

import nn_model
import nn_runtime as rt
from nn_encode import MAX_ACTIVE, encode_pychess

VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
         chess.ROOK: 500, chess.QUEEN: 900}


def make_eval(w):
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)
    args = (w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"], w["out_w"], w["out_b"])

    def ev(board):
        ids = encode_pychess(board)
        buf[:len(ids)] = ids
        return float(rt.nn_forward(buf, len(ids), *args))
    return ev


def nominal(board):
    total = 0
    for _sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        signed = VALUE[piece.piece_type] * (1 if piece.color == board.turn else -1)
        total += signed
    return total


def quiet(board):
    if board.is_check():
        return False
    if any(board.is_capture(m) for m in board.legal_moves):
        return False
    mirrored = board.mirror()          # check the opponent has no capture either
    return not any(mirrored.is_capture(m) for m in mirrored.legal_moves)


def build_cases(n_target=600, seed=5):
    """Quiet, roughly balanced, legal positions with one piece removed."""
    rng = random.Random(seed)
    cases = []
    attempts = 0
    while len(cases) < n_target and attempts < 60000:
        attempts += 1
        board = chess.Board()
        for _ in range(rng.randint(10, 60)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or not quiet(board):
            continue
        if abs(nominal(board)) > 200:
            continue
        candidates = [(sq, p) for sq, p in board.piece_map().items()
                      if p.piece_type != chess.KING]
        if not candidates:
            continue
        sq, piece = rng.choice(candidates)
        stripped = board.copy()
        stripped.remove_piece_at(sq)
        # Removing a piece must leave a legal, non-terminal, still-quiet position
        # with the SAME side to move, or the comparison is not controlled.
        if not stripped.is_valid() or stripped.is_game_over() or not quiet(stripped):
            continue
        cases.append((board, stripped, piece.color == board.turn,
                      VALUE[piece.piece_type]))
    return cases


def report(name, ev, cases):
    print(f"\n{name}")
    print(f"  {'piece':>6} {'n':>4} | {'mover loses it':>26} | {'opponent loses it':>26}")
    print(f"  {'':>6} {'':>4} | {'mean dcp':>12} {'correct':>13} | "
          f"{'mean dcp':>12} {'correct':>13}")
    for value in (100, 320, 330, 500, 900):
        line = f"  {value:6d}"
        counts = 0
        cells = []
        for mover_loses in (True, False):
            subset = [(b, s) for b, s, ml, v in cases if v == value and ml == mover_loses]
            if not subset:
                cells.append(f"{'-':>12} {'-':>13}")
                continue
            deltas = np.array([ev(s) - ev(b) for b, s in subset])
            want_negative = mover_loses
            correct = np.mean((deltas < 0) == want_negative) * 100
            counts = max(counts, len(subset))
            cells.append(f"{deltas.mean():+12.1f} {correct:11.0f}%  ")
        print(f"{line} {counts:4d} | {cells[0]} | {cells[1]}")
    deltas = np.array([(ev(s) - ev(b)) * (-1 if ml else 1) for b, s, ml, v in cases])
    values = np.array([v for _b, _s, _ml, v in cases])
    print(f"  overall: correct direction {np.mean(deltas > 0) * 100:.0f}% of "
          f"{len(cases)} cases")
    print(f"  slope of |eval change| on true value (1.0 = perfect scaling): "
          f"{np.polyfit(values, np.abs(deltas), 1)[0]:.3f}")


if __name__ == "__main__":
    cases = build_cases()
    print(f"built {len(cases)} quiet, roughly balanced, legal comparisons")
    for path in sys.argv[1:]:
        report(path, make_eval(nn_model.load_weights(path)), cases)
