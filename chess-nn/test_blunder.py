"""Does the evaluation hang material when used to pick a move?

The 8M net looks reasonable offline -- Pearson 0.83, 86% sign agreement, material
tracked at 0.87 of its overall response -- yet scores ~2% against the
hand-crafted engine at equal depth. Offline metrics average over positions; a
search is steered by the eval's WORST rankings, not its average one.

This isolates eval quality from search: for each quiet position pick the move
that minimises the opponent's static eval (a 1-ply choice), then ask a simple
2-ply material swap whether that move simply hangs material. The hand-crafted
evaluation is run through exactly the same procedure as the control.

    PYTHONPATH=engine python test_blunder.py net.npz [more.npz ...]
"""
import random
import sys

import numpy as np
import chess

import nn_model
import nn_runtime as rt
from nn_encode import MAX_ACTIVE, encode_pychess
from test_material import build_cases, VALUE


def material(board, colour):
    total = 0
    for _sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        total += VALUE[piece.piece_type] * (1 if piece.color == colour else -1)
    return total


def swing_after(board, us):
    """Worst 2-ply material outcome for `us`: opponent's best capture, our best
    recapture. Crude next to a real SEE, but it catches a plainly hung piece."""
    base = material(board, us)
    worst = base
    for capture in [m for m in board.legal_moves if board.is_capture(m)]:
        board.push(capture)
        best_back = material(board, us)
        for recapture in [m for m in board.legal_moves if board.is_capture(m)]:
            board.push(recapture)
            best_back = max(best_back, material(board, us))
            board.pop()
        worst = min(worst, best_back)
        board.pop()
    return worst - base


def nn_chooser(path):
    w = nn_model.load_weights(path)
    args = (w["acc_w"], w["acc_b"], w["l1_w"], w["l1_b"], w["out_w"], w["out_b"])
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)

    def choose(board):
        best, best_move = None, None
        for move in board.legal_moves:
            board.push(move)
            ids = encode_pychess(board)
            buf[:len(ids)] = ids
            score = rt.nn_forward(buf, len(ids), *args)   # opponent to move
            board.pop()
            if best is None or score < best:
                best, best_move = score, move
        return best_move
    return choose


def handcrafted_chooser():
    from bb_numba import board_np
    import search_numba as engine

    def choose(board):
        best, best_move = None, None
        for move in board.legal_moves:
            board.push(move)
            score = engine.evaluate(board_np(board.fen()), engine._NNW, 0)
            board.pop()
            if best is None or score < best:
                best, best_move = score, move
        return best_move
    return choose


def report(name, choose, positions):
    blunders = 0
    lost = []
    for board in positions:
        move = choose(board)
        probe = board.copy()
        probe.push(move)
        swing = swing_after(probe, board.turn)
        if swing <= -200:
            blunders += 1
        lost.append(swing)
    lost = np.array(lost)
    print(f"  {name:34} hangs >=200cp in {blunders:3d}/{len(positions)} "
          f"({blunders / len(positions) * 100:5.1f}%)   mean swing {lost.mean():+7.1f} cp")


if __name__ == "__main__":
    cases = build_cases(n_target=300, seed=17)
    positions = [b for b, _s, _ml, _v in cases]
    print(f"1-ply move choice on {len(positions)} quiet, roughly balanced positions")
    print("(lower is better; a 2-ply capture/recapture swap judges each move)\n")
    report("hand-crafted evaluate()", handcrafted_chooser(), positions)
    for path in sys.argv[1:]:
        report(path, nn_chooser(path), positions)
