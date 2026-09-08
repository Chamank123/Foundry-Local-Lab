"""Encoder parity on positions reached through the ENGINE's own make/unmake.

test_nn.py builds every position from a FEN string, so both encoders are handed
the same parsed state and agree by construction. The engine does not play that
way: it reaches positions by applying bb_numba.make(), which sets bd[EP] after
EVERY double pawn push, whether or not an en-passant capture is actually legal.
python-chess models the same thing, but Board.fen() defaults to
en_passant="legal" and DROPS the square when no capture is available -- so a
FEN round trip can silently hide a disagreement that the live engine would hit.

This walks random games with the engine's own move generator and make(),
carrying a python-chess board alongside move for move, and compares the two
encoders at every ply. Run from the NN folder with the engine importable:

    PYTHONPATH=engine python test_engine_parity.py
"""
import random
import sys

import numpy as np
import chess

from bb_numba import board_np, gen_legal, make, STM, EP, NO_EP
from nn_encode import encode_board_np, encode_pychess, MAX_ACTIVE

PROMO_PIECE = {1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}


def to_uci_move(m, pyboard):
    """Translate an engine move int into the equivalent python-chess Move."""
    frm = m & 63
    to = (m >> 6) & 63
    promo = (m >> 12) & 7
    move = chess.Move(frm, to, promotion=PROMO_PIECE[promo] if promo else None)
    if move not in pyboard.legal_moves:
        raise AssertionError(f"engine move {move.uci()} illegal in {pyboard.fen()}")
    return move


def walk(games=300, max_plies=120, seed=11):
    rng = random.Random(seed)
    buf = np.empty(MAX_ACTIVE, dtype=np.int32)
    out = np.empty(256, dtype=np.int32)

    positions = 0
    mismatches = []
    ep_set_no_legal_capture = 0
    ep_set_total = 0
    fen_would_hide = 0

    for _ in range(games):
        bd = board_np(chess.STARTING_FEN)
        pyboard = chess.Board()
        for _ply in range(max_plies):
            n = gen_legal(bd, out)
            if n == 0 or pyboard.is_game_over():
                break

            # --- compare the two encoders on the CURRENT position ---
            cnt = encode_board_np(bd, buf)
            engine_ids = sorted(int(v) for v in buf[:cnt])
            pychess_ids = sorted(encode_pychess(pyboard))
            positions += 1
            if engine_ids != pychess_ids:
                mismatches.append({
                    "fen": pyboard.fen(),
                    "epd": pyboard.epd(),
                    "engine_ep": int(bd[EP]),
                    "pychess_ep": pyboard.ep_square,
                    "only_engine": sorted(set(engine_ids) - set(pychess_ids)),
                    "only_pychess": sorted(set(pychess_ids) - set(engine_ids)),
                })

            # --- track how often the engine holds an EP square with no capture ---
            if int(bd[EP]) != NO_EP:
                ep_set_total += 1
                legal_ep = any(pyboard.is_en_passant(mv) for mv in pyboard.legal_moves)
                if not legal_ep:
                    ep_set_no_legal_capture += 1
                # would a FEN round trip lose this square?
                if chess.Board(pyboard.fen()).ep_square != pyboard.ep_square:
                    fen_would_hide += 1

            move_int = int(out[rng.randrange(n)])
            pyboard.push(to_uci_move(move_int, pyboard))
            bd = make(bd, move_int)

            assert (int(bd[STM]) == 0) == pyboard.turn, "side-to-move desynchronised"

    return positions, mismatches, ep_set_total, ep_set_no_legal_capture, fen_would_hide


if __name__ == "__main__":
    positions, mismatches, ep_total, ep_no_cap, hidden = walk()
    print(f"positions compared (engine make/unmake path): {positions}")
    print(f"  engine held an EP square: {ep_total}")
    print(f"    of those, NO legal en-passant capture existed: {ep_no_cap}")
    print(f"    of those, a FEN round trip would DROP the square: {hidden}")
    print(f"encoder mismatches: {len(mismatches)}")
    for bad in mismatches[:5]:
        print(f"  fen={bad['fen']}")
        print(f"    engine ep={bad['engine_ep']} pychess ep={bad['pychess_ep']}")
        print(f"    only in engine  : {bad['only_engine']}")
        print(f"    only in pychess : {bad['only_pychess']}")
    if mismatches:
        print(f"\nFAIL: {len(mismatches)}/{positions} positions disagree")
        sys.exit(1)
    print("\nPASS: encoders agree on every position reached through engine make()")
