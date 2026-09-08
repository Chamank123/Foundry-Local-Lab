"""AI Chessathon submission entry point.

A from-scratch chess engine: a numba-JIT bitboard move generator (perft-verified
against the standard positions) with an alpha-beta search — transposition table,
principal variation search, null-move pruning, late move reductions, check
extensions, quiescence, and a tapered piece-square evaluation.

No external engine, no published network, no runtime databases: every move is
produced by this code. Dependencies are numpy and numba only (both preinstalled
in the competition image). JIT compilation is triggered at import, so it is paid
inside the pre-clock initialisation budget.

Public API:
    get_move(fen: str, time_left_ms: int) -> str   # returns a UCI move
"""

import search_numba as _engine

# Track the fullmove counter so we reset per-game state cleanly whether the
# harness starts a fresh process per game or reuses one (fullmove resets to 1).
_last_fullmove = [10 ** 9]


def get_move(fen, time_left_ms):
    try:
        parts = fen.split()
        fullmove = int(parts[5]) if len(parts) >= 6 else 1
        if fullmove < _last_fullmove[0]:
            _engine.new_game()
        _last_fullmove[0] = fullmove
    except Exception:
        pass
    return _engine.get_move(fen, time_left_ms)
