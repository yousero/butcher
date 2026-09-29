"""
Конвертер Lichess puzzle CSV → PGN, совместимый с training/puzzle_loader.py.

Берёт первый ход из Moves как "решение" для позиции FEN.

Использование:
    python scripts/lichess_to_pgn.py lichess_db_puzzle.csv data/puzzles.pgn --max 20000
"""
import argparse
import csv
import os
import sys

import chess
import chess.pgn


def convert(csv_path, pgn_path, max_puzzles=None, min_rating=None, max_rating=None,
            themes=None):
    themes = set(themes) if themes else None
    written = 0
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as fin, \
         open(pgn_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            if max_puzzles and written >= max_puzzles:
                break

            fen = row["FEN"]
            moves = row["Moves"].split()
            if not moves:
                skipped += 1
                continue

            rating = int(row["Rating"])
            if min_rating and rating < min_rating:
                continue
            if max_rating and rating > max_rating:
                continue

            if themes:
                row_themes = set(row["Themes"].split())
                if not (row_themes & themes):
                    continue

            try:
                board = chess.Board(fen)
            except ValueError:
                skipped += 1
                continue

            # Первый ход в Moves — решение для позиции FEN
            try:
                sol = chess.Move.from_uci(moves[0])
            except ValueError:
                skipped += 1
                continue
            if sol not in board.legal_moves:
                skipped += 1
                continue

            game = chess.pgn.Game()
            game.headers["Event"] = "Lichess Puzzle"
            game.headers["Site"] = row.get("GameUrl", "")
            game.headers["FEN"] = fen
            game.headers["SetUp"] = "1"
            game.headers["Rating"] = str(rating)
            game.headers["Themes"] = row.get("Themes", "")
            node = game.add_variation(sol)
            # Опционально: дописать остальные ходы решения
            for mv_uci in moves[1:]:
                try:
                    node = node.add_variation(chess.Move.from_uci(mv_uci))
                except ValueError:
                    break
            fout.write(str(game) + "\n\n")
            written += 1

    print(f"Готово: {written} задач записано в {pgn_path}, пропущено {skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("pgn_path")
    ap.add_argument("--max", type=int, default=20000)
    ap.add_argument("--min-rating", type=int, default=None)
    ap.add_argument("--max-rating", type=int, default=None)
    ap.add_argument("--themes", nargs="*", default=None,
                    help="например: mate mateIn1 mateIn2 endgame")
    args = ap.parse_args()
    convert(args.csv_path, args.pgn_path, args.max,
            args.min_rating, args.max_rating, args.themes)