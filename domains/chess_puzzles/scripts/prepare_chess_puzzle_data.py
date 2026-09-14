#!/usr/bin/env python3
"""Prepare the Lichess chess-puzzle RSFT, MH, and test splits.

This consolidates the original split/de-duplication logic formerly stored in
``autoharness_chess_puzzle/grpo_dataset.py`` into the domain's canonical
data-preparation entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import chess
import pandas as pd


DEFAULT_DATASET = "Lichess/chess-puzzles"
DOMAIN_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DOMAIN_DIR / "data" / "chess_puzzle"
SPLIT_SIZES = {"train": 16384, "test": 256, "mh_val": 256}

if str(DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(DOMAIN_DIR))


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _moves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.lower() for item in value.split() if item.strip()]
    if isinstance(value, list):
        return [str(item).lower() for item in value if str(item).strip()]
    return []


def _themes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _position_key(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 2:
        return parts[0] + " " + parts[1]
    return fen


def _board_ascii(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece is not None else ".")
        rows.append(f"{rank + 1} " + " ".join(cells))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def _initial_observation(board: chess.Board) -> str:
    side = "White" if board.turn == chess.WHITE else "Black"
    rows = [
        "Task: solve the chess puzzle one move at a time.",
        f"Side to move: {side}",
        f"FEN: {board.fen()}",
        "Board:",
        _board_ascii(board),
        "Puzzle move number: 1",
        "Legal moves:",
    ]
    legal_moves = sorted(board.legal_moves, key=lambda move: move.uci())
    rows.extend(f"- [{move.uci()}] SAN={board.san(move)}" for move in legal_moves)
    return "\n".join(rows)


def _initial_prompt(extra_info: dict[str, Any]) -> list[dict[str, str]]:
    try:
        from autoharness_chess_puzzle.harness import load_harness

        harness = load_harness(DOMAIN_DIR / "environments" / "chess_puzzle" / "base_harness.py")
        board = chess.Board(str(extra_info["start_fen"]))
        observation = _initial_observation(board)
        if harness.format_observation is not None:
            observation = str(
                harness.format_observation(
                    observation,
                    turn_count=0,
                    side_to_move="white" if board.turn == chess.WHITE else "black",
                    last_opponent_move=None,
                    max_turns=harness.max_turns or 9,
                )
            )
        system = harness.system_prompt or (
            "You solve chess tactics by choosing one legal move at a time. "
            "Return exactly one UCI move in <move></move> tags."
        )
        user_template = harness.user_prompt or (
            "Solve the current chess puzzle position.\n\n{observation}\n\n"
            "Return exactly one move as <move>uci</move>."
        )
        try:
            user = user_template.format(observation=observation)
        except Exception:
            user = user_template + "\n\n" + observation
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    except Exception:
        return [
            {
                "role": "system",
                "content": "You solve chess puzzles by returning one UCI move at a time.",
            },
            {
                "role": "user",
                "content": "Solve the current chess puzzle position. Return exactly one UCI move.",
            },
        ]


def convert_lichess_row(row: dict[str, Any]) -> dict[str, Any] | None:
    initial_fen = str(row.get("FEN") or row.get("fen") or "").strip()
    raw_moves = _moves(row.get("Moves") or row.get("moves"))
    if not initial_fen or len(raw_moves) < 2:
        return None
    try:
        board = chess.Board(initial_fen)
        trigger_move = chess.Move.from_uci(raw_moves[0])
        if trigger_move not in board.legal_moves:
            return None
        board.push(trigger_move)
        start_fen = board.fen()
        solution_moves = raw_moves[1:]
        verify_board = chess.Board(start_fen)
        for move_uci in solution_moves:
            move = chess.Move.from_uci(move_uci)
            if move not in verify_board.legal_moves:
                return None
            verify_board.push(move)
    except Exception:
        return None

    puzzle_id = str(
        row.get("PuzzleId")
        or row.get("id")
        or stable_hash(initial_fen + " " + " ".join(raw_moves))[:16]
    )
    rating = row.get("Rating") or row.get("rating")
    try:
        rating_int = int(rating)
    except (TypeError, ValueError):
        rating_int = None
    themes = _themes(row.get("Themes") or row.get("themes"))
    extra_info = {
        "task": "chess-puzzle-v0",
        "source": DEFAULT_DATASET,
        "puzzle_id": puzzle_id,
        "initial_fen": initial_fen,
        "trigger_move": raw_moves[0],
        "start_fen": start_fen,
        "solution_moves": solution_moves,
        "raw_moves": raw_moves,
        "rating": rating_int,
        "themes": themes,
        "position_key": _position_key(start_fen),
    }
    return {
        "data_source": "chess-puzzle-v0",
        "prompt": _initial_prompt(extra_info),
        "ability": "chess_puzzle",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": extra_info,
    }


def _split_for_key(key: str, sizes: dict[str, int]) -> str:
    total = sum(sizes.values())
    bucket = int(stable_hash(key)[:12], 16) % total
    cursor = 0
    for name in ("mh_val", "test", "train"):
        cursor += sizes[name]
        if bucket < cursor:
            return name
    return "train"


def _split_path(output_dir: Path, split: str, size: int) -> Path:
    if split == "train":
        return output_dir / f"lichess_puzzles_train_{size}.parquet"
    if split == "test":
        return output_dir / f"lichess_puzzles_test_{size}.parquet"
    return output_dir / f"lichess_puzzles_mh_val_{size}.parquet"


def _decode_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _load_existing_split(path: Path, expected_size: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    if len(frame) != expected_size:
        return []
    return frame.to_dict("records")


def _position_keys(records: list[dict[str, Any]], split: str) -> set[str]:
    keys: list[str] = []
    for record in records:
        key = str(_decode_extra_info(record.get("extra_info")).get("position_key") or "")
        if not key:
            raise RuntimeError(f"{split} contains a row without extra_info.position_key")
        keys.append(key)
    unique = set(keys)
    if len(unique) != len(keys):
        raise RuntimeError(f"{split} contains duplicate chess positions")
    return unique


def _iter_hf_parquet_rows(
    dataset_name: str, batch_size: int = 1024
) -> Iterator[dict[str, Any]]:
    """Read Hub parquet shards without leaving an Arrow streaming worker alive."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    pattern = f"datasets/{dataset_name}/data/train-*.parquet"
    parquet_paths = sorted(filesystem.glob(pattern))
    if not parquet_paths:
        raise RuntimeError(
            f"no train parquet shards found for {dataset_name!r} at {pattern!r}"
        )
    for parquet_path in parquet_paths:
        with filesystem.open(parquet_path, "rb") as source:
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
                yield from batch.to_pylist()


def _validate_splits(rows: dict[str, list[dict[str, Any]]], sizes: dict[str, int]) -> None:
    seen: set[str] = set()
    for split, expected_size in sizes.items():
        if len(rows[split]) != expected_size:
            raise RuntimeError(
                f"{split} has {len(rows[split])} rows; expected {expected_size}"
            )
        keys = _position_keys(rows[split], split)
        overlap = seen.intersection(keys)
        if overlap:
            raise RuntimeError(
                f"{split} overlaps another split on {len(overlap)} chess positions"
            )
        seen.update(keys)


def build_splits(
    *,
    output_dir: Path,
    dataset_name: str = DEFAULT_DATASET,
    train_size: int = 16384,
    test_size: int = 256,
    mh_val_size: int = 256,
    seed: int = 42,
    max_scan: int = 2_000_000,
    min_rating: int | None = None,
    max_rating: int | None = None,
    min_solution_plies: int = 1,
    max_solution_plies: int = 9,
    preserve_existing_eval_splits: bool = True,
) -> dict[str, Path]:
    sizes = {"train": train_size, "test": test_size, "mh_val": mh_val_size}
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    seen_positions: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    preserved: dict[str, int] = {}
    if preserve_existing_eval_splits:
        for split in ("test", "mh_val"):
            existing = _load_existing_split(
                _split_path(output_dir, split, sizes[split]), sizes[split]
            )
            if not existing:
                continue
            existing_keys = _position_keys(existing, split)
            overlap = seen_positions.intersection(existing_keys)
            if overlap:
                raise RuntimeError(
                    f"preserved {split} overlaps another eval split on "
                    f"{len(overlap)} positions"
                )
            rows[split] = existing
            preserved[split] = len(existing)
            seen_positions.update(existing_keys)

    stream = _iter_hf_parquet_rows(dataset_name)
    scanned = 0
    try:
        for raw in stream:
            scanned += 1
            if scanned > max_scan:
                break
            converted = convert_lichess_row(dict(raw))
            if converted is None:
                continue
            extra = converted["extra_info"]
            rating = extra.get("rating")
            if min_rating is not None and rating is not None and int(rating) < min_rating:
                continue
            if max_rating is not None and rating is not None and int(rating) > max_rating:
                continue
            plies = len(extra["solution_moves"])
            if plies < min_solution_plies or plies > max_solution_plies:
                continue
            position_key = str(extra["position_key"])
            if position_key in seen_positions:
                continue
            split_key = stable_hash(position_key + f":{seed}")
            split = _split_for_key(split_key, sizes)
            if len(rows[split]) >= sizes[split]:
                for fallback in ("mh_val", "test", "train"):
                    if len(rows[fallback]) < sizes[fallback]:
                        split = fallback
                        break
                else:
                    break
            seen_positions.add(position_key)
            converted["extra_info"]["split"] = split
            rows[split].append(converted)
            if all(len(rows[name]) >= sizes[name] for name in sizes):
                break
    finally:
        stream.close()

    missing = {
        name: sizes[name] - len(values)
        for name, values in rows.items()
        if len(values) < sizes[name]
    }
    if missing:
        raise RuntimeError(f"could not fill requested splits after scanning {scanned} rows: {missing}")
    _validate_splits(rows, sizes)

    paths = {
        "train": _split_path(output_dir, "train", train_size),
        "test": _split_path(output_dir, "test", test_size),
        "mh_val": _split_path(output_dir, "mh_val", mh_val_size),
    }
    for name, path in paths.items():
        frame = pd.DataFrame(rows[name])
        frame.to_parquet(path, index=False)
    manifest = {
        "dataset": dataset_name,
        "seed": seed,
        "sizes": {name: len(values) for name, values in rows.items()},
        "paths": {name: str(path) for name, path in paths.items()},
        "max_scan": max_scan,
        "filters": {
            "min_rating": min_rating,
            "max_rating": max_rating,
            "min_solution_plies": min_solution_plies,
            "max_solution_plies": max_solution_plies,
        },
        "preserved_existing_eval_splits": preserved,
        "disjoint_key": "start_fen piece-placement plus active color",
        "unique_positions": sum(len(values) for values in rows.values()),
        "pairwise_position_overlap": 0,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--train-size", type=int, default=SPLIT_SIZES["train"])
    parser.add_argument("--test-size", type=int, default=SPLIT_SIZES["test"])
    parser.add_argument("--mh-val-size", type=int, default=SPLIT_SIZES["mh_val"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scan", type=int, default=2_000_000)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--min-solution-plies", type=int, default=1)
    parser.add_argument("--max-solution-plies", type=int, default=9)
    parser.add_argument(
        "--no-preserve-existing-eval-splits",
        action="store_true",
        help="Rebuild test and MH validation splits instead of preserving existing files of the requested sizes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_splits(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        train_size=args.train_size,
        test_size=args.test_size,
        mh_val_size=args.mh_val_size,
        seed=args.seed,
        max_scan=args.max_scan,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        min_solution_plies=args.min_solution_plies,
        max_solution_plies=args.max_solution_plies,
        preserve_existing_eval_splits=not args.no_preserve_existing_eval_splits,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
