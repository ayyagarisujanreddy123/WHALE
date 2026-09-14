#!/usr/bin/env python3
"""Reconstruct the exact ReTool DAPO RSFT and Meta-Harness artifacts.

The one-off generator was not retained in the ReTool repository history; only
the resulting parquet files were committed. This implementation recovers the
observed procedure exactly:

1. Keep the first row for each stable ``extra_info.index`` in the public
   DAPO-Math-17K train stream. The source used by the experiment exposed the
   same 17,917 examples repeatedly.
2. Build the 256-row MH artifact with
   ``Dataset.shuffle(seed=42).select(range(256))`` and record sample metadata.

The historical MH artifact is a subset of the RSFT pool. AIME 2024/2025 use
their official test inputs directly and undergo no local split or de-dup step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, IterableDataset, load_dataset


DEFAULT_TRAIN_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "retool"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(
        _jsonable(row),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _dedup_key(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("index") not in (None, ""):
        return f"extra_info.index:{extra['index']}"
    return f"row:{_canonical_row(row)}"


def deduplicate_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first occurrence of every stable DAPO row key."""

    unique: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    for row_number, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        key = _dedup_key(row)
        canonical = _canonical_row(row)
        previous = seen.get(key)
        if previous is not None:
            duplicate_count += 1
            if previous != canonical:
                raise ValueError(
                    f"dedup key {key!r} maps to different rows "
                    f"(at source row {row_number})"
                )
            continue
        seen[key] = canonical
        unique.append(row)
        if row_number % 100_000 == 0:
            print(
                f"scanned {row_number:,} rows; unique={len(unique):,}; "
                f"duplicates={duplicate_count:,}",
                flush=True,
            )
    return unique, duplicate_count


def _load_train_rows(args: argparse.Namespace) -> Dataset | IterableDataset:
    if args.train_input is not None:
        return load_dataset("parquet", data_files=str(args.train_input), split="train")
    return load_dataset(
        args.train_dataset,
        split=args.train_split,
        streaming=not args.no_streaming,
    )


def build_meta_harness(train: Dataset, *, size: int, seed: int) -> Dataset:
    if size <= 0 or size > len(train):
        raise ValueError(f"invalid MH size {size}; train contains {len(train)} rows")
    sampled = train.shuffle(seed=seed).select(range(size))

    def annotate(row: dict[str, Any], index: int) -> dict[str, Any]:
        extra = row.get("extra_info")
        extra = dict(extra) if isinstance(extra, dict) else {}
        extra["mh_sample_seed"] = seed
        extra["mh_sample_index"] = index
        row["extra_info"] = extra
        return row

    return sampled.map(annotate, with_indices=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--train-input",
        type=Path,
        default=None,
        help="Optional local source parquet; otherwise --train-dataset is loaded.",
    )
    parser.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the complete HF source in memory before de-duplication.",
    )
    parser.add_argument("--expected-train-size", type=int, default=17_917)
    parser.add_argument("--mh-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, duplicate_count = deduplicate_rows(_load_train_rows(args))
    if args.expected_train_size > 0 and len(rows) != args.expected_train_size:
        raise ValueError(
            f"expected {args.expected_train_size:,} unique DAPO rows, "
            f"found {len(rows):,}; pin/check the source revision or override "
            "--expected-train-size"
        )

    train = Dataset.from_list(rows)
    harness = build_meta_harness(train, size=args.mh_size, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "dapo_math_17k_dedup.parquet"
    harness_path = (
        args.output_dir
        / f"meta_harness_dapo17k_dedup_seed{args.seed}_{args.mh_size}.parquet"
    )
    train.to_parquet(str(train_path))
    harness.to_parquet(str(harness_path))

    print(
        f"wrote RSFT: {train_path} ({len(train):,} rows; "
        f"removed {duplicate_count:,} repeated rows)"
    )
    print(f"wrote MH:   {harness_path} ({len(harness):,} rows, seed={args.seed})")


if __name__ == "__main__":
    main()
