#!/usr/bin/env python3
"""Prepare the exact SearchQA RSFT, MH, validation, and test splits.

The one-off generators for the final RSFT/MH artifacts and balanced validation
artifact were not retained in the Search-R1 Git history. Their deterministic
rules were recovered from the retained parquet files and consolidated here as
the domain's canonical data-preparation entrypoint.

Inputs:
  - the retained cleaned 19,202-row training pool;
  - the earlier balanced 256-row training exclusion set;
  - the full seven-source Search-R1 test mixture.

Outputs:
  - 18,946-row RSFT data;
  - 256-row MH data (200 HotpotQA + 56 NQ, seed 42);
  - 700-row validation data (100 per source, seed 42);
  - 51,013-row held-out test data (full test minus validation).

The historical MH data is intentionally a subset of the RSFT pool. The test
output is disjoint from validation by ``(data_source, extra_info.index)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "searchR1_processed"
DEFAULT_HARNESS_COUNTS = {
    "searchR1_hotpotqa": 200,
    "searchR1_nq": 56,
}
EXPECTED_TEST_SOURCES = {
    "searchR1_2wikimultihopqa",
    "searchR1_bamboogle",
    "searchR1_hotpotqa",
    "searchR1_musique",
    "searchR1_nq",
    "searchR1_popqa",
    "searchR1_triviaqa",
}


def _extra(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _structural_key(row: pd.Series) -> tuple[str, str] | None:
    index = _extra(row.get("extra_info")).get("index")
    if index is None:
        return None
    return (str(row.get("data_source", "")), str(index))


def _question_key(row: pd.Series) -> str | None:
    question = str(_extra(row.get("extra_info")).get("question", "")).strip()
    return question or None


def _required_structural_keys(
    frame: pd.DataFrame, label: str
) -> list[tuple[str, str]]:
    keys = [_structural_key(row) for _, row in frame.iterrows()]
    if any(key is None for key in keys):
        raise ValueError(f"{label} contains rows without extra_info.index")
    required = [key for key in keys if key is not None]
    if len(required) != len(set(required)):
        raise ValueError(
            f"{label} contains duplicate (data_source, extra_info.index) keys"
        )
    return required


def _parse_counts(values: list[str] | None) -> dict[str, int]:
    if not values:
        return dict(DEFAULT_HARNESS_COUNTS)
    counts: dict[str, int] = {}
    for value in values:
        try:
            source, raw_count = value.rsplit("=", 1)
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"invalid --harness-count {value!r}; expected DATA_SOURCE=COUNT"
            ) from exc
        source = source.strip()
        if not source or count <= 0:
            raise argparse.ArgumentTypeError(
                f"invalid --harness-count {value!r}; count must be positive"
            )
        if source in counts:
            raise argparse.ArgumentTypeError(f"duplicate harness source: {source}")
        counts[source] = count
    return counts


def build_weight_split(source: pd.DataFrame, exclude: pd.DataFrame) -> pd.DataFrame:
    """Remove the earlier balanced harness set from the cleaned train pool."""

    _required_structural_keys(source, "weight source")
    exclude_structural = set(
        _required_structural_keys(exclude, "weight exclusion set")
    )
    exclude_questions = {
        key for _, row in exclude.iterrows() if (key := _question_key(row)) is not None
    }

    keep: list[bool] = []
    for _, row in source.iterrows():
        structural = _structural_key(row)
        question = _question_key(row)
        matched = structural in exclude_structural or (
            question is not None and question in exclude_questions
        )
        keep.append(not matched)

    output = source.loc[keep].reset_index(drop=True)
    removed = len(source) - len(output)
    if removed != len(exclude):
        raise ValueError(
            f"expected to remove {len(exclude)} exclusion rows, removed {removed}; "
            "check that the inputs come from the same preprocessing run"
        )
    _required_structural_keys(output, "RSFT output")
    return output


def build_harness_split(
    weight: pd.DataFrame, *, counts: dict[str, int], seed: int
) -> pd.DataFrame:
    """Reproduce the distribution-aligned MH sample."""

    sampled: list[pd.DataFrame] = []
    for source in sorted(counts):
        group = weight.loc[weight["data_source"] == source]
        count = counts[source]
        if len(group) < count:
            raise ValueError(f"{source} has {len(group)} rows, cannot sample {count}")
        sampled.append(group.sample(n=count, random_state=seed))

    output = pd.concat(sampled, ignore_index=True)
    output = output.sample(frac=1, random_state=seed).reset_index(drop=True)
    harness_keys = set(_required_structural_keys(output, "MH output"))
    weight_keys = set(_required_structural_keys(weight, "RSFT output"))
    if not harness_keys <= weight_keys:
        raise AssertionError("MH output contains rows outside the RSFT pool")
    return output


def build_validation_split(
    full_test: pd.DataFrame, *, per_source: int, seed: int
) -> pd.DataFrame:
    """Reproduce the sorted, balanced ``test_100.parquet`` artifact."""

    _required_structural_keys(full_test, "full test input")
    sources = set(full_test["data_source"].astype(str).unique())
    if sources != EXPECTED_TEST_SOURCES:
        raise ValueError(
            "unexpected SearchQA test sources; "
            f"missing={sorted(EXPECTED_TEST_SOURCES - sources)}, "
            f"extra={sorted(sources - EXPECTED_TEST_SOURCES)}"
        )

    sampled: list[pd.DataFrame] = []
    for source, group in full_test.groupby("data_source", sort=True):
        if len(group) < per_source:
            raise ValueError(
                f"{source} has only {len(group)} rows, cannot sample {per_source}"
            )
        sampled.append(group.sample(n=per_source, random_state=seed))
    output = pd.concat(sampled, ignore_index=True)
    _required_structural_keys(output, "validation output")
    return output


def build_final_test(
    full_test: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    """Remove validation rows from the full test mixture without reordering."""

    full_keys = _required_structural_keys(full_test, "full test input")
    validation_keys = set(
        _required_structural_keys(validation, "validation output")
    )
    keep = [key not in validation_keys for key in full_keys]
    output = full_test.loc[keep].reset_index(drop=True)
    if len(full_test) - len(output) != len(validation):
        raise ValueError("not every validation row matched the full test input")
    output_keys = set(_required_structural_keys(output, "held-out test output"))
    if output_keys & validation_keys:
        raise AssertionError("held-out test overlaps validation")
    return output


def _expect_rows(frame: pd.DataFrame, expected: int, label: str) -> None:
    if expected > 0 and len(frame) != expected:
        raise ValueError(f"expected {expected:,} {label} rows, found {len(frame):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight-source",
        type=Path,
        default=DATA_DIR / "train_filtered_20000_post_processed.parquet",
    )
    parser.add_argument(
        "--weight-exclude",
        type=Path,
        default=DATA_DIR / "train_meta_harness_256.parquet",
    )
    parser.add_argument(
        "--full-test", type=Path, default=DATA_DIR / "test.parquet"
    )
    parser.add_argument(
        "--weight-out",
        type=Path,
        default=DATA_DIR / "train_filtered_20000-mh256.parquet",
    )
    parser.add_argument(
        "--harness-out", type=Path, default=DATA_DIR / "mh_dist_align_256.parquet"
    )
    parser.add_argument(
        "--validation-out", type=Path, default=DATA_DIR / "test_100.parquet"
    )
    parser.add_argument(
        "--test-out",
        type=Path,
        default=DATA_DIR / "final_test_minus_test_100.parquet",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-per-source", type=int, default=100)
    parser.add_argument("--expected-weight-rows", type=int, default=18_946)
    parser.add_argument("--expected-harness-rows", type=int, default=256)
    parser.add_argument("--expected-validation-rows", type=int, default=700)
    parser.add_argument("--expected-test-rows", type=int, default=51_013)
    parser.add_argument(
        "--harness-count",
        action="append",
        metavar="DATA_SOURCE=COUNT",
        help="Repeat to override the historical 200 HotpotQA / 56 NQ allocation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_per_source <= 0:
        raise ValueError("--validation-per-source must be positive")

    weight_source = pd.read_parquet(args.weight_source)
    weight_exclude = pd.read_parquet(args.weight_exclude)
    full_test = pd.read_parquet(args.full_test)

    weight = build_weight_split(weight_source, weight_exclude)
    harness = build_harness_split(
        weight, counts=_parse_counts(args.harness_count), seed=args.seed
    )
    validation = build_validation_split(
        full_test, per_source=args.validation_per_source, seed=args.seed
    )
    test = build_final_test(full_test, validation)

    _expect_rows(weight, args.expected_weight_rows, "RSFT")
    _expect_rows(harness, args.expected_harness_rows, "MH")
    _expect_rows(validation, args.expected_validation_rows, "validation")
    _expect_rows(test, args.expected_test_rows, "held-out test")

    for frame, path in (
        (weight, args.weight_out),
        (harness, args.harness_out),
        (validation, args.validation_out),
        (test, args.test_out),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    print(f"wrote RSFT:      {args.weight_out} ({len(weight):,} rows)")
    print(f"wrote MH:        {args.harness_out} ({len(harness):,} rows)")
    print(f"wrote validation:{args.validation_out} ({len(validation):,} rows)")
    print(f"wrote test:      {args.test_out} ({len(test):,} rows)")


if __name__ == "__main__":
    main()
