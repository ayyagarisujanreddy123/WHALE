# Dataset preparation

Each domain exposes one canonical data-preparation entrypoint under
`domains/<domain>/scripts/`:

| Domain | Entrypoint |
|---|---|
| Search QA (Search-R1) | `prepare_search_qa_data.py` |
| Mathematical Reasoning (ReTool) | `prepare_math_reasoning_data.py` |
| Chess Puzzles (Lichess) | `prepare_chess_puzzle_data.py` |

All three entrypoints use `--seed 42` by default and prepare the data roles used
by the experiments:

| Role | Symbol | Consumer |
|---|---|---|
| RSFT training | `D_weight` | rollout generation and verifier-accepted SFT updates |
| MH training | `D_harness` | candidate-harness scoring |
| Test | `D_test` | reported held-out evaluation |

Search QA and Mathematical Reasoning historically sampled `D_harness` from
`D_weight`. Chess makes all three roles position-disjoint. The scripts preserve
those protocols rather than imposing a new split policy.

The one-off Search QA and ReTool generators were not retained in their source
Git histories; their deterministic rules were recovered from the retained
parquet artifacts. The reconstructed outputs were compared with those
artifacts, including row order. The Lichess split/de-duplication logic was
retained in its source repository and is consolidated into its canonical
entrypoint here.

Generated parquet files are not committed to this release repository.

---

## 1. Search QA

**Source inputs.** The retained 19,202-row cleaned NQ/HotpotQA training pool,
the earlier balanced 256-row exclusion set, and the full seven-source Search-R1
test mixture.

```bash
cd domains/search_qa
python scripts/prepare_search_qa_data.py
```

The single entrypoint performs both training and evaluation splitting:

| Output | Rows | Construction |
|---|---:|---|
| `data/searchR1_processed/train_filtered_20000-mh256.parquet` | 18,946 | `D_weight`; remove the earlier 128-NQ/128-HotpotQA exclusion set |
| `data/searchR1_processed/mh_dist_align_256.parquet` | 256 | `D_harness`; sample 200 HotpotQA and 56 NQ rows from `D_weight`, then reshuffle |
| `data/searchR1_processed/test_100.parquet` | 700 | validation; independently sample 100 rows from each of seven test sources |
| `data/searchR1_processed/final_test_minus_test_100.parquet` | 51,013 | `D_test`; full test mixture minus validation |

Training rows are matched by `(data_source, extra_info.index)`, with normalized
question text as a defensive exclusion cross-check. Test rows use the
structural key. The script validates the exact row counts, expected seven test
sources, uniqueness, MH membership in RSFT, and zero validation/test overlap.

The original Search-R1 repository did retain
`scripts/data_process/build_final_test_excluding_val.py`; it remains checked in
for provenance and standalone use. Older NQ-only, no-tool, and legacy FlashRAG
builders in `scripts/data_process/` are experiment variants, not additional
steps required by this canonical split.

### Retrieval corpus and index

```bash
python scripts/download.py --save_path data/wiki-18
cat data/wiki-18/part_* > data/wiki-18/e5_Flat.index
gzip -d data/wiki-18/wiki-18.jsonl.gz
```

This downloads the 2018 Wikipedia retrieval corpus and flat FAISS index used by
Search-R1. `search_r1/search/index_builder.py` can rebuild the index, while
`search_r1/search/retrieval_server_async.py` serves it during training and
harness search.

---

## 2. Mathematical Reasoning

**Source.** `BytedTsinghua-SIA/DAPO-Math-17k` for RSFT/MH; AIME 2024 and AIME
2025 for held-out test.

```bash
cd domains/math_reasoning
python scripts/prepare_math_reasoning_data.py
```

| Output | Rows | Construction |
|---|---:|---|
| `data/retool/dapo_math_17k_dedup.parquet` | 17,917 | `D_weight`; keep the first occurrence of each `extra_info.index` |
| `data/retool/meta_harness_dapo17k_dedup_seed42_256.parquet` | 256 | `D_harness`; `Dataset.shuffle(seed=42).select(range(256))` |

The source observed by the experiment exposed the same 17,917 DAPO examples
repeatedly. The script streams the source, rejects conflicting duplicate IDs,
validates the final row count, and annotates the retained MH sample metadata.

The trainer loads `Maxwell-Jia/AIME_2024` and `yentinglin/aime_2025` directly,
30 rows each. They do not undergo a local split or de-duplication step, so the
canonical preparation script does not create redundant test parquets.

---

## 3. Chess Puzzles

**Source.** `Lichess/chess-puzzles`.

```bash
cd domains/chess_puzzles
python scripts/prepare_chess_puzzle_data.py
```

| Output | Rows | Role |
|---|---:|---|
| `data/chess_puzzle/lichess_puzzles_train_16384.parquet` | 16,384 | `D_weight` |
| `data/chess_puzzle/lichess_puzzles_mh_val_256.parquet` | 256 | `D_harness` |
| `data/chess_puzzle/lichess_puzzles_test_256.parquet` | 256 | `D_test` |

The entrypoint consolidates the split/de-duplication implementation formerly
stored in `autoharness_chess_puzzle/grpo_dataset.py`. It reads the Hub parquet
shards in bounded batches, validates legal move sequences with `python-chess`,
keeps solutions between 1 and 9 plies, globally de-duplicates by starting
position and side to move, and uses a second stable hash for deterministic split
assignment. It validates uniqueness and pairwise disjointness before writing;
the manifest records the configuration, output paths, counts, and overlap result.
The old module path remains as a thin compatibility shim and contains no second
copy of the preparation logic.

Pass `--test-size 1024` only to reproduce the larger auxiliary test pool; the
reported experiment uses 256 test positions.

---

## 4. The base harness h0

Every condition in a domain starts from the same `h0`, at
`domains/<domain>/environments/<env>/base_harness.py`. For Search QA it forwards
the model's query unchanged, returns a single passage truncated to 200 tokens,
and terminates after two assistant turns; the shared retrieval and prompt
machinery it builds on lives beside it in `search_r1_env.py`.

## 5. Verifiers

Each domain scores a trajectory with a binary verifier `R(x, tau)` that is fixed
across the whole study; it is used identically for accepting RSFT trajectories,
for scoring harness candidates, and for test evaluation.

| Domain | Verifier |
|---|---|
| Search QA | answer extracted from `<answer>` tags, then judged against the reference answers by `gpt-5.4-mini` at temperature 0 (`verl/utils/reward_score/llm_judge.py`) |
| Mathematical Reasoning | strict `\boxed{...}` match against the ground truth, following DAPO |
| Chess Puzzles | one UCI move parsed and checked for legality with `python-chess`, then compared with the reference move; the reward is 1 only when the whole reference sequence is played |
