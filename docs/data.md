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
| `data/searchR1_processed/test_100.parquet` | 700 | `D_test`; independently sample 100 rows from each of the seven test sources. This is the evaluation set reported in the paper (`VAL_FILES` of the training driver) |
| `data/searchR1_processed/final_test_minus_test_100.parquet` | 51,013 | auxiliary held-out pool: full test mixture minus `D_test`; not used for the reported results |

The two training-side inputs (the 19,202-row cleaned pool and the 256-row
exclusion set) are intermediate artifacts of the original experiment. Their
generators were not retained and the files are not distributed with this
repository, so `D_weight` and `D_harness` cannot be rebuilt from the public
sources with this repository alone; `test.parquet` is produced by
`scripts/data_process/qa_search_test_merge.py`. The launchers default to the
outputs above: `TRAIN_FILES=train_filtered_20000-mh256.parquet`,
`VAL_FILES=test_100.parquet`, and `eval_dataset_path=mh_dist_align_256.parquet`
in `configs/meta_harness/searchr1.toml`.

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

`meta_harness/config-retool.json` points `eval_dataset_path` at the `D_harness`
file above. The DAPO source is read from the Hub without a pinned revision; the
script checks that de-duplication yields exactly 17,917 rows.

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

The reported experiment was generated with exactly these sizes (16,384 / 256 /
256) and `--seed 42`. Split assignment hashes each position together with the
seed and takes the remainder modulo the sum of the three sizes, so changing any
size changes the partition of every puzzle; keep the defaults to reproduce the
paper's splits. `--test-size 1024` produces a larger auxiliary pool that was
not used for the reported results. The Hub dataset `Lichess/chess-puzzles` is
updated over time and no revision is pinned, so an identical reproduction also
requires the same snapshot of that dataset.

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
