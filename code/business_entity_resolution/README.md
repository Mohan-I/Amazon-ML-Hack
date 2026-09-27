# Business Entity Resolution — Pipeline

Reproduces `output/candidate_pairs_{train,test}.tsv` and
`output/matching_results.tsv` from the raw dataset.

## Setup

```bash
pip install -r requirements.txt
```

Requires the challenge dataset placed at `../../dataset/{train,test}/*.tsv`
relative to this README (i.e. the standard `student_resource/dataset/` layout).

## Pipeline (run in order)

### 1. Blocking — candidate generation

```bash
cd src
python blocking.py --split train --max-candidates 20
python blocking.py --split test  --max-candidates 20
```

- Cleans names/addresses (`normalize.py`): lowercasing, abbreviation
  expansion (Pvt->private, Rd->road), unidecode transliteration for
  Devanagari/accented text, PIN/ZIP extraction.
- Builds a memory-safe inverted index (name-prefix key, PIN key, and
  per-token key) instead of an all-pairs join, so it scales to the
  ~24M-row dataset without exhausting RAM.
- Re-ranks each S1 entity's raw candidates with RapidFuzz and keeps the
  top `--max-candidates` (candidate_pairs.tsv size counts toward the
  final ranking, so this is deliberately capped).
- Prints blocking recall against `train_ground_truth.tsv` for the train
  split (the ceiling on the whole pipeline's achievable F0.5).
- Caches cleaned data to `cache/*.parquet` so re-runs skip the slow raw
  TSV parse.

Output: `output/candidate_pairs_train.tsv`, `output/candidate_pairs_test.tsv`.

### 2. Matching — threshold tuning + scoring

```bash
python match.py tune                    # sweeps thresholds on TRAIN, picks the
                                         # one maximizing macro F0.5, saves it to
                                         # output/best_threshold.txt
python match.py apply --split test      # scores TEST candidates and writes
                                         # output/matching_results.tsv
```

Current scorer (`match.py`) is a RapidFuzz name-similarity + PIN-match
rule, used as a placeholder while the LightGBM feature-based model is
being built. Swapping in a trained model only requires replacing
`score_pairs()`'s scoring logic — the threshold-tuning and output-writing
code is unchanged either way.

### 3. Validate before submitting

```bash
cd ..                                    # back to student_resource/
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs_test.tsv \
    --test-dir dataset/test --check-ids
```

## Files

```
src/
  normalize.py   # text cleaning (names, addresses, PIN extraction)
  blocking.py    # candidate generation (blocking)
  metric.py      # official F_0.5 scorer
  match.py       # threshold tuning + final matching_results.tsv writer
```
