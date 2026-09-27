"""
score_ml.py — Test Candidate Scoring for Business Entity Resolution (Person 2).

Pipeline:
  1. Load Person 1's test candidate pairs (output/candidate_pairs_test.tsv)
  2. Load and preprocess test records (test_source1, test_source2, test_source3)
  3. Extract identical feature vectors using features.py
  4. Load the trained LightGBM model
  5. Compute match probability for every candidate pair
  6. Write output/pair_scores_test.tsv
  7. Run verification checks (row counts, uniqueness, [0, 1] range)

Usage:
  python score_ml.py
"""

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import (  # noqa: E402
    FEATURE_NAMES,
    load_and_preprocess_records,
    extract_features_for_pairs,
)
from data_utils import load_candidate_pairs  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
OUT = ROOT / "output"
MODELS = ROOT / "code" / "business_entity_resolution" / "models"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Score Test Candidate Pairs with LightGBM")
    parser.add_argument("--candidate-file", type=str, default=str(OUT / "candidate_pairs_test.tsv"))
    parser.add_argument("--model-file", type=str, default=str(MODELS / "lgbm_matcher.txt"))
    parser.add_argument("--output-file", type=str, default=str(OUT / "pair_scores_test.tsv"))
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    t_start = time.time()
    cand_path = Path(args.candidate_file)
    model_path = Path(args.model_file)
    out_path = Path(args.output_file)

    if not cand_path.exists():
        log(f"ERROR: Candidate file not found at {cand_path}")
        log("Please run Person 1's blocking script for test first:")
        log("  python code/business_entity_resolution/src/blocking.py --split test")
        sys.exit(1)

    if not model_path.exists():
        log(f"ERROR: Trained model artifact not found at {model_path}")
        log("Please run Person 2's training script first:")
        log("  python code/business_entity_resolution/src/train_ml.py")
        sys.exit(1)

    # 1. Load test candidate pairs
    pairs, universe_s1 = load_candidate_pairs(cand_path)
    n_pairs = len(pairs)
    log(f"Total test candidate pairs to score: {n_pairs:,}")

    if n_pairs == 0:
        log("WARNING: Zero candidate pairs found in candidate file!")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_id\tmatch_probability\n")
        log(f"Wrote empty header to {out_path}")
        return

    # 2. Collect required entities and load test records for lookup
    needed_eids = {sid for sid, _ in pairs} | {cid for _, cid in pairs}
    log(f"Unique entities required for features: {len(needed_eids):,}")
    test_lookup = load_and_preprocess_records(split="test", needed_eids=needed_eids)

    # 3. Load trained model
    log(f"Loading LightGBM model from {model_path.name} ...")
    booster = lgb.Booster(model_file=str(model_path))

    # 4. Extract features in batches and stream predictions to output file
    log(f"Scoring pairs in batches of {args.batch_size:,} and writing to {out_path.name} ...")
    total_written = 0
    all_probs = []

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_id\tmatch_probability\n")

        for start_idx in range(0, n_pairs, args.batch_size):
            end_idx = min(start_idx + args.batch_size, n_pairs)
            batch_pairs = pairs[start_idx:end_idx]

            # Compute features for batch
            X_batch = extract_features_for_pairs(batch_pairs, test_lookup, batch_size=args.batch_size)

            # Predict probabilities
            probs_batch = booster.predict(X_batch)
            all_probs.extend(probs_batch)

            # Write batch
            lines = [
                f"{sid}\t{cid}\t{prob:.6f}\n"
                for (sid, cid), prob in zip(batch_pairs, probs_batch)
            ]
            f.writelines(lines)
            total_written += len(lines)
            log(f"  scored and wrote {total_written:,}/{n_pairs:,} pairs ({time.time() - t_start:.1f}s elapsed)")

    log(f"Successfully wrote {total_written:,} pair scores to {out_path}")

    # 5. Verification checks
    log("\n" + "=" * 60)
    log("VERIFICATION OF TEST SCORE OUTPUT")
    log("=" * 60)
    assert total_written == n_pairs, f"Count mismatch! Expected {n_pairs:,}, wrote {total_written:,}"
    log(f"1. Row count matches candidate pairs: {total_written:,} rows (PASS)")

    probs_arr = np.array(all_probs, dtype=np.float32)
    assert np.all(probs_arr >= 0.0) and np.all(probs_arr <= 1.0), "Probabilities out of [0, 1] range!"
    log(f"2. Probability range valid: min={probs_arr.min():.4f}, max={probs_arr.max():.4f}, mean={probs_arr.mean():.4f} (PASS)")

    # Check pair uniqueness
    unique_pair_set = set(pairs)
    assert len(unique_pair_set) == n_pairs, f"Duplicate candidate pairs detected in input! Unique: {len(unique_pair_set):,}, Total: {n_pairs:,}"
    log(f"3. Pair uniqueness: all {n_pairs:,} pairs are unique (PASS)")
    log("=" * 60 + "\n")
    log(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
