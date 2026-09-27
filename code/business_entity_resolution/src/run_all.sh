#!/usr/bin/env bash
# run_all.sh — ML SQUAD end-to-end reproduction.
# Reproduces output/matching_results.tsv and output/candidate_pairs.tsv
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC_DIR"

PY="${PYTHON:-python}"
MAX_CAND="${MAX_CAND:-20}"

echo "============================================================"
echo "   ML SQUAD — Amazon ML Challenge 2026 E2E Reproduction"
echo "   Max candidates per S1: $MAX_CAND"
echo "============================================================"

# --- Step 1: Blocking (candidate generation) ------------------------------
echo ""
echo "[1/5] Blocking — TRAIN (for training + threshold tuning)"
$PY blocking.py --split train --max-candidates "$MAX_CAND"

echo ""
echo "[2/5] Blocking — TEST (for submission)"
$PY blocking.py --split test  --max-candidates "$MAX_CAND"

# --- Step 2: Train LightGBM model -----------------------------------------
echo ""
echo "[3/5] Training LightGBM matcher"
$PY train_ml.py

# --- Step 3: Threshold tuning --------------------------------------------
echo ""
echo "[4/5] Tuning threshold for macro F0.5"
$PY threshold_tune.py

# --- Step 4: Score test candidates ---------------------------------------
echo ""
echo "[5/5] Scoring TEST candidates + generating submission"
$PY score_ml.py
$PY final_match.py

# --- Step 5: Official validator ------------------------------------------
echo ""
echo "============================================================"
echo "   Running official submission validator"
echo "============================================================"
PROJECT_ROOT="$(cd "$SRC_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

$PY utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids

echo ""
echo "DONE. Submission files:"
echo "  - output/matching_results.tsv"
echo "  - output/candidate_pairs.tsv"