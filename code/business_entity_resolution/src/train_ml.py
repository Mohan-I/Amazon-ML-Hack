"""
train_ml.py — Train LightGBM Matching Model for Entity Resolution (Person 2).

Pipeline:
  1. Ingest candidate pairs produced by Person 1's blocking.py
  2. Extract comprehensive pairwise similarity features (features.py)
  3. Attach ground truth labels (data_utils.py)
  4. Perform leak-free entity-level train/validation split
  5. Train LightGBM binary classifier
  6. Evaluate Precision, Recall, F0.5, TP, FP, FN, and official per-entity macro F0.5
  7. Save the trained model artifact for downstream inference

Usage:
  python train_ml.py
  python train_ml.py --limit 10000   # fast test on first 10,000 candidate pairs
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import (  # noqa: E402
    FEATURE_NAMES,
    load_and_preprocess_records,
    extract_features_for_pairs,
)
from data_utils import (  # noqa: E402
    load_ground_truth,
    load_candidate_pairs,
    create_training_labels,
    entity_level_train_val_split,
)
from metric import macro_f05  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
OUT = ROOT / "output"
MODELS = ROOT / "code" / "business_entity_resolution" / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calculate_f05(precision: float, recall: float) -> float:
    if precision <= 0 or recall <= 0:
        return 0.0
    beta_sq = 0.25  # 0.5^2
    denom = beta_sq * precision + recall
    return (1.25 * precision * recall) / denom if denom > 0 else 0.0


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pairs: List[tuple],
    gt_map: Dict[str, Set[str]],
    val_s1_set: Set[str],
) -> None:
    """Evaluates validation performance across multiple candidate thresholds."""
    log("\n" + "=" * 80)
    log("VALIDATION EVALUATION REPORT (Person 2 Model Diagnostics)")
    log("=" * 80)
    log(f"{'Threshold':>10} | {'Prec':>8} | {'Recall':>8} | {'Pair F0.5':>10} | {'Macro F0.5':>10} | {'TP':>7} | {'FP':>7} | {'FN':>7} | {'Pred Pos':>8}")
    log("-" * 80)

    # Build per-entity candidate probability map for fast macro F0.5 calculation
    val_s1_cands = {}
    for (sid, cid), prob, label in zip(pairs, y_prob, y_true):
        if sid not in val_s1_cands:
            val_s1_cands[sid] = []
        val_s1_cands[sid].append((cid, prob))

    val_gt = {sid: gt_map.get(sid, set()) for sid in val_s1_set}

    best_thresh = 0.5
    best_macro = -1.0

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        preds = (y_prob >= t).astype(np.int32)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        pred_pos = tp + fp

        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = calculate_f05(prec, rec)

        # Entity macro F0.5
        ent_preds = {}
        for sid, cand_list in val_s1_cands.items():
            ent_preds[sid] = {cid for cid, prob in cand_list if prob >= t}

        mf05 = macro_f05(ent_preds, val_gt)
        if mf05 > best_macro:
            best_macro = mf05
            best_thresh = t

        log(f"{t:10.2f} | {prec:8.4f} | {rec:8.4f} | {f05:10.4f} | {mf05:10.4f} | {tp:7d} | {fp:7d} | {fn:7d} | {pred_pos:8d}")

    log("=" * 80)
    log(f"Diagnostic best reference threshold: {best_thresh:.2f} (Macro F0.5 = {best_macro:.4f})")
    log("Note: Person 3 is responsible for final threshold tuning and submission generation.")
    log("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Business Entity Resolution Model")
    parser.add_argument("--candidate-file", type=str, default=str(OUT / "candidate_pairs_train.tsv"))
    parser.add_argument("--gt-file", type=str, default=str(DATA / "train" / "train_ground_truth.tsv"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of candidate pairs for testing")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default=str(MODELS / "lgbm_matcher.txt"))
    args = parser.parse_args()

    t_start = time.time()
    cand_path = Path(args.candidate_file)
    gt_path = Path(args.gt_file)

    if not cand_path.exists():
        log(f"ERROR: Candidate file not found at {cand_path}")
        log("Please run Person 1's blocking script first:")
        log("  python code/business_entity_resolution/src/blocking.py --split train")
        sys.exit(1)

    if not gt_path.exists():
        log(f"ERROR: Ground truth file not found at {gt_path}")
        sys.exit(1)

    # 1. Load candidate pairs
    pairs, universe_s1 = load_candidate_pairs(cand_path)
    if args.limit and len(pairs) > args.limit:
        log(f"Limiting candidate pairs to {args.limit:,} rows for rapid testing ...")
        pairs = pairs[:args.limit]

    # 2. Load ground truth and create binary targets
    gt_map = load_ground_truth(gt_path)
    labels = create_training_labels(pairs, gt_map)

    # 3. Collect required entities and load preprocessed records for feature lookup
    needed_eids = {sid for sid, _ in pairs} | {cid for _, cid in pairs}
    log(f"Unique entities required for features: {len(needed_eids):,}")
    record_lookup = load_and_preprocess_records(split="train", needed_eids=needed_eids)

    # 4. Extract pairwise features
    X = extract_features_for_pairs(pairs, record_lookup)

    # Sanity checks on feature matrix
    assert np.isfinite(X).all(), "Feature matrix contains NaN or Inf values!"
    assert X.shape[0] == len(labels), "Row count mismatch between features and labels!"
    log(f"Feature matrix shape: {X.shape} (All values finite numeric)")

    # 5. Entity-level train/validation split (strictly leak-free)
    train_idx, val_idx, train_s1_set, val_s1_set = entity_level_train_val_split(
        pairs, val_ratio=args.val_ratio, seed=args.seed
    )

    X_train, y_train = X[train_idx], labels[train_idx]
    X_val, y_val = X[val_idx], labels[val_idx]
    val_pairs = [pairs[i] for i in val_idx]

    log(f"Training set:   {X_train.shape[0]:,} pairs, positive rate = {y_train.mean()*100:.2f}%")
    log(f"Validation set: {X_val.shape[0]:,} pairs, positive rate = {y_val.mean()*100:.2f}%")

    # 6. Train LightGBM model
    log(f"Training LightGBM model (n_estimators={args.n_estimators}, lr={args.learning_rate}, num_leaves={args.num_leaves}) ...")
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )

    t0_fit = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    log(f"Model training complete in {time.time() - t0_fit:.1f}s")

    # 7. Validation evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    evaluate_predictions(y_val, val_probs, val_pairs, gt_map, val_s1_set)

    # 8. Feature importances
    importances = model.feature_importances_
    sorted_feat_idx = np.argsort(importances)[::-1]
    log("Top 15 Most Important Features:")
    for rank, idx in enumerate(sorted_feat_idx[:15], 1):
        log(f"  {rank:2d}. {FEATURE_NAMES[idx]:<28} : {importances[idx]:d}")

    # 9. Save model artifact
    out_model_path = Path(args.model_out)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_model_path))
    log(f"Saved trained LightGBM model to {out_model_path}")
    log(f"Total pipeline elapsed time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
