"""
threshold_tune.py — Person 3: threshold tuning + final decision logic.

Given Person 2's pair_scores_train.tsv (source1_entity_id, candidate_entity_id,
match_probability), tunes the probability threshold to maximize the official
macro F_0.5 metric (precision-weighted, singletons included).

Outputs:
  - output/best_threshold.json   (chosen threshold + validation metrics)
  - output/threshold_curve.csv   (full sweep for the docs)
  - output/threshold_curve.png   (precision/recall/F0.5 vs threshold)

Usage:
  python threshold_tune.py --scores output/pair_scores_train.tsv
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from metric import macro_f05  # noqa: E402
from data_utils import load_ground_truth, entity_level_train_val_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_entity_metrics(preds: dict, gt: dict):
    """Returns (macro_f05, mean_precision, mean_recall, tp, fp, fn, n_entities)."""
    tp = fp = fn = 0
    prec_list, rec_list = [], []
    for sid, true in gt.items():
        pred = preds.get(sid, set())
        tp_e = len(pred & true)
        fp_e = len(pred - true)
        fn_e = len(true - pred)
        tp += tp_e; fp += fp_e; fn += fn_e
        if pred:
            prec_list.append(tp_e / len(pred))
        if true:
            rec_list.append(tp_e / len(true))
    return (
        macro_f05(preds, gt),
        float(np.mean(prec_list)) if prec_list else 0.0,
        float(np.mean(rec_list)) if rec_list else 0.0,
        tp, fp, fn, len(gt),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, default=str(OUT / "pair_scores_train.tsv"))
    ap.add_argument("--candidate-file", type=str, default=str(OUT / "candidate_pairs_train.tsv"))
    ap.add_argument("--gt-file", type=str, default=str(DATA / "train" / "train_ground_truth.tsv"))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=str, default="0.05,0.95,0.025",
                    help="start,end,step for threshold sweep")
    args = ap.parse_args()

    t0 = time.time()
    log(f"loading scored pairs from {args.scores} ...")
    scored = pl.read_csv(args.scores, separator="\t", infer_schema_length=0)
    scored = scored.with_columns(pl.col("match_probability").cast(pl.Float64))
    log(f"  {scored.height:,} scored pairs")

    log("loading candidate universe (to preserve S1 entity order) ...")
    cand = pl.read_csv(args.candidate_file, separator="\t", infer_schema_length=0)
    universe_s1 = cand["source1_entity_id"].to_list()

    log("loading ground truth ...")
    gt = load_ground_truth(Path(args.gt_file))

    # Reproduce the EXACT entity-level val split from train_ml.py (same seed)
    pairs_for_split = list(zip(
        scored["source1_entity_id"].to_list(),
        scored["candidate_entity_id"].to_list(),
    ))
    _, val_idx, _, val_s1_set = entity_level_train_val_split(
        pairs_for_split, val_ratio=args.val_ratio, seed=args.seed
    )
    log(f"validation S1 entities (leak-free): {len(val_s1_set):,}")

    val_scored = scored.filter(pl.col("source1_entity_id").is_in(list(val_s1_set)))
    val_gt = {sid: gt.get(sid, set()) for sid in val_s1_set}
    log(f"  {val_scored.height:,} val scored pairs")

    # Pre-group by S1 for fast per-threshold recomputation
    s1_ids = val_scored["source1_entity_id"].to_list()
    c_ids = val_scored["candidate_entity_id"].to_list()
    probs = val_scored["match_probability"].to_list()
    by_s1 = {}
    for sid, cid, p in zip(s1_ids, c_ids, probs):
        by_s1.setdefault(sid, []).append((cid, p))

    start, end, step = [float(x) for x in args.grid.split(",")]
    thresholds = np.arange(start, end + 1e-9, step)

    log(f"sweeping {len(thresholds)} thresholds from {start} to {end} ...")
    rows = []
    best = {"threshold": 0.5, "macro_f05": -1.0}
    for t in thresholds:
        preds = {}
        for sid in val_s1_set:
            preds[sid] = {cid for cid, p in by_s1.get(sid, []) if p >= t}
        mf05, mean_p, mean_r, tp, fp, fn, n = per_entity_metrics(preds, val_gt)
        rows.append({
            "threshold": round(float(t), 4),
            "macro_f05": mf05,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
            "pair_tp": tp, "pair_fp": fp, "pair_fn": fn,
            "entities": n,
        })
        log(f"  t={t:.3f}  macro_F0.5={mf05:.4f}  P={mean_p:.4f}  R={mean_r:.4f}  TP={tp}  FP={fp}  FN={fn}")
        if mf05 > best["macro_f05"]:
            best = {"threshold": round(float(t), 4), "macro_f05": mf05,
                    "mean_precision": mean_p, "mean_recall": mean_r}

    curve = pl.DataFrame(rows)
    curve.write_csv(OUT / "threshold_curve.csv")
    log(f"saved threshold curve -> output/threshold_curve.csv")

    # Optional plot (matplotlib is optional; skip gracefully)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curve["threshold"], curve["macro_f05"], label="Macro F0.5", lw=2)
        ax.plot(curve["threshold"], curve["mean_precision"], label="Mean Precision", ls="--")
        ax.plot(curve["threshold"], curve["mean_recall"], label="Mean Recall", ls=":")
        ax.axvline(best["threshold"], color="red", alpha=0.4,
                   label=f"Chosen t={best['threshold']:.3f}")
        ax.set_xlabel("Probability threshold"); ax.set_ylabel("Score")
        ax.set_title("Validation: Macro F0.5 vs Threshold (ML SQUAD)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "threshold_curve.png", dpi=130)
        log("saved plot -> output/threshold_curve.png")
    except Exception as e:
        log(f"plot skipped ({e})")

    # Conservative bias: if two thresholds tie within 0.001 macro F0.5,
    # prefer the HIGHER one (F0.5 rewards precision -> fewer false merges).
    near = curve.filter(pl.col("macro_f05") >= best["macro_f05"] - 0.001)
    conservative = float(near["threshold"].max())
    best["threshold_conservative"] = conservative
    best["note"] = (
        f"Chose t={best['threshold']} (max macro F0.5). Conservative tie-break "
        f"t={conservative} available if leaderboard favors precision."
    )

    with open(OUT / "best_threshold.json", "w") as f:
        json.dump(best, f, indent=2)
    log(f"BEST THRESHOLD = {best['threshold']}  macro_F0.5={best['macro_f05']:.4f}  "
        f"P={best['mean_precision']:.4f}  R={best['mean_recall']:.4f}")
    log(f"saved -> output/best_threshold.json")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()