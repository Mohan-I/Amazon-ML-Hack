"""
match.py — Part 2 stand-in: a rule-based matcher on top of the blocking
candidates, used to get a real submission in while the ML model (LightGBM)
is being built separately. Same input/output contract, so it's a drop-in
swap later: candidate_pairs.tsv -> matching_results.tsv.

Score = RapidFuzz name similarity (token_sort_ratio on the core name),
with a bonus for a shared PIN/ZIP code (address agreement).

Modes:
  tune   — sweep thresholds on TRAIN, using the real ground truth, report
           the threshold that maximizes macro F_0.5 (score you'd actually
           get on the leaderboard's metric).
  apply  — use a fixed threshold on TEST (no ground truth available) to
           produce output/matching_results.tsv.
"""

import argparse
import sys
import time
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from metric import macro_f05  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
OUT = ROOT / "output"
CACHE = ROOT / "code" / "business_entity_resolution" / "cache"

PIN_BONUS = 15  # added to the name-similarity score when both PINs match and are non-empty


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_lookup(split: str) -> dict:
    """entity_id -> (name_core, pin) for S1 + S2 + S3 of this split, from the
    parquet cache blocking.py already built."""
    lookup = {}
    for source in ("source1", "source2", "source3"):
        path = CACHE / f"{split}_{source}_clean.parquet"
        df = pl.read_parquet(path)
        for eid, core, pin in zip(
            df["entity_id"].to_list(), df["name_core"].to_list(), df["pin"].to_list()
        ):
            lookup[eid] = (core, pin)
    return lookup


def load_candidates(split: str) -> pl.DataFrame:
    path = OUT / f"candidate_pairs_{split}.tsv"
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def score_pairs(cand_df: pl.DataFrame, lookup: dict) -> pl.DataFrame:
    """Returns a DataFrame [source1_entity_id, candidate_entity_id, score].
    RapidFuzz scoring itself is an unavoidable O(pairs) pass (done once,
    here) — the part we must NOT redo per-threshold is anything downstream
    of this (that's what made the naive tune() loop take hours)."""
    sids, cids, sims = [], [], []
    for sid, cands in zip(
        cand_df["source1_entity_id"].to_list(), cand_df["candidate_entity_ids"].to_list()
    ):
        if not cands:
            continue
        s1_core, s1_pin = lookup.get(sid, ("", ""))
        for cid in cands.split(","):
            c_core, c_pin = lookup.get(cid, ("", ""))
            sim = fuzz.token_sort_ratio(s1_core, c_core)
            if s1_pin and c_pin and s1_pin == c_pin:
                sim = min(100, sim + PIN_BONUS)
            sids.append(sid)
            cids.append(cid)
            sims.append(float(sim))
    return pl.DataFrame({"source1_entity_id": sids, "candidate_entity_id": cids, "score": sims})


def load_ground_truth(gt_path: Path) -> dict:
    gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=0)
    out = {}
    for sid, matched in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        out[sid] = set(matched.split(",")) if matched else set()
    return out


def tune(args):
    log("loading train candidates ...")
    cand_df = load_candidates("train")
    universe = cand_df["source1_entity_id"].to_list()
    log(f"{len(universe):,} S1 rows in candidate file")

    scored_cache = CACHE / "train_scored_pairs.parquet"
    if scored_cache.exists():
        log(f"loading cached scored pairs from {scored_cache.name} ...")
        scored_df = pl.read_parquet(scored_cache)
    else:
        log("loading name/pin lookup from cache ...")
        lookup = load_lookup("train")
        log("scoring all candidate pairs (one pass, ~15-20 min for millions of pairs) ...")
        scored_df = score_pairs(cand_df, lookup)
        scored_df.write_parquet(scored_cache)
        log(f"cached scored pairs -> {scored_cache.name}")
    log(f"{scored_df.height:,} scored pairs")

    log("loading ground truth ...")
    gt = load_ground_truth(DATA / "train" / "train_ground_truth.tsv")
    true_count = {sid: len(s) for sid, s in gt.items()}

    log("marking true pairs (vectorized join) ...")
    gt_sids, gt_cids = [], []
    for sid, s in gt.items():
        for cid in s:
            gt_sids.append(sid)
            gt_cids.append(cid)
    gt_df = pl.DataFrame(
        {"source1_entity_id": gt_sids, "candidate_entity_id": gt_cids, "is_true": True}
    )
    scored_df = scored_df.join(
        gt_df, on=["source1_entity_id", "candidate_entity_id"], how="left"
    ).with_columns(pl.col("is_true").fill_null(False))

    universe_df = pl.DataFrame({"source1_entity_id": universe})
    universe_df = universe_df.with_columns(
        pl.col("source1_entity_id")
        .map_elements(lambda s: true_count.get(s, 0), return_dtype=pl.Int64)
        .alias("true_n")
    )

    log("sweeping thresholds (vectorized — seconds per threshold, not minutes) ...")
    best_t, best_f05 = None, -1.0
    for t in range(50, 101, 2):
        filt = scored_df.filter(pl.col("score") >= t)
        agg = filt.group_by("source1_entity_id").agg(
            pl.len().alias("pred_n"), pl.col("is_true").sum().alias("tp")
        )
        merged = universe_df.join(agg, on="source1_entity_id", how="left").with_columns(
            pl.col("pred_n").fill_null(0), pl.col("tp").fill_null(0)
        )
        merged = merged.with_columns(
            [
                pl.when(pl.col("pred_n") > 0)
                .then(pl.col("tp") / pl.col("pred_n"))
                .otherwise(0.0)
                .alias("precision"),
                pl.when(pl.col("true_n") > 0)
                .then(pl.col("tp") / pl.col("true_n"))
                .otherwise(0.0)
                .alias("recall"),
            ]
        )
        merged = merged.with_columns(
            pl.when((pl.col("true_n") == 0) & (pl.col("pred_n") == 0))
            .then(1.0)
            .when((pl.col("true_n") == 0) & (pl.col("pred_n") > 0))
            .then(0.0)
            .when((pl.col("precision") == 0) & (pl.col("recall") == 0))
            .then(0.0)
            .otherwise(
                (1.25 * pl.col("precision") * pl.col("recall"))
                / (0.25 * pl.col("precision") + pl.col("recall"))
            )
            .alias("f05")
        )
        f05 = merged["f05"].mean()
        n_matched = merged.filter(pl.col("pred_n") > 0).height
        log(f"  threshold={t:3d}  macro F0.5={f05:.4f}  (entities matched: {n_matched:,})")
        if f05 > best_f05:
            best_f05, best_t = f05, t

    log(f"BEST THRESHOLD = {best_t}  ->  macro F0.5 = {best_f05:.4f}")
    with open(OUT / "best_threshold.txt", "w") as f:
        f.write(str(best_t))
    log(f"saved best threshold to output/best_threshold.txt")


def apply_(args):
    threshold = args.threshold
    if threshold is None:
        bt_path = OUT / "best_threshold.txt"
        if bt_path.exists():
            threshold = float(bt_path.read_text().strip())
            log(f"using tuned threshold from file: {threshold}")
        else:
            threshold = 80
            log(f"no tuned threshold found, defaulting to {threshold}")

    log(f"loading {args.split} candidates ...")
    cand_df = load_candidates(args.split)
    s1_ids = cand_df["source1_entity_id"].to_list()

    log("loading name/pin lookup from cache ...")
    lookup = load_lookup(args.split)

    log("scoring all candidate pairs ...")
    scored_df = score_pairs(cand_df, lookup)

    preds_df = (
        scored_df.filter(pl.col("score") >= threshold)
        .group_by("source1_entity_id")
        .agg(pl.col("candidate_entity_id").alias("cands"))
    )
    preds = dict(zip(preds_df["source1_entity_id"].to_list(), preds_df["cands"].to_list()))

    out_path = OUT / "matching_results.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            matches = preds.get(sid, [])
            f.write(f"{sid}\t{','.join(matches)}\n")
    log(f"wrote {out_path}")

    if args.split == "train":
        gt = load_ground_truth(DATA / "train" / "train_ground_truth.tsv")
        preds_sets = {k: set(v) for k, v in preds.items()}
        f05 = macro_f05(preds_sets, gt)
        log(f"macro F0.5 on train at threshold {threshold}: {f05:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p_tune = sub.add_parser("tune")
    p_tune.set_defaults(func=tune)

    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--split", choices=["train", "test"], default="test")
    p_apply.add_argument("--threshold", type=float, default=None)
    p_apply.set_defaults(func=apply_)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
