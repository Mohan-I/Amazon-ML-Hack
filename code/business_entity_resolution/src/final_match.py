"""
final_match.py — Person 3: apply tuned threshold -> matching_results.tsv.

Reads:
  - output/pair_scores_test.tsv         (Person 2's model scores on test candidates)
  - output/candidate_pairs_test.tsv     (Person 1's blocking output for test)
  - output/best_threshold.json          (Person 3's tuned threshold)

Writes:
  - output/matching_results.tsv         (scored on leaderboard)
  - output/candidate_pairs.tsv          (blocking set for final package)

Decision logic:
  For each Source-1 entity, keep every candidate with probability >= threshold.
  If none pass -> empty matched_entity_ids (singleton prediction).
  Enforce: no duplicates, S2-/S3- IDs only, one row per S1.

Usage:
  python final_match.py                       # uses best_threshold.json
  python final_match.py --threshold 0.62      # manual override
  python final_match.py --conservative        # use conservative tie-break t
"""

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_threshold(args) -> float:
    if args.threshold is not None:
        log(f"using manual threshold: {args.threshold}")
        return float(args.threshold)
    jpath = OUT / "best_threshold.json"
    if not jpath.exists():
        log(f"WARNING: {jpath.name} not found. Defaulting to 0.60 "
            f"(Person 2's diagnostic best).")
        return 0.60
    meta = json.loads(jpath.read_text())
    key = "threshold_conservative" if args.conservative else "threshold"
    t = float(meta.get(key, meta.get("threshold", 0.60)))
    log(f"loaded threshold from {jpath.name} [{key}] = {t}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, default=str(OUT / "pair_scores_test.tsv"))
    ap.add_argument("--candidates", type=str, default=str(OUT / "candidate_pairs_test.tsv"))
    ap.add_argument("--out-matching", type=str, default=str(OUT / "matching_results.tsv"))
    ap.add_argument("--out-candidates", type=str, default=str(OUT / "candidate_pairs.tsv"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--conservative", action="store_true",
                    help="use the higher conservative tie-break threshold")
    args = ap.parse_args()

    t0 = time.time()
    threshold = load_threshold(args)

    # ---- 1. Load candidate universe (S1 order + full candidate set) ----
    log(f"loading candidate file {Path(args.candidates).name} ...")
    cand = pl.read_csv(args.candidates, separator="\t", infer_schema_length=0)
    s1_order = cand["source1_entity_id"].to_list()
    cand_map = dict(zip(s1_order, cand["candidate_entity_ids"].to_list()))
    log(f"  {len(s1_order):,} S1 entities, candidate universe loaded")

    # ---- 2. Load model scores ----
    log(f"loading model scores {Path(args.scores).name} ...")
    scores = pl.read_csv(args.scores, separator="\t", infer_schema_length=0)
    scores = scores.with_columns(pl.col("match_probability").cast(pl.Float64))
    log(f"  {scores.height:,} scored pairs")

    # ---- 3. Decision logic: keep pairs with prob >= threshold ----
    kept = scores.filter(pl.col("match_probability") >= threshold)
    log(f"  {kept.height:,} pairs pass threshold {threshold} "
        f"({kept.height / max(scores.height, 1) * 100:.2f}% of candidates)")

    # Sanity: enforce S2-/S3- only, drop any S1 self-matches
    bad_prefix = kept.filter(~pl.col("candidate_entity_id").str.starts_with("S2-") &
                             ~pl.col("candidate_entity_id").str.starts_with("S3-"))
    if bad_prefix.height:
        log(f"  WARNING: dropped {bad_prefix.height:,} non-S2/S3 candidate IDs")
        kept = kept.filter(pl.col("candidate_entity_id").str.starts_with("S2-") |
                           pl.col("candidate_entity_id").str.starts_with("S3-"))

    # Group matches per S1, de-duplicate IDs within a list
    grouped = kept.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").unique().alias("mids")
    )
    preds = {sid: sorted(set(mids)) for sid, mids in
             zip(grouped["source1_entity_id"].to_list(), grouped["mids"].to_list())}

    # ---- 4. Write matching_results.tsv (one row per S1, empty when singleton) ----
    log(f"writing {Path(args.out_matching).name} ...")
    n_singletons = 0
    n_matched = 0
    with open(args.out_matching, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_order:
            mids = preds.get(sid, [])
            if mids:
                n_matched += 1
                f.write(f"{sid}\t{','.join(mids)}\n")
            else:
                n_singletons += 1
                f.write(f"{sid}\t\n")
    log(f"  wrote {len(s1_order):,} rows "
        f"({n_matched:,} matched, {n_singletons:,} singletons)")

    # ---- 5. Write candidate_pairs.tsv (renamed from test blocking output) ----
    log(f"writing {Path(args.out_candidates).name} ...")
    with open(args.out_candidates, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            cids = cand_map.get(sid, "")
            f.write(f"{sid}\t{cids if cids else ''}\n")
    log(f"  wrote candidate set")

    # ---- 6. Subset invariant: every matched ID must appear in candidates ----
    violations = 0
    for sid, mids in preds.items():
        cset = set((cand_map.get(sid) or "").split(",")) if cand_map.get(sid) else set()
        cset.discard("")
        if set(mids) - cset:
            violations += 1
    if violations:
        log(f"  WARNING: {violations:,} S1 entities have matches outside candidates "
            f"(pipeline bug — validator will warn)")
    else:
        log(f"  subset check PASS: all matches ⊆ candidates")

    log(f"done in {time.time() - t0:.1f}s "
        f"(threshold={threshold}, matched={n_matched:,}, singletons={n_singletons:,})")


if __name__ == "__main__":
    main()