"""
blocking.py — Part 1: scalable candidate generation (blocking) for the
Business Entity Resolution Challenge.

Design notes (why this scales to ~24M rows across S1/S2/S3 in ~12GB RAM):
  * All string cleaning is vectorized in Polars (Rust-backed) instead of a
    per-row Python loop. The one unavoidable per-row step (unidecode
    transliteration for Devanagari / accented French text) is applied only
    to the small subset of rows that actually contain non-ASCII characters.
  * Candidate generation is an INVERTED INDEX (dict: key -> capped list of
    ids), not a SQL-style join. A join on a generic key (e.g. a 4-letter
    name prefix shared by thousands of businesses) builds a full cross
    product of every matching pair before it can be filtered, which is what
    exhausted memory on the first attempt. The inverted index instead caps
    each bucket at BUCKET_CAP entries as it is built, so a single common
    prefix can never blow up memory, and S1 entities are processed in
    chunks with results flushed to disk immediately, so peak memory is
    bounded by one chunk, not the whole dataset.
  * A cheap RapidFuzz re-rank caps candidates per Source-1 entity, because
    candidate_pairs.tsv now counts toward the final ranking (smaller,
    high-recall candidate sets score better).

Usage:
    python blocking.py --split train --max-candidates 20
    python blocking.py --split test  --max-candidates 20
"""

import argparse
import collections
import sys
import time
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz
from unidecode import unidecode

sys.path.insert(0, str(Path(__file__).parent))
from normalize import _NAME_ABBREV, _ADDR_ABBREV, _LEGAL_SUFFIXES  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]  # .../student_resource
DATA = ROOT / "dataset"
OUT = ROOT / "output"
CACHE = ROOT / "code" / "business_entity_resolution" / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

NAME_KEY_LEN = 4
BUCKET_CAP = 150       # max ids stored per (country, key) bucket while indexing
TOKEN_BUCKET_CAP = 300  # token buckets are broader, allow a bit more before truncating
RAW_CAND_CAP = 200     # max raw candidates considered per S1 row before scoring
CHUNK_SIZE = 20_000    # S1 rows processed per batch (bounds peak memory)
N_SIG_TOKENS = 2       # number of longest/most-distinctive words used as token-blocking keys
MIN_TOKEN_LEN = 3      # ignore very short words (low signal, huge buckets) as token keys


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Vectorized cleaning
# ---------------------------------------------------------------------------

def transliterate_non_ascii(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply unidecode only to rows containing non-ASCII characters."""
    df = df.with_row_index("_idx")
    mask = pl.col(col).fill_null("").str.contains(r"[^\x00-\x7F]")
    subset = df.filter(mask).select("_idx", col)
    if subset.height > 0:
        vals = subset[col].to_list()
        ascii_vals = [unidecode(v) if v is not None else "" for v in vals]
        subset = subset.with_columns(pl.Series("_ascii", ascii_vals))
        df = df.join(subset.select("_idx", "_ascii"), on="_idx", how="left")
        df = df.with_columns(
            pl.when(pl.col("_ascii").is_not_null())
            .then(pl.col("_ascii"))
            .otherwise(pl.col(col))
            .alias(col)
        ).drop("_ascii")
    return df.drop("_idx")


def clean_name_expr(col: str) -> pl.Expr:
    e = pl.col(col).fill_null("").str.to_lowercase()
    e = e.str.replace(r"^[\-\*<>~\.\s]+", "")
    e = e.str.replace_all(r"[^a-z0-9&\s]", " ")
    for pat, repl in _NAME_ABBREV.items():
        e = e.str.replace_all(pat, repl)
    e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    return e.alias("clean_name")


def clean_addr_expr(col: str) -> pl.Expr:
    e = pl.col(col).fill_null("").str.to_lowercase()
    e = e.str.replace_all(r"[^a-z0-9\s,]", " ")
    for pat, repl in _ADDR_ABBREV.items():
        e = e.str.replace_all(pat, repl)
    e = e.str.replace_all(r"\s+", " ").str.strip_chars()
    return e.alias("clean_addr")


def add_derived_columns(df: pl.DataFrame) -> pl.DataFrame:
    df = transliterate_non_ascii(df, "business_name")
    df = transliterate_non_ascii(df, "business_address")

    df = df.with_columns(
        [clean_name_expr("business_name"), clean_addr_expr("business_address")]
    )

    legal_list = list(_LEGAL_SUFFIXES)
    df = df.with_columns(
        pl.col("clean_name")
        .str.split(" ")
        .list.eval(pl.element().filter(~pl.element().is_in(legal_list)))
        .list.join(" ")
        .alias("name_core")
    )

    df = df.with_columns(
        [
            pl.col("name_core")
            .str.replace_all(r"[^a-z0-9]", "")
            .str.slice(0, NAME_KEY_LEN)
            .alias("name_key"),
            pl.col("clean_addr")
            .str.extract(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b", 1)
            .fill_null("")
            .alias("pin"),
        ]
    )

    # Significant tokens: the N longest words (>= MIN_TOKEN_LEN chars) in the
    # core name, sorted alphabetically so token ORDER doesn't matter — this
    # is what lets "Balaji Traders" and "Traders Balaji" still block together,
    # and lets a typo in one word still match via the other word.
    df = df.with_columns(
        pl.col("name_core")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() >= MIN_TOKEN_LEN))
        .list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True))
        .list.head(N_SIG_TOKENS)
        .alias("sig_tokens")
    )

    # keep only what downstream needs — drops the raw/clean text columns to
    # save memory; Part 2 (features) re-derives clean text itself.
    return df.select("entity_id", "country", "name_key", "pin", "name_core", "sig_tokens")


def load_source(split: str, source: str, limit: int | None = None, use_cache: bool = True) -> pl.DataFrame:
    path = DATA / split / f"{split}_{source}.tsv"
    cache_path = CACHE / f"{split}_{source}_clean.parquet"

    if limit is None and use_cache and cache_path.exists():
        log(f"loading cached {cache_path.name} ...")
        df = pl.read_parquet(cache_path)
        log(f"{cache_path.name}: {df.height:,} rows ready (from cache)")
        return df

    log(f"reading {path.name} ...")
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=0,  # read everything as Utf8, safest for messy text
        quote_char=None,
        n_rows=limit,
    )
    df = add_derived_columns(df)
    log(f"{path.name}: {df.height:,} rows ready")

    if limit is None and use_cache:
        df.write_parquet(cache_path)
        log(f"cached cleaned data -> {cache_path.name}")

    return df


# ---------------------------------------------------------------------------
# Inverted-index candidate generation (memory-safe: no join / cross product)
# ---------------------------------------------------------------------------

def build_index(other: pl.DataFrame, key_col: str, cap: int = BUCKET_CAP) -> dict:
    """dict[(country, key)] -> capped list of entity_id, skipping blank keys."""
    idx: dict = collections.defaultdict(list)
    for country, key, eid in zip(
        other["country"].to_list(), other[key_col].to_list(), other["entity_id"].to_list()
    ):
        if not key:
            continue
        bucket = idx[(country, key)]
        if len(bucket) < cap:
            bucket.append(eid)
    return idx


def build_token_index(other: pl.DataFrame, cap: int = TOKEN_BUCKET_CAP) -> dict:
    """dict[(country, token)] -> capped list of entity_id, one entry per
    significant word in the business name (not just the whole-name prefix).
    This is what lets word-reordering, and typos confined to one word,
    still produce a match."""
    idx: dict = collections.defaultdict(list)
    for country, tokens, eid in zip(
        other["country"].to_list(), other["sig_tokens"].to_list(), other["entity_id"].to_list()
    ):
        if not tokens:
            continue
        for tok in tokens:
            bucket = idx[(country, tok)]
            if len(bucket) < cap:
                bucket.append(eid)
    return idx


def process_and_write(s1: pl.DataFrame, other: pl.DataFrame, out_path: Path, max_candidates: int) -> None:
    log("building name-key index ...")
    name_idx = build_index(other, "name_key")
    log(f"name-key index: {len(name_idx):,} buckets")
    log("building pin index ...")
    pin_idx = build_index(other, "pin")
    log(f"pin index: {len(pin_idx):,} buckets")
    log("building token index ...")
    token_idx = build_token_index(other)
    log(f"token index: {len(token_idx):,} buckets")

    core_lookup = dict(zip(other["entity_id"].to_list(), other["name_core"].to_list()))

    s1_ids = s1["entity_id"].to_list()
    s1_country = s1["country"].to_list()
    s1_nkey = s1["name_key"].to_list()
    s1_pin = s1["pin"].to_list()
    s1_core = s1["name_core"].to_list()
    s1_tokens = s1["sig_tokens"].to_list()
    n = len(s1_ids)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        t0 = time.time()
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            for i in range(start, end):
                eid, country, nkey, pin, core, tokens = (
                    s1_ids[i], s1_country[i], s1_nkey[i], s1_pin[i], s1_core[i], s1_tokens[i]
                )
                cand_ids = set()
                if nkey:
                    cand_ids.update(name_idx.get((country, nkey), ()))
                if pin:
                    cand_ids.update(pin_idx.get((country, pin), ()))
                if tokens:
                    for tok in tokens:
                        cand_ids.update(token_idx.get((country, tok), ()))

                if not cand_ids:
                    f.write(f"{eid}\t\n")
                    continue

                # Score every raw candidate (never truncate a plain set by
                # slicing — set iteration order is arbitrary, so slicing
                # first silently drops true matches at random). The bucket
                # caps already bound how large cand_ids can get, so scoring
                # all of them is cheap and safe.
                scored = sorted(
                    (
                        (fuzz.token_sort_ratio(core, core_lookup.get(cid, "")), cid)
                        for cid in cand_ids
                    ),
                    key=lambda t: t[0],
                    reverse=True,
                )
                top = [cid for _, cid in scored[:max_candidates]]
                f.write(f"{eid}\t{','.join(top)}\n")

            log(f"  processed {end:,}/{n:,} S1 rows ({time.time() - t0:.1f}s elapsed)")

    log(f"wrote {out_path}")


def eval_recall(out_path: Path, gt_path: Path) -> None:
    gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=0)
    gt_map: dict = {}
    n_true = 0
    for sid, matched in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        s = set(matched.split(",")) if matched else set()
        gt_map[sid] = s
        n_true += len(s)

    cand = pl.read_csv(out_path, separator="\t", infer_schema_length=0)
    total_cands = 0
    entities_with_cands = 0
    found = 0
    for sid, cands in zip(cand["source1_entity_id"].to_list(), cand["candidate_entity_ids"].to_list()):
        c = set(cands.split(",")) if cands else set()
        if c:
            entities_with_cands += 1
            total_cands += len(c)
        true = gt_map.get(sid, set())
        found += len(true & c)

    recall = found / n_true if n_true else 1.0
    avg_cands = total_cands / entities_with_cands if entities_with_cands else 0

    log(f"TRUE matches in ground truth : {n_true:,}")
    log(f"TRUE matches captured by blocking : {found:,}")
    log(f"BLOCKING RECALL (upper bound on final F0.5) : {recall:.4f}")
    log(f"avg candidates per S1 entity (with >=1 candidate) : {avg_cands:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="row limit per source file, for smoke testing")
    args = ap.parse_args()

    t0 = time.time()
    s1 = load_source(args.split, "source1", args.limit)
    s2 = load_source(args.split, "source2", args.limit)
    s3 = load_source(args.split, "source3", args.limit)

    log("combining S2 + S3 into one candidate pool ...")
    other = pl.concat([s2, s3])
    del s2, s3

    out_path = OUT / f"candidate_pairs_{args.split}.tsv"
    process_and_write(s1, other, out_path, args.max_candidates)

    if args.split == "train":
        gt_path = DATA / "train" / "train_ground_truth.tsv"
        eval_recall(out_path, gt_path)

    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
