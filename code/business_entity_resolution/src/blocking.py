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
from typing import Optional

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
HOUSE_STREET_CAP = 50  # house + street token composite bucket cap
PIN_HOUSE_CAP = 50     # pin + house composite bucket cap
ADDR_TOKEN_CAP = 100   # distinctive address token bucket cap
CHUNK_SIZE = 20_000    # S1 rows processed per batch (bounds peak memory)
N_SIG_TOKENS = 3       # number of distinctive words used as token-blocking keys
MIN_TOKEN_LEN = 3      # ignore very short words (low signal, huge buckets) as token keys

ADDR_STOPWORDS = [
    "road", "street", "avenue", "boulevard", "lane", "drive", "suite", "floor",
    "unit", "apartment", "building", "near", "opposite", "behind", "north", "south",
    "east", "west", "po", "box", "fl", "no", "shop", "dr", "rd", "st", "ave",
    "blvd", "ln", "apt", "first", "second", "third", "block", "sector", "plot",
    "district", "state", "city",
    "nagar", "colony", "marg", "bazaar", "bazar", "complex", "plaza", "chowk",
    "arcade", "mansion", "tower", "towers", "heights", "residency", "park", "garden",
    "market", "center", "centre", "mall", "enclave", "vihar", "puram", "gali",
    "delhi", "mumbai", "bangalore", "bengaluru", "kolkata", "hyderabad", "chennai",
    "pune", "ahmedabad", "jaipur", "surat", "lucknow", "kanpur", "nagpur", "indore",
    "thane", "bhopal", "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik",
    "faridabad", "meerut", "rajkot", "varanasi", "srinagar", "aurangabad", "dhanbad",
    "amritsar", "navi", "allahabad", "prayagraj", "ranchi", "howrah", "coimbatore",
    "jabalpur", "gwalior", "vijayawada", "jodhpur", "madurai", "raipur", "kota",
    "guwahati", "chandigarh", "noida", "gurgaon", "gurugram"
]

GENERIC_NAME_WORDS = [
    "services", "associates", "enterprises", "solutions", "holdings", "group",
    "international", "consulting", "management", "industries", "trading", "products",
    "systems", "global", "company", "business", "corporation", "india", "pvt", "ltd",
    "limited", "private"
]


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

    return df.with_columns(
        [
            pl.col("name_core")
            .str.replace_all(r"[^a-z0-9]", "")
            .str.slice(0, NAME_KEY_LEN)
            .alias("name_key"),
            pl.col("clean_addr")
            .str.extract(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b", 1)
            .fill_null("")
            .alias("pin"),
            pl.col("clean_addr")
            .str.extract(r"\b0*(\d+[-/]?\d*)\b", 1)
            .fill_null("")
            .alias("house_no"),
            pl.col("name_core")
            .str.split(" ")
            .list.eval(pl.element().filter(
                (pl.element().str.len_chars() >= MIN_TOKEN_LEN) &
                (~pl.element().is_in(GENERIC_NAME_WORDS))
            ))
            .list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True))
            .list.head(N_SIG_TOKENS)
            .alias("sig_tokens"),
            pl.col("clean_addr")
            .str.replace_all(r"[^a-z0-9\s]", " ")
            .str.split(" ")
            .list.eval(pl.element().filter(
                (pl.element().str.len_chars() >= 4) &
                (~pl.element().is_in(ADDR_STOPWORDS)) &
                (~pl.element().str.contains(r"^\d+$"))
            ))
            .list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True))
            .list.head(2)
            .alias("addr_tokens"),
        ]
    ).select("entity_id", "country", "name_key", "pin", "name_core", "sig_tokens", "house_no", "addr_tokens")


def load_source(split: str, source: str, limit: Optional[int] = None, use_cache: bool = True) -> pl.DataFrame:
    path = DATA / split / f"{split}_{source}.tsv"
    cache_path = CACHE / f"{split}_{source}_clean_v2.parquet"

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
    significant word in the business name."""
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


def build_house_street_index(other: pl.DataFrame, cap: int = HOUSE_STREET_CAP) -> dict:
    """dict[(country, f'{house}_{street_token}')] -> capped list of entity_id."""
    idx: dict = collections.defaultdict(list)
    for country, house, tokens, eid in zip(
        other["country"].to_list(), other["house_no"].to_list(), other["addr_tokens"].to_list(), other["entity_id"].to_list()
    ):
        if not house or not tokens:
            continue
        for tok in tokens:
            bucket = idx[(country, f"{house}_{tok}")]
            if len(bucket) < cap:
                bucket.append(eid)
    return idx


def build_pin_house_index(other: pl.DataFrame, cap: int = PIN_HOUSE_CAP) -> dict:
    """dict[(country, f'{pin}_{house}')] -> capped list of entity_id."""
    idx: dict = collections.defaultdict(list)
    for country, pin, house, eid in zip(
        other["country"].to_list(), other["pin"].to_list(), other["house_no"].to_list(), other["entity_id"].to_list()
    ):
        if not pin or not house:
            continue
        bucket = idx[(country, f"{pin}_{house}")]
        if len(bucket) < cap:
            bucket.append(eid)
    return idx


def build_addr_token_index(other: pl.DataFrame, cap: int = ADDR_TOKEN_CAP) -> dict:
    """dict[(country, addr_token)] -> capped list of entity_id for distinctive tokens (>=5 chars)."""
    idx: dict = collections.defaultdict(list)
    for country, tokens, eid in zip(
        other["country"].to_list(), other["addr_tokens"].to_list(), other["entity_id"].to_list()
    ):
        if not tokens:
            continue
        for tok in tokens:
            if len(tok) >= 5:
                bucket = idx[(country, tok)]
                if len(bucket) < cap:
                    bucket.append(eid)
    return idx


def process_and_write(s1: pl.DataFrame, other: pl.DataFrame, out_path: Path, max_candidates: int) -> tuple:
    log("building name-key index ...")
    name_idx = build_index(other, "name_key", cap=BUCKET_CAP)
    log(f"name-key index: {len(name_idx):,} buckets")
    log("building pin index ...")
    pin_idx = build_index(other, "pin", cap=BUCKET_CAP)
    log(f"pin index: {len(pin_idx):,} buckets")
    log("building token index ...")
    token_idx = build_token_index(other, cap=TOKEN_BUCKET_CAP)
    log(f"token index: {len(token_idx):,} buckets")
    log("building house-street composite index ...")
    house_street_idx = build_house_street_index(other, cap=HOUSE_STREET_CAP)
    log(f"house-street index: {len(house_street_idx):,} buckets")
    log("building pin-house composite index ...")
    pin_house_idx = build_pin_house_index(other, cap=PIN_HOUSE_CAP)
    log(f"pin-house index: {len(pin_house_idx):,} buckets")
    log("building address-token index ...")
    addr_tok_idx = build_addr_token_index(other, cap=ADDR_TOKEN_CAP)
    log(f"address-token index: {len(addr_tok_idx):,} buckets")

    log("building compact candidate metadata lookup ...")
    c_eids = other["entity_id"].to_list()
    c_cores = other["name_core"].to_list()
    c_pins = other["pin"].to_list()
    c_houses = other["house_no"].to_list()
    c_stoks = [tuple(x) if x is not None else () for x in other["sig_tokens"].to_list()]
    c_atoks = [tuple(x) if x is not None else () for x in other["addr_tokens"].to_list()]
    cand_meta = dict(zip(c_eids, zip(c_cores, c_pins, c_houses, c_stoks, c_atoks)))
    del c_eids, c_cores, c_pins, c_houses, c_stoks, c_atoks
    del other
    import gc
    gc.collect()
    log("freed candidate pool DataFrame, metadata lookup ready")

    s1_ids = s1["entity_id"].to_list()
    s1_country = s1["country"].to_list()
    s1_nkey = s1["name_key"].to_list()
    s1_pin = s1["pin"].to_list()
    s1_core = s1["name_core"].to_list()
    s1_tokens = s1["sig_tokens"].to_list()
    s1_house = s1["house_no"].to_list()
    s1_addr_tokens = s1["addr_tokens"].to_list()
    n = len(s1_ids)

    total_raw_cands = 0
    total_final_cands = 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        t0 = time.time()
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            for i in range(start, end):
                eid, country, nkey, pin, core, tokens, house, atoks = (
                    s1_ids[i], s1_country[i], s1_nkey[i], s1_pin[i], s1_core[i],
                    s1_tokens[i], s1_house[i], s1_addr_tokens[i]
                )
                cand_ids = set()
                if nkey:
                    cand_ids.update(name_idx.get((country, nkey), ()))
                if pin:
                    cand_ids.update(pin_idx.get((country, pin), ()))
                if tokens:
                    for tok in tokens:
                        cand_ids.update(token_idx.get((country, tok), ()))
                if house and atoks:
                    for at in atoks:
                        cand_ids.update(house_street_idx.get((country, f"{house}_{at}"), ()))
                if pin and house:
                    cand_ids.update(pin_house_idx.get((country, f"{pin}_{house}"), ()))
                if atoks:
                    for at in atoks:
                        if len(at) >= 5:
                            cand_ids.update(addr_tok_idx.get((country, at), ()))

                raw_n = len(cand_ids)
                total_raw_cands += raw_n

                if not cand_ids:
                    f.write(f"{eid}\t\n")
                    continue

                # Shortlist generation (Req 4, 5, 6)
                s1_tok_set = set(tokens) if tokens else set()
                s1_at_set = set(atoks) if atoks else set()
                pin_pfx = pin[:3] if len(pin) >= 3 else ""

                if len(cand_ids) <= 100:
                    shortlist = list(cand_ids)
                else:
                    cand_scored = []
                    for cid in cand_ids:
                        meta = cand_meta.get(cid)
                        if not meta:
                            continue
                        c_core, c_pin, c_house, c_st, c_at = meta
                        score = 0
                        # 1. Exact normalized/core name match
                        if core and core == c_core:
                            score += 100
                        # 2. Exact PIN match
                        if pin and c_pin:
                            if pin == c_pin:
                                score += 25
                            elif pin_pfx and c_pin.startswith(pin_pfx):
                                score += 10
                        # 3. Exact house number match
                        if house and c_house and house == c_house:
                            score += 25
                        # 4. Address-token overlap
                        if s1_at_set and c_at:
                            for a in c_at:
                                if a in s1_at_set:
                                    score += 20
                                    break
                        # 5. Significant-name-token overlap
                        if s1_tok_set and c_st:
                            for t in c_st:
                                if t in s1_tok_set:
                                    score += 20
                                    break
                        # 6. Name_key match
                        if nkey and c_core.startswith(nkey):
                            score += 20
                        cand_scored.append((score, cid))

                    cand_scored.sort(key=lambda x: x[0], reverse=True)
                    shortlist = [cid for _, cid in cand_scored[:100]]

                # Run rapidfuzz ONLY on that shortlist (Req 7)
                final_scored = []
                for cid in shortlist:
                    meta = cand_meta.get(cid)
                    c_core, c_pin, c_house, c_st, c_at = meta if meta else ("", "", "", (), ())
                    sim = fuzz.token_sort_ratio(core, c_core)

                    bonus = 0
                    if house and c_house and house == c_house:
                        if pin and c_pin and pin == c_pin:
                            bonus = 25
                        else:
                            bonus = 15
                    elif pin and c_pin and pin == c_pin:
                        bonus = 10

                    if s1_at_set and c_at:
                        for a in c_at:
                            if a in s1_at_set:
                                bonus += 10
                                break

                    final_score = min(100, sim + bonus)
                    final_scored.append((final_score, cid))

                # Final top max_candidates (Req 8)
                final_scored.sort(key=lambda t: t[0], reverse=True)
                top = [cid for _, cid in final_scored[:max_candidates]]
                total_final_cands += len(top)
                f.write(f"{eid}\t{','.join(top)}\n")

            log(f"  processed {end:,}/{n:,} S1 rows ({time.time() - t0:.1f}s elapsed)")

    log(f"wrote {out_path}")
    log(f"Candidates generated before final top-K: {total_raw_cands:,} (avg {total_raw_cands/n:.1f}/S1)")
    log(f"Candidates generated after final top-K : {total_final_cands:,} (avg {total_final_cands/n:.1f}/S1)")
    return total_raw_cands, total_final_cands


def eval_recall(out_path: Path, gt_path: Path) -> None:
    cand = pl.read_csv(out_path, separator="\t", infer_schema_length=0)
    cand_s1_ids = cand["source1_entity_id"].to_list()
    cand_s1_set = set(cand_s1_ids)

    gt = pl.read_csv(gt_path, separator="\t", infer_schema_length=0)
    if len(cand_s1_set) < gt.height:
        gt = gt.filter(pl.col("source1_entity_id").is_in(cand_s1_set))

    gt_map: dict = {}
    n_sample_true = 0
    s1_with_true = 0
    for sid, matched in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        s = set(matched.split(",")) if matched and matched.strip() else set()
        gt_map[sid] = s
        if s:
            s1_with_true += 1
            n_sample_true += len(s)

    total_cands = 0
    max_cands = 0
    entities_with_cands = 0
    found = 0
    for sid, cands in zip(cand_s1_ids, cand["candidate_entity_ids"].to_list()):
        c = set(cands.split(",")) if cands and cands.strip() else set()
        c_len = len(c)
        if c_len > max_cands:
            max_cands = c_len
        if c:
            entities_with_cands += 1
            total_cands += c_len
        true = gt_map.get(sid, set())
        found += len(true & c)

    recall = found / n_sample_true if n_sample_true else 1.0
    avg_cands_per_sampled = total_cands / len(cand_s1_ids) if cand_s1_ids else 0.0

    log(f"Number of sampled S1 entities                     : {len(cand_s1_ids):,}")
    log(f"Sampled S1 entities with at least one true match   : {s1_with_true:,}")
    log(f"Total TRUE matches in the sample                   : {n_sample_true:,}")
    log(f"Captured TRUE matches                              : {found:,}")
    log(f"BLOCKING RECALL (upper bound on final F0.5)        : {recall:.4f}")
    log(f"Average candidates per sampled S1                  : {avg_cands_per_sampled:.1f}")
    log(f"Maximum candidates per sampled S1                  : {max_cands:,}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="row limit per source file, for smoke testing")
    ap.add_argument("--val-only", "--validate", dest="val_only", action="store_true", help="validation-only mode: sample S1 while indexing FULL S2 and S3")
    ap.add_argument("--val-samples", type=int, default=10000, help="number of S1 records to sample for validation mode")
    ap.add_argument("--out-file", type=str, default=None, help="custom output candidate filename")
    args = ap.parse_args()

    t0 = time.time()

    if args.val_only:
        log(f"=== VALIDATION-ONLY MODE: Sampling {args.val_samples:,} S1 rows against FULL S2 + S3 index ===")
        # Load S1 from cache/disk without limit, then slice the first val_samples records
        s1 = load_source("train", "source1", limit=None).head(args.val_samples)
        log(f"selected {s1.height:,} S1 records for validation")

        # Load FULL S2 and S3 (limit=None ensures all records are indexed)
        s2 = load_source("train", "source2", limit=None)
        s3 = load_source("train", "source3", limit=None)

        log("combining FULL S2 + S3 into one candidate pool ...")
        other = pl.concat([s2, s3])
        del s2, s3

        out_path = Path(args.out_file) if args.out_file else (OUT / f"candidate_pairs_val_{args.max_candidates}.tsv")
        process_and_write(s1, other, out_path, args.max_candidates)

        gt_path = DATA / "train" / "train_ground_truth.tsv"
        eval_recall(out_path, gt_path)
    else:
        s1 = load_source(args.split, "source1", args.limit)
        s2 = load_source(args.split, "source2", args.limit)
        s3 = load_source(args.split, "source3", args.limit)

        log("combining S2 + S3 into one candidate pool ...")
        other = pl.concat([s2, s3])
        del s2, s3

        out_path = Path(args.out_file) if args.out_file else (OUT / f"candidate_pairs_{args.split}.tsv")
        process_and_write(s1, other, out_path, args.max_candidates)

        if args.split == "train":
            gt_path = DATA / "train" / "train_ground_truth.tsv"
            eval_recall(out_path, gt_path)

    log(f"done in {time.time() - t0:.1f}s")



if __name__ == "__main__":
    main()
