"""
features.py — Feature engineering for Business Entity Resolution (Person 2).

Generates numeric pairwise features comparing a Source 1 entity with candidate
Source 2 / Source 3 entities across:
  - Business name (character, token, n-gram, length differences)
  - Address (character, token, n-gram, length differences)
  - Structured fields (country match, PIN match/mismatch/prefix, city match, house number match)
  - General metadata (token counts, missing indicators, combined name+address similarity)

Designed to run fast on large candidate sets and handle null/empty values safely.
"""

import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

sys.path.insert(0, str(Path(__file__).parent))
from normalize import clean_name, name_core, clean_address, extract_pin, extract_city  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
CACHE = ROOT / "code" / "business_entity_resolution" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

_HOUSE_NO_RE = re.compile(r"\b(\d+[-/]?\d*)\b")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_house_no(clean_addr: str) -> str:
    """Extracts leading or standalone house/street/building number."""
    if not clean_addr:
        return ""
    m = _HOUSE_NO_RE.search(clean_addr)
    return m.group(1) if m else ""


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Jaccard similarity on character n-grams."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    ngrams1 = {s1[i:i+n] for i in range(len(s1) - n + 1)}
    ngrams2 = {s2[i:i+n] for i in range(len(s2) - n + 1)}
    u = len(ngrams1 | ngrams2)
    return len(ngrams1 & ngrams2) / u if u > 0 else 0.0


def token_jaccard(s1: str, s2: str) -> float:
    """Jaccard similarity on whitespace-split tokens."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    toks1 = set(s1.split())
    toks2 = set(s2.split())
    u = len(toks1 | toks2)
    return len(toks1 & toks2) / u if u > 0 else 0.0


# Field tuple definition for cached record lookup:
# (clean_name, core_name, clean_addr, country, pin, city, house_no, name_len, addr_len, name_n_toks, addr_n_toks)
RecordTuple = Tuple[str, str, str, str, str, str, str, int, int, int, int]

FEATURE_NAMES = [
    # Business name features
    "name_exact",
    "name_core_exact",
    "name_levenshtein",
    "name_core_levenshtein",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_char_ngram_jaccard",
    "name_token_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_tokens_diff",
    # Address features
    "addr_exact",
    "addr_levenshtein",
    "addr_jaro_winkler",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_char_ngram_jaccard",
    "addr_token_jaccard",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_tokens_diff",
    # Structured features
    "country_match",
    "has_pin_both",
    "pin_match",
    "pin_mismatch",
    "pin_prefix_match",
    "has_city_both",
    "city_match",
    "city_token_overlap",
    "has_house_both",
    "house_match",
    "house_mismatch",
    # General & combined features
    "missing_name_s1",
    "missing_name_cand",
    "missing_addr_s1",
    "missing_addr_cand",
    "s1_name_tokens",
    "cand_name_tokens",
    "s1_addr_tokens",
    "cand_addr_tokens",
    "combined_token_sort_ratio",
    "combined_token_set_ratio",
]


def load_and_preprocess_records(
    split: str,
    needed_eids: Set[str] | None = None,
    limit: int | None = None,
) -> Dict[str, RecordTuple]:
    """Loads Source 1, Source 2, and Source 3 records for a split and computes
    normalized fields stored as a compact in-memory lookup dictionary:
    entity_id -> RecordTuple.
    If needed_eids is provided, only loads and processes those entities.
    """
    lookup: Dict[str, RecordTuple] = {}
    for source in ("source1", "source2", "source3"):
        path = DATA / split / f"{split}_{source}.tsv"
        log(f"loading records from {path.name} ...")
        df = pl.read_csv(
            path,
            separator="\t",
            infer_schema_length=0,
            quote_char=None,
            n_rows=limit,
        )
        if needed_eids is not None:
            df = df.filter(pl.col("entity_id").is_in(needed_eids))

        eids = df["entity_id"].fill_null("").to_list()
        names = df["business_name"].fill_null("").to_list()
        addrs = df["business_address"].fill_null("").to_list()
        countries = df["country"].fill_null("").to_list()

        for eid, raw_n, raw_a, raw_c in zip(eids, names, addrs, countries):
            cn = clean_name(raw_n)
            core = name_core(cn)
            ca = clean_address(raw_a)
            ctry = (raw_c or "").strip().lower()
            pin = extract_pin(ca)
            city = extract_city(ca)
            house = extract_house_no(ca)

            cn_len = len(cn)
            ca_len = len(ca)
            cn_toks = len(cn.split()) if cn else 0
            ca_toks = len(ca.split()) if ca else 0

            lookup[eid] = (
                cn, core, ca, ctry, pin, city, house,
                cn_len, ca_len, cn_toks, ca_toks
            )
        log(f"  {source}: indexed {len(df):,} records (total lookup size: {len(lookup):,})")

    return lookup


def compute_pair_features(
    s1_rec: RecordTuple,
    c_rec: RecordTuple,
) -> List[float]:
    """Computes all 45 numeric features for a single (Source 1, Candidate) pair."""
    (
        s1_cn, s1_core, s1_ca, s1_ctry, s1_pin, s1_city, s1_house,
        s1_cn_len, s1_ca_len, s1_cn_toks, s1_ca_toks
    ) = s1_rec

    (
        c_cn, c_core, c_ca, c_ctry, c_pin, c_city, c_house,
        c_cn_len, c_ca_len, c_cn_toks, c_ca_toks
    ) = c_rec

    # --- Business name features ---
    name_exact = 1.0 if (s1_cn and s1_cn == c_cn) else 0.0
    name_core_exact = 1.0 if (s1_core and s1_core == c_core) else 0.0
    name_lev = float(Levenshtein.normalized_similarity(s1_cn, c_cn))
    name_core_lev = float(Levenshtein.normalized_similarity(s1_core, c_core))
    name_jw = float(JaroWinkler.similarity(s1_cn, c_cn))
    name_ts = float(fuzz.token_sort_ratio(s1_cn, c_cn)) / 100.0
    name_tset = float(fuzz.token_set_ratio(s1_cn, c_cn)) / 100.0
    name_partial = float(fuzz.partial_ratio(s1_cn, c_cn)) / 100.0
    name_ngram = float(char_ngram_jaccard(s1_cn, c_cn, n=3))
    name_tok_jaccard = float(token_jaccard(s1_cn, c_cn))
    name_len_diff = float(abs(s1_cn_len - c_cn_len))
    name_max_len = max(s1_cn_len, c_cn_len, 1)
    name_len_ratio = float(min(s1_cn_len, c_cn_len) / name_max_len)
    name_toks_diff = float(abs(s1_cn_toks - c_cn_toks))

    # --- Address features ---
    addr_exact = 1.0 if (s1_ca and s1_ca == c_ca) else 0.0
    addr_lev = float(Levenshtein.normalized_similarity(s1_ca, c_ca))
    addr_jw = float(JaroWinkler.similarity(s1_ca, c_ca))
    addr_ts = float(fuzz.token_sort_ratio(s1_ca, c_ca)) / 100.0
    addr_tset = float(fuzz.token_set_ratio(s1_ca, c_ca)) / 100.0
    addr_partial = float(fuzz.partial_ratio(s1_ca, c_ca)) / 100.0
    addr_ngram = float(char_ngram_jaccard(s1_ca, c_ca, n=3))
    addr_tok_jaccard = float(token_jaccard(s1_ca, c_ca))
    addr_len_diff = float(abs(s1_ca_len - c_ca_len))
    addr_max_len = max(s1_ca_len, c_ca_len, 1)
    addr_len_ratio = float(min(s1_ca_len, c_ca_len) / addr_max_len)
    addr_toks_diff = float(abs(s1_ca_toks - c_ca_toks))

    # --- Structured features ---
    country_match = 1.0 if (s1_ctry and s1_ctry == c_ctry) else 0.0

    both_pin = bool(s1_pin and c_pin)
    has_pin_both = 1.0 if both_pin else 0.0
    pin_match = 1.0 if (both_pin and s1_pin == c_pin) else 0.0
    pin_mismatch = 1.0 if (both_pin and s1_pin != c_pin) else 0.0
    pin_prefix_match = (
        1.0 if (both_pin and len(s1_pin) >= 3 and len(c_pin) >= 3 and s1_pin[:3] == c_pin[:3])
        else 0.0
    )

    both_city = bool(s1_city and c_city)
    has_city_both = 1.0 if both_city else 0.0
    city_match = 1.0 if (both_city and s1_city == c_city) else 0.0
    city_token_overlap = float(token_jaccard(s1_city, c_city))

    both_house = bool(s1_house and c_house)
    has_house_both = 1.0 if both_house else 0.0
    house_match = 1.0 if (both_house and s1_house == c_house) else 0.0
    house_mismatch = 1.0 if (both_house and s1_house != c_house) else 0.0

    # --- General & combined features ---
    missing_name_s1 = 1.0 if not s1_cn else 0.0
    missing_name_cand = 1.0 if not c_cn else 0.0
    missing_addr_s1 = 1.0 if not s1_ca else 0.0
    missing_addr_cand = 1.0 if not c_ca else 0.0

    comb_s1 = f"{s1_cn} {s1_ca}".strip()
    comb_c = f"{c_cn} {c_ca}".strip()
    comb_ts = float(fuzz.token_sort_ratio(comb_s1, comb_c)) / 100.0
    comb_tset = float(fuzz.token_set_ratio(comb_s1, comb_c)) / 100.0

    return [
        name_exact,
        name_core_exact,
        name_lev,
        name_core_lev,
        name_jw,
        name_ts,
        name_tset,
        name_partial,
        name_ngram,
        name_tok_jaccard,
        name_len_diff,
        name_len_ratio,
        name_toks_diff,
        addr_exact,
        addr_lev,
        addr_jw,
        addr_ts,
        addr_tset,
        addr_partial,
        addr_ngram,
        addr_tok_jaccard,
        addr_len_diff,
        addr_len_ratio,
        addr_toks_diff,
        country_match,
        has_pin_both,
        pin_match,
        pin_mismatch,
        pin_prefix_match,
        has_city_both,
        city_match,
        city_token_overlap,
        has_house_both,
        house_match,
        house_mismatch,
        missing_name_s1,
        missing_name_cand,
        missing_addr_s1,
        missing_addr_cand,
        float(s1_cn_toks),
        float(c_cn_toks),
        float(s1_ca_toks),
        float(c_ca_toks),
        comb_ts,
        comb_tset,
    ]


EMPTY_RECORD: RecordTuple = ("", "", "", "", "", "", "", 0, 0, 0, 0)


def extract_features_for_pairs(
    pairs: List[Tuple[str, str]],
    lookup: Dict[str, RecordTuple],
    batch_size: int = 50_000,
) -> np.ndarray:
    """Computes a 2D numpy array of features of shape (len(pairs), num_features)
    for the list of (source1_entity_id, candidate_entity_id) pairs.
    """
    n_pairs = len(pairs)
    n_feats = len(FEATURE_NAMES)
    log(f"extracting features for {n_pairs:,} candidate pairs ({n_feats} features each) ...")

    out = np.zeros((n_pairs, n_feats), dtype=np.float32)
    t0 = time.time()

    for start_idx in range(0, n_pairs, batch_size):
        end_idx = min(start_idx + batch_size, n_pairs)
        for i in range(start_idx, end_idx):
            sid, cid = pairs[i]
            s1_rec = lookup.get(sid, EMPTY_RECORD)
            c_rec = lookup.get(cid, EMPTY_RECORD)
            out[i] = compute_pair_features(s1_rec, c_rec)

        elapsed = time.time() - t0
        log(f"  extracted {end_idx:,}/{n_pairs:,} pairs ({elapsed:.1f}s elapsed, {end_idx / max(elapsed, 0.001):.0f} pairs/sec)")

    return out
