"""
normalize.py — Part 1: text cleaning for business_name / business_address.

Works across US, India and France (test-only country) without hard-coding
country-specific branches into the control flow — the same cleaning
functions run for every row; only the abbreviation dictionaries differ
in scope (name vs address), not by country.
"""

import re
import unicodedata
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# Legal-suffix / abbreviation expansions applied as whole-word replacements.
# Order matters: longer/more specific patterns first.
_NAME_ABBREV = {
    r"\bpvt\b": "private",
    r"\bp\.v\.t\b": "private",
    r"\bltd\b": "limited",
    r"\bl\.t\.d\b": "limited",
    r"\bllp\b": "llp",
    r"\bllc\b": "llc",
    r"\binc\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bcorp\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bco\b": "co",
    r"\bcompany\b": "co",
    r"\b&\b": "and",
    r"\bpriv\b": "private",
}

_LEGAL_SUFFIXES = {
    "private", "limited", "llp", "llc", "inc", "corp", "co",
    "pvt", "ltd", "plc", "sarl", "sas", "gmbh",
}

_STRIP_LEADING_JUNK = re.compile(r"^[\-\*<>~\.\s]+")


def clean_name(raw: str) -> str:
    """Lowercase, transliterate, expand abbreviations, strip punctuation."""
    if raw is None:
        return ""
    s = str(raw)
    s = unidecode(s)  # transliterate Devanagari / accented French chars -> ASCII
    s = s.lower()
    s = _STRIP_LEADING_JUNK.sub("", s)  # drop junk like "--", "<<" prefixes
    s = re.sub(r"[^a-z0-9&\s]", " ", s)  # drop punctuation except &
    for pattern, repl in _NAME_ABBREV.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_core(clean: str) -> str:
    """Name with legal suffixes removed — used for the blocking key / core similarity."""
    tokens = [t for t in clean.split() if t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def name_block_key(clean_core: str, k: int = 4) -> str:
    """First k alnum chars of the core name — cheap high-recall blocking key."""
    letters = re.sub(r"[^a-z0-9]", "", clean_core)
    return letters[:k] if letters else ""


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

_ADDR_ABBREV = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bstr\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bapt\b": "apartment",
    r"\bunit\b": "unit",
    r"\bno\b": "number",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
}

# PIN/ZIP patterns: Indian 6-digit, US 5(-4) digit zip, French 5-digit.
_PIN_RE = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b")


def clean_address(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = unidecode(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s,]", " ", s)
    for pattern, repl in _ADDR_ABBREV.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_pin(raw_or_clean: str) -> str:
    if not raw_or_clean:
        return ""
    m = _PIN_RE.search(raw_or_clean)
    return m.group(1) if m else ""


def extract_city(clean_addr: str) -> str:
    """Heuristic: address components are comma-separated; the city is usually
    the token right before the state/country tail, or the longest alpha token
    that isn't a house-number / PIN. Falls back to '' when unclear."""
    if not clean_addr:
        return ""
    parts = [p.strip() for p in clean_addr.split(",") if p.strip()]
    # prefer a part with only letters/spaces (no digits) and length >= 3,
    # scanning from the end since city/state/country tend to trail.
    for p in reversed(parts):
        letters_only = re.sub(r"[^a-z\s]", "", p).strip()
        if letters_only and len(letters_only) >= 3 and not re.search(r"\d", p):
            return letters_only
    return ""
