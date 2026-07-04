"""
EV Sentiment — Cleaning Pipeline
==================================
Input  : data/master_reviews.csv   (raw output from scraper.py)
Output : data/clean_reviews.csv    (ready for sentiment + BERTopic)

What scraper._clean_batch() already did (don't repeat):
  ✓ drop empty review_text / user_name
  ✓ collapse whitespace, strip trailing "....."
  ✓ clip rating 1–5
  ✓ MD5 review_id + deduplication

What this file adds:
  → HTML entity decode      → Unicode normalise
  → remove URLs / mentions  → remove repeated punctuation
  → word count + is_short   → is_rating_missing
  → rating_sentiment        → posted_date → datetime
"""

import re
import html
import unicodedata
import pandas as pd
from pathlib import Path
from datetime import datetime
from logger import log

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "master_reviews.csv"
OUTPUT_PATH= DATA_DIR / "clean_reviews.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_WORDS = 5    # reviews shorter than this are flagged (not dropped)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — HTML entity decoding
# ─────────────────────────────────────────────────────────────────────────────
# Why: Some review text arrives with raw HTML entities because the site renders
# them in the browser but doesn't pre-decode them server-side.
# Examples that appear in your data:
#   &amp;   → &       (very common in Bikedekho JSON)
#   &nbsp;  → space   (non-breaking space — looks like a space but isn't)
#   &#39;   → '       (apostrophe in "don't", "it's")
#   &lt;    → <
#   &gt;    → >
# html.unescape() handles all standard named and numeric HTML entities in one call.
# It is idempotent — safe to call on already-clean text.

def decode_html_entities(text: str) -> str:
    return html.unescape(text)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Unicode normalisation (NFC)
# ─────────────────────────────────────────────────────────────────────────────
# Why: The same visible character can be encoded in multiple ways in Unicode.
# "café" can be stored as:
#   NFC  → é  is a single code point U+00E9
#   NFD  → e + combining accent ◌́  (two code points)
# If one review has NFC and another has NFD, string equality and TF-IDF both
# treat them as different — this inflates vocabulary and breaks deduplication.
# NFC is the standard web encoding. normalise("NFC", text) collapses all forms.
#
# We also replace the most common "smart" typography that copy-paste introduces:
#   \u2019  '  right single quote  → plain apostrophe '
#   \u2018  '  left single quote   → plain apostrophe '
#   \u201c  "  left double quote   → plain double quote "
#   \u201d  "  right double quote  → plain double quote "
#   \u2013  –  en-dash             → hyphen -
#   \u2014  —  em-dash             → hyphen -
#   \u00a0     non-breaking space  → regular space
# These appear frequently in Bikewale reviews (copy-pasted from Word/WhatsApp).

def normalise_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u2019": "'",  "\u2018": "'",
        "\u201c": '"',  "\u201d": '"',
        "\u2013": "-",  "\u2014": "-",
        "\u00a0": " ",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Remove URLs
# ─────────────────────────────────────────────────────────────────────────────
# Why: URLs carry no sentiment signal and add noise to TF-IDF vocabulary.
# The pattern covers:
#   http://...   https://...   www....
# \S+ matches any non-whitespace run — URLs don't contain spaces.
# We replace with a single space (not empty string) to avoid accidental
# word-merging: "visit https://ola.com now" → "visit  now" not "visitnow".

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Remove @mentions
# ─────────────────────────────────────────────────────────────────────────────
# Why: Bikewale allows comments and some reviewers tag brand accounts (@OlaElectric).
# @mentions are usernames, not review content. The pattern:
#   @       literal at-sign
#   \w+     one or more word characters (letters, digits, underscore)
# Same space-replacement logic as URLs.

_MENTION_RE = re.compile(r"@\w+")

def remove_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Remove repeated punctuation
# ─────────────────────────────────────────────────────────────────────────────
# Why: Indian reviewers expressively write "very good scooter!!!!!!" or
# "bakwaas.......". These add nothing to sentiment that one ! or . doesn't.
# Keeping them intact would also cause tokenisers to produce rare tokens
# ("!!!!!!!" is a different token from "!!!") bloating the vocabulary.
#
# Pattern breakdown:
#   ([!?.,])   capture group — one of these punctuation characters
#   {2,}       two or more repetitions of the same character
#   \1         back-reference — must be the SAME character (not !? mixed)
# Replacement: \1 — just one of that character.
# Examples:
#   "!!!!!!"  → "!"
#   "....."   → "."    (scraper strips trailing ..... but mid-text ones remain)
#   "????"    → "?"
# Note: we intentionally don't collapse repeated letters ("veryyyy") —
# that's emotionally meaningful and xlm-roberta handles it.

_REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{1,}")

def remove_repeated_punctuation(text: str) -> str:
    return _REPEAT_PUNCT_RE.sub(r"\1", text)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Final whitespace normalisation
# ─────────────────────────────────────────────────────────────────────────────
# Why: Steps 3 and 4 replace matches with " " which can create multiple
# consecutive spaces. This collapses them back to one and strips ends.
# \s+ matches any whitespace: space, tab, newline, carriage return.
# We do this as a final pass after all substitutions are complete.

_WHITESPACE_RE = re.compile(r"\s+")

def normalise_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


# ═════════════════════════════════════════════════════════════════════════════
# MASTER TEXT CLEANER — applies all steps in the correct order
# ─────────────────────────────────────────────────────────────────────────────
# Order matters:
#   1. HTML decode first  — so &amp;amp; doesn't confuse later regexes
#   2. Unicode normalise  — before any string matching
#   3. Remove URLs        — before punctuation cleanup (URLs contain dots/slashes)
#   4. Remove mentions    — straightforward substitution
#   5. Repeated punct     — after URLs removed (URLs have dots that would collapse)
#   6. Whitespace         — final pass, cleans up gaps left by removals

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = decode_html_entities(text)
    text = normalise_unicode(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_repeated_punctuation(text)
    text = normalise_whitespace(text)
    return text


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Word count + is_short flag
# ─────────────────────────────────────────────────────────────────────────────
# Why: Very short reviews ("Good", "ok", "nice scooter") carry almost no
# signal for BERTopic — there aren't enough words to form a topic.
# We FLAG them (is_short=True) rather than drop them — they're still valid
# for overall sentiment distribution but excluded from topic modelling.
#
# \w+ matches sequences of word characters (letters, digits, underscore).
# It correctly ignores punctuation, so "good!!" counts as 1 word, not 2.
# Hindi/Devanagari characters are word characters in Python regex Unicode mode.

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — is_rating_missing flag
# ─────────────────────────────────────────────────────────────────────────────
# Why: Bikewale sometimes returns 0 filled stars (the user didn't rate).
# Bikedekho JSON occasionally has null. We flag these separately so downstream
# code can choose to exclude them from rating-based analysis without silently
# skewing averages. A missing rating is different from a 1-star rating.
# pd.isna() catches both NaN (float) and None (object dtype).

def flag_missing_rating(rating) -> bool:
    return pd.isna(rating)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — Derive rating_sentiment from scraped star rating
# ─────────────────────────────────────────────────────────────────────────────
# Why: This gives us a GROUND TRUTH sentiment label derived from the user's
# own star rating — completely independent of any ML model.
# We use it in two ways:
#   a) As a baseline to validate xlm-roberta's predictions
#      (if model says Positive but user gave 1 star → flag for inspection)
#   b) As a fallback label for rows where the model fails or is uncertain
#
# Thresholds (standard for 5-star scales):
#   4–5  → Positive   (clear satisfaction)
#   3    → Neutral    (mixed / average)
#   1–2  → Negative   (clear dissatisfaction)
# Returns None if rating is missing — don't impute sentiment from nothing.

def derive_rating_sentiment(rating) -> str | None:
    if pd.isna(rating):
        return None
    rating = float(rating)
    if rating >= 4.0:
        return "positive"
    elif rating == 3.0:
        return "neutral"
    else:
        return "negative"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — Parse posted_date → unified datetime
# ─────────────────────────────────────────────────────────────────────────────
# Why: Your scraper stores posted_date as a DD-MM-YYYY string across all
# three sources. For time-series analysis (sentiment trend over months,
# seasonality around EV launches) we need an actual datetime object.
# We parse the one format the scraper standardised to and store as
# YYYY-MM-DD (ISO 8601) — universally sortable as a string AND parseable
# by pandas, Tableau, Power BI, and Streamlit without extra config.
#
# errors="coerce" on pd.to_datetime means unparseable dates become NaT
# (Not a Time) instead of raising an exception — safe for production data.

def parse_posted_date(df: pd.DataFrame) -> pd.DataFrame:
    df["posted_date"] = pd.to_datetime(
        df["posted_date"], format="%d-%m-%Y", errors="coerce"
    )
    # Store as ISO string for CSV portability — datetime objects don't
    # survive a CSV round-trip cleanly across all tools.
    df["posted_date"] = df["posted_date"].dt.strftime("%Y-%m-%d")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — applies all steps to the full dataframe
# ═════════════════════════════════════════════════════════════════════════════

def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("EV Cleaner — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    log.info("Loaded %d rows from %s", len(df), input_path.name)

    # ── Step 1–6: Clean review_text ───────────────────────────────────────────
    log.info("Cleaning review text...")
    df["review_text_raw"] = df["review_text"]          # preserve original
    df["review_text"]     = df["review_text"].apply(
        lambda x: clean_text(str(x)) if pd.notna(x) else ""
    )

    # Drop rows that became empty after cleaning
    before = len(df)
    df = df[df["review_text"].str.len() > 0]
    log.info("  Dropped %d rows that were empty after cleaning", before - len(df))

    # ── Step 7: Word count + is_short ─────────────────────────────────────────
    log.info("Flagging short reviews...")
    df["word_count"] = df["review_text"].apply(word_count)
    df["is_short"]   = df["word_count"] < MIN_WORDS
    log.info("  Short reviews (< %d words): %d", MIN_WORDS, df["is_short"].sum())

    # ── Step 8: is_rating_missing ─────────────────────────────────────────────
    df["rating"]            = pd.to_numeric(df["rating"], errors="coerce")
    df["is_rating_missing"] = df["rating"].apply(flag_missing_rating)
    log.info("  Missing ratings: %d", df["is_rating_missing"].sum())

    # ── Step 9: rating_sentiment ──────────────────────────────────────────────
    log.info("Deriving rating_sentiment from star ratings...")
    df["rating_sentiment"] = df["rating"].apply(derive_rating_sentiment)
    dist = df["rating_sentiment"].value_counts(dropna=False).to_dict()
    log.info("  Distribution: %s", dist)

    # ── Step 10: Parse posted_date ────────────────────────────────────────────
    log.info("Parsing posted_date...")
    df = parse_posted_date(df)
    bad_dates = df["posted_date"].isna().sum()
    if bad_dates:
        log.warning("  %d dates could not be parsed → NaT", bad_dates)

    # ── Final column order ────────────────────────────────────────────────────
    # review_text_raw kept for debugging but placed last
    ordered = [
        "review_id", "brand", "model", "source",
        "user_name", "review_text",
        "rating", "is_rating_missing", "rating_sentiment",
        "posted_date", "scraped_at",
        "word_count", "is_short",
        "review_text_raw",
    ]
    df = df[[c for c in ordered if c in df.columns]]

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Clean data saved → %s (%d rows)", output_path, len(df))
    log.info("=" * 60)

    return df


if __name__ == "__main__":
    run()