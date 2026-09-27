import re
import html
import unicodedata
import pandas as pd
from pathlib import Path
from datetime import datetime
from log_manager.logger import get_logger
log = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "batch.csv"
OUTPUT_PATH= DATA_DIR / "clean_reviews.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_WORDS = 5    # reviews shorter than this are flagged (not dropped)


def decode_html_entities(text: str) -> str:
    return html.unescape(text)


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



_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)



_MENTION_RE = re.compile(r"@\w+")

def remove_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)



_REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{1,}")

def remove_repeated_punctuation(text: str) -> str:
    return _REPEAT_PUNCT_RE.sub(r"\1", text)



_WHITESPACE_RE = re.compile(r"\s+")

def normalise_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()



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



def derive_rating_sentiment(rating) -> str | None:
    if pd.isna(rating):
        return None
    if rating >= 4.0:
        return "positive"
    elif rating==3.0:
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

def merge_with_existing(incoming_batch:pd.DataFrame):
    existing_df = pd.read_csv(OUTPUT_PATH)
    if len(existing_df) > 0:
        return pd.concat([existing_df,incoming_batch],ignore_index=True)
    return incoming_batch


def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("EV Cleaner — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # Load 
    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    # df = pd.read_csv(input_path)
    #Bug Fix 1 (Runner Cleaner only on the new reviews)
    # df = df[df['scraped_at'] == datetime.now().strftime("%d-%m-%Y")]
    log.info("Loaded %d rows from %s", len(df), input_path.name)

    # Step 1–6: Clean review_text 
    log.info("Cleaning review text...")
    df["review_text_raw"] = df["review_text"]          # preserve original
    df["review_text"]     = df["review_text"].apply(
        lambda x: clean_text(str(x)) if pd.notna(x) else ""
    )

    # Drop rows that became empty after cleaning
    before = len(df)
    df = df[df["review_text"].str.len() > 0]
    log.info("  Dropped %d rows that were empty after cleaning", before - len(df))

    # Step 7: Word count + is_short
    log.info("Flagging short reviews...")
    df["word_count"] = df["review_text"].apply(word_count)
    df["is_short"]   = df["word_count"] < MIN_WORDS
    log.info("  Short reviews (< %d words): %d", MIN_WORDS, df["is_short"].sum())

    # Step 8: is_rating_missing
    df["rating"]            = pd.to_numeric(df["rating"], errors="coerce")
    df['rating_rounded'] = df['rating'].round()
    df["is_rating_missing"] = df["rating"].apply(flag_missing_rating)
    log.info("  Missing ratings: %d", df["is_rating_missing"].sum())

    #  Step 9: rating_sentiment 
    log.info("Deriving rating_sentiment from star ratings...")
    df["rating_sentiment"] = df["rating_rounded"].apply(derive_rating_sentiment)
    dist = df["rating_sentiment"].value_counts(dropna=False).to_dict()
    log.info("  Distribution: %s", dist)

    #  Step 10: Parse posted_date
    log.info("Parsing posted_date...")
    df = parse_posted_date(df)
    bad_dates = df["posted_date"].isna().sum()
    if bad_dates:
        log.warning("  %d dates could not be parsed → NaT", bad_dates)

    # Final column order 
    # review_text_raw kept for debugging but placed last
    ordered = [
        "review_id", "brand", "model", "source",
        "user_name", "review_text",
        "rating","rating_rounded", "is_rating_missing", "rating_sentiment",
        "posted_date", "scraped_at",
        "word_count", "is_short",
        "review_text_raw",
    ]
    df = df[[c for c in ordered if c in df.columns]]

    final_clean_df = merge_with_existing(df)

    # Save 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Clean data saved → %s (%d rows)", output_path, len(final_clean_df))
    log.info("=" * 60)

    return final_clean_df


if __name__ == "__main__":
    run()