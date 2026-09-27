import re
import pandas as pd
import streamlit as st
from pathlib import Path

BRAND_COLORS = {
    "TVS":          "#E8593C",
    "Ather":        "#1D9E75",
    "Ola Electric": "#7F77DD",
    "Bajaj":        "#BA7517",
    "Hero":         "#378ADD",
}

SOURCE_COLORS = {
    "91wheels":  "#E8593C",
    "bikewale":  "#1D9E75",
    "bikedekho": "#7F77DD",
}

SENTIMENT_COLORS = {
    "positive": "#1D9E75",
    "neutral":  "#BA7517",
    "negative": "#E8593C",
}
BASE_DIR    = Path(__file__).parent.parent

STOPWORDS_DIR = BASE_DIR/"resources"
ENGLISH_STOPWORDS = STOPWORDS_DIR/"english_stopwords.txt"
EV_STOPWORDS = STOPWORDS_DIR/"ev_stopwords.txt"
def load_english_stopwords() -> set[str]:
    
    with open(ENGLISH_STOPWORDS,encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        }

def load_ev_stopwords() -> set[str]:
    
    with open(EV_STOPWORDS,encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        }

def load_stopwords() -> set[str]:
    return load_english_stopwords() | load_ev_stopwords()

STOPWORDS = load_stopwords()

@st.cache_data(show_spinner="Loading reviews...")
# def load_data(path: str = "data/sentiment_reviews.csv") -> pd.DataFrame:
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # df = pd.read_csv(path)

    # Ensure types
    df["rating"]         = pd.to_numeric(df["rating"],         errors="coerce")
    df["rating_rounded"] = pd.to_numeric(df["rating_rounded"], errors="coerce")
    df["model_conf"]     = pd.to_numeric(df["model_conf"],     errors="coerce")
    df["word_count"]     = pd.to_numeric(df["word_count"],     errors="coerce")
    df["posted_date"]    = pd.to_datetime(df["posted_date"],   errors="coerce")
    df["is_short"]       = df["is_short"].astype(str).str.lower() == "true"

    # Derived columns
    df["month"]       = df["posted_date"].dt.to_period("M").astype(str)
    df["month_label"] = df["posted_date"].dt.strftime("%b %Y")
    df["source_label"] = df["source"].str.title()

    # Normalise labels to lowercase
    for col in ["final_label", "rating_sentiment", "model_label", "blend_tier"]:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    return df

def top_words(texts: pd.Series, n: int = 40) -> dict[str, int]:
    """Return top-n words from a series of review texts, excluding stopwords."""
    word_freq: dict[str, int] = {}
    for text in texts.dropna():
        for word in re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower()):
            if word not in STOPWORDS:
                word_freq[word] = word_freq.get(word, 0) + 1
    return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n])


def kpi_card(label: str, value: str, delta: str = "", color: str = "#1D9E75"):
    """Render a single KPI card using st.metric."""
    st.metric(label=label, value=value, delta=delta if delta else None)


def source_breakdown(df: pd.DataFrame) -> dict[str, int]:
    return df["source"].value_counts().to_dict()


def pct(n: int, total: int) -> str:
    if total == 0: return "0%"
    return f"{n/total*100:.1f}%"
