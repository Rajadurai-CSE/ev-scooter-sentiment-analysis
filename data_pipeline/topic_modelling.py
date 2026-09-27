import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from log_manager.logger import get_logger
log = get_logger(__name__)

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP as UMAP2D
warnings.filterwarnings("ignore")


BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
INPUT_PATH  = DATA_DIR / "sentiment_reviews.csv"
OUTPUT_PATH = DATA_DIR / "topic_reviews.csv"
INFO_PATH   = DATA_DIR / "topic_info.csv"
STOPWORDS_DIR = BASE_DIR/"resources"
ENGLISH_STOPWORDS = STOPWORDS_DIR/"english_stopwords.txt"
EV_STOPWORDS = STOPWORDS_DIR/"ev_stopwords.txt"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
MIN_WORDS       = 8
MIN_DOCS        = 20


SYNONYMS = {
    r"\bmileage\b": "range",
    r"\bmillage\b": "range",
    r"\bkmpl\b": "range",
    r"\bkm\b": "kilometer",
    r"\bkms\b": "kilometer",
    r"\bkilometers\b": "kilometer",

    
    r"\bpickup\b": "acceleration",
    r"\bpick up\b": "acceleration",
    r"\btorque\b": "power",
    r"\btop speed\b": "speed",

    # Ride
    r"\brides\b": "ride",
    r"\briding\b": "ride",
    r"\bcommuting\b": "ride",
    r"\bcommute\b": "ride",
     r"\bdrive\b": "ride",

    # Service
    r"\bsvc\b": "service",
    r"\bservicing\b": "service",
    r"\bservice centre\b": "service_center",
    r"\bservice center\b": "service_center",

    # Build
    r"\bfit and finish\b": "build_quality",
    r"\bbuild quality\b": "build_quality",

    # Software
    r"\bota\b": "software_update",
    r"\bsoftware update\b": "software_update",
}

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



# ── Preprocessing ─────────────────────────────────────────────────────────────

def camel_split(text: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)


#Requires Change --> Vector operation is preferred
def clean_for_topic(texts: list[str]) -> list[str]:

    final_res = []
    for i in texts:
        if not isinstance(i, str) or not i.strip():
            final_res.append("")
        else:
            text = camel_split(i)
            text = text.lower()
            # Apply synonym replacements
            for pattern, replacement in SYNONYMS.items():
                text = re.sub(pattern, replacement, text)
            text = re.sub(r"[^a-z\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            final_res.append(text)
    return final_res


# def preprocess_docs(
#     raw_texts: list[str],
#     min_words: int = MIN_WORDS
# ) -> tuple[list[str], list[str], list[int]]:
#     """
#     BERTopic uses text in two separate streams:

#     Stream A — EMBEDDING  : raw text → SentenceTransformer → clusters
#     Stream B — LABELLING  : cleaned text → CountVectorizer → topic words

#     Returns both streams aligned, plus kept_idx to map back to df rows.
#     """
#     raw_kept   = []
#     clean_kept = []
#     kept_idx   = []

#     for i, text in enumerate(raw_texts):
#         cleaned = clean_for_topic(str(text))
#         words   = [w for w in cleaned.split() if w not in EV_STOPWORDS and len(w) > 2]
#         # if len(words) >= min_words:
#         raw_kept.append(str(text))
#         clean_kept.append(" ".join(words))
#         kept_idx.append(i)

#     return raw_kept, clean_kept, kept_idx


#  Model builder 

def build_topic_model(n_docs: int,embedder):
  

    EV_STOPWORDS_FINAL = load_stopwords()
    n_neighbors      = min(15, max(5,  n_docs // 10))
    min_cluster_size = min(15, max(5,  n_docs // 15))
    nr_topics        = min(10, max(4,  n_docs // 30))

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        stop_words=list(EV_STOPWORDS_FINAL),
        ngram_range=(1, 2),
        min_df=2,
    )

    return BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=KeyBERTInspired(),
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=False,
    )


# def merge_with_existing(incoming_batch:pd.DataFrame):
#     existing_df = pd.read_csv(OUTPUT_PATH)
#     if len(existing_df) > 0:
#         return pd.concat([existing_df,incoming_batch],ignore_index=True)
#     return incoming_batch

#  Main pipeline 

def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("EV Topic Modelling — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    if not input_path.exists():
        log.error("Input not found: %s — run sentiment.py first", input_path)
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    log.info("Loaded %d rows from %s", len(df), input_path.name)

    required = ["brand", "model", "review_text_en", "final_label"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.error("Missing columns: %s", missing)
        return pd.DataFrame()

    df["cluster"]       = -1
    df["cluster_label"] = "unassigned"
    df["cluster_prob"]  = 0.0

    log.info("Loading embedding model: %s", EMBEDDING_MODEL)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding model loaded.")

    brand_model_map = {
        brand: df[df["brand"] == brand]["model"].unique().tolist()
        for brand in df["brand"].unique()
    }

    all_topic_info = []

    for brand, models in brand_model_map.items():
        for model_name in models:
            log.info("─" * 50)
            log.info("%s — %s", brand, model_name)

            mask      = (df["brand"] == brand) & (df["model"] == model_name) & (df["is_short"]==False)
            subset    = df[mask].copy()
            raw_texts = subset["review_text_en"].fillna("").tolist()

            if len(raw_texts) < MIN_DOCS:
                log.warning("  %d docs < MIN_DOCS (%d) — skipping", len(raw_texts), MIN_DOCS)
                continue

            res_texts = clean_for_topic(raw_texts)

            #2. Feed raw text for embedding generation
            embeddings = embedder.encode(raw_texts, show_progress_bar=False)


            topic_model          = build_topic_model(len(res_texts),embedder)
            topics, probs        = topic_model.fit_transform(res_texts, embeddings=embeddings)

            try:
                umap_2d   = UMAP2D(n_components=2, min_dist=0.1,
                                    metric="cosine", random_state=42)
                coords_2d = umap_2d.fit_transform(embeddings)
                df.loc[subset.index, "umap_x"] = coords_2d[:, 0].round(4)
                df.loc[subset.index, "umap_y"] = coords_2d[:, 1].round(4)
                log.info("  2D UMAP coords saved for %d docs", len(coords_2d))
            except Exception as e:
                log.warning("  2D UMAP failed: %s — doc scatter will use fallback", e)

            df.loc[subset.index, "cluster"] = topics
            df.loc[subset.index, "cluster_prob"] = [
                float(np.array(p).max()) if hasattr(p, "__len__") else float(p)
                for p in probs
            ]

            t_info     = topic_model.get_topic_info()
            id_to_name = dict(zip(t_info["Topic"], t_info["Name"]))
            df.loc[subset.index, "cluster_label"] = (
                df.loc[subset.index, "cluster"]
                  .map(id_to_name)
                  .fillna("unassigned")
            )

            slug      = f"bertopic_{brand}_{model_name}".replace(" ", "_").lower()
            save_path = MODELS_DIR / slug
            topic_model.save(
                str(save_path),
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=False,
            )
            log.info("  Model saved → %s", save_path)

            valid = t_info[t_info["Topic"] != -1].copy()
            valid["brand"] = brand
            valid["model"] = model_name
            all_topic_info.append(valid)

            n_outliers  = (np.array(topics) == -1).sum()
            outlier_pct = n_outliers / len(topics) * 100
            log.info(
                "  ✓ %d topics | %d outliers (%.1f%%) | %s",
                valid["Topic"].nunique(), n_outliers, outlier_pct,
                valid["Name"].head(5).tolist(),
            )

    col_order = [
        "review_id","brand","model","source","user_name",
        "review_text","review_text_en",
        "rating","rating_sentiment",
        "model_label","model_conf","score_pos","score_neu","score_neg","umap_x","umap_y",
        "final_label","blend_tier",
        "cluster","cluster_label","cluster_prob",
        "posted_date","scraped_at","word_count","is_short",
    ]
    df_out = df[[c for c in col_order if c in df.columns]]
    # df_out = merge_with_existing(df_out)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("topic_reviews.csv → %s (%d rows)", output_path, len(df_out))

    if all_topic_info:
        topic_info_df = pd.concat(all_topic_info, ignore_index=True)
        topic_info_df = topic_info_df[["brand","model","Topic","Name","Count","Representation"]]
        topic_info_df.to_csv(INFO_PATH, index=False, encoding="utf-8-sig")
        log.info("topic_info.csv → %s (%d topics)", INFO_PATH, len(topic_info_df))
    else:
        log.warning("No topics produced — check MIN_DOCS threshold")

    assigned   = (df_out["cluster"] != -1).sum()
    unassigned = (df_out["cluster"] == -1).sum()
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Assigned : %d (%.1f%%)", assigned, assigned / len(df_out) * 100)
    log.info("  Outliers : %d (%.1f%%)", unassigned, unassigned / len(df_out) * 100)
    log.info("=" * 60)

    return df_out


if __name__ == "__main__":
    run()