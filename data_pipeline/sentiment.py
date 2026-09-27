import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from transformers import pipeline as hf_pipeline
import asyncio
from googletrans import Translator
from log_manager.logger import get_logger
log = get_logger(__name__)
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
VAL_DIR    = DATA_DIR / "validation"
INPUT_PATH = DATA_DIR / "clean_reviews.csv"
OUTPUT_PATH= DATA_DIR / "sentiment_reviews.csv"

VAL_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32        
MAX_TOKENS = 512       

# Confidence thresholds
_CONF  = 0.60   


translator = Translator()


def rating_to_label(rating) -> str | None:
    r = float(rating)
    if r >= 4.0: return  "positive"
    if r == 3.0: return  "neutral"
    return              "negative"



def confidence_blend(
    model_label: str, #positive
    model_conf: float, # 0.7
    rating_sentiment:str, # positive
) -> tuple[str, str]:

    if model_conf >= _CONF:
        final_label = model_label
        tier  = "high"

    else:
        if rating_sentiment is not None:
            final_label = rating_sentiment
            tier  = "low_rating_fallback"
        else:
            final_label = model_label
            tier  = "low_model_fallback"

    return final_label, tier

async def translate_to_english(texts):
    tasks = [translator.translate(text, dest="en") for text in texts]
    results = await asyncio.gather(*tasks)
    return [r.text for r in results]


def load_model():

    log.info("Loading model: %s", MODEL_NAME)
    classifier = hf_pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        top_k=None,           # return all 3 class scores
        truncation=True,
        max_length=MAX_TOKENS,
        device=0,           
    )
    log.info("Model loaded.")
    return classifier


def run_inference(classifier, texts: list[str]) -> list[dict]:

    results = []
    total   = len(texts)

    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        # Replace empty strings — model errors on empty input
        batch = [t if t.strip() else "no review text" for t in batch]

        try:
            raw = classifier(batch)    # list of lists (top_k=None)
        except Exception as e:
            log.warning("Batch %d–%d failed: %s", start, start+len(batch), e)
            # Return neutral with zero confidence for failed batch
            raw = [[{"label":"neutral","score":0.0}]*3] * len(batch)

        for item in raw:
            # item is a list of dicts [{label, score}, {label, score}, {label, score}]
            best   = max(item, key=lambda x: x["score"])
            scores = {d["label"].lower(): round(d["score"], 4) for d in item}
            results.append({
                "model_label": best["label"].lower(),
                "model_conf":  round(best["score"], 4),
                "score_pos":   scores.get("positive", 0.0),
                "score_neu":   scores.get("neutral",  0.0),
                "score_neg":   scores.get("negative", 0.0),
            })

        log.info("  Inference: %d / %d rows done", min(start + BATCH_SIZE, total), total)

    return results


def validate(df: pd.DataFrame) -> dict:

    log.info("Running validation...")

    has_rating = df["rating_sentiment"].notna()
    val_df     = df[has_rating].copy()

    if val_df.empty:
        log.warning("No rows with rating_sentiment — skipping validation.")
        return {}

    # ── Agreement ─────────────────────────────────────────────────────────────
    val_df["agrees"] = val_df["final_label"] == val_df["rating_sentiment"]
    overall_agree    = val_df["agrees"].mean()

    # ── Per-tier agreement ────────────────────────────────────────────────────
    tier_agreement = (
        val_df.groupby("blend_tier")["agrees"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "agreement_rate", "count": "n_reviews"})
        .round(4)
        .to_dict(orient="index")
    )

    # ── Confusion matrix  (final_label vs rating_sentiment) ──────────────────
    # Rows = final_label, Cols = rating_sentiment (ground truth)
    labels = ["positive", "neutral", "negative"]
    conf_matrix = pd.crosstab(
        val_df["final_label"],
        val_df["rating_sentiment"],
        rownames=["predicted"],
        colnames=["actual"],
    ).reindex(index=labels, columns=labels, fill_value=0)

    disagreements = val_df[~val_df["agrees"]][[
        "brand", "model", "source",
        "review_text","review_text_en", "rating",
        "model_label", "model_conf", "blend_tier",
        "rating_sentiment", "final_label",
    ]].copy()



    val_path = VAL_DIR / "blend_validation.csv"
    disagreements.to_csv(val_path, index=False, encoding="utf-8-sig")
    log.info("  Disagreements saved → %s (%d rows)", val_path, len(disagreements))

    report = {
        "run_date":             datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_validated":      int(len(val_df)),
        "overall_agreement":    round(float(overall_agree), 4),
        "overall_disagreement": round(1 - float(overall_agree), 4),
        "per_tier":             tier_agreement,
        "confusion_matrix":     conf_matrix.to_dict(),
        "n_disagreements":      int(len(disagreements)),
        "blend_distribution":   df["blend_tier"].value_counts().to_dict(),
        "final_label_dist":     df["final_label"].value_counts().to_dict(),
        "conf_distribution": {
            "mean":   round(float(df["model_conf"].mean()), 4),
            "pct_high":   round(float((df["model_conf"] >= _CONF).mean()), 4),
            "pct_low":    round(float((df["model_conf"] < _CONF).mean()), 4),
        },
    }

    report_path = VAL_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("  Validation report → %s", report_path)

    log.info("=" * 55)
    log.info("VALIDATION SUMMARY")
    log.info("  Rows validated      : %d", len(val_df))
    log.info("  Overall agreement   : %.1f%%", overall_agree * 100)
    log.info("  Disagreements       : %d", len(disagreements))
    log.info("  Blend distribution  :")
    for tier, count in df["blend_tier"].value_counts().items():
        pct = count / len(df) * 100
        log.info("    %-25s %d  (%.1f%%)", tier, count, pct)
    log.info("  Confidence stats    :")
    log.info("    high (≥0.60)      : %.1f%%", report["conf_distribution"]["pct_high"]*100)
    log.info("    low  (<0.60)      : %.1f%%", report["conf_distribution"]["pct_low"]*100)
    log.info("  Confusion matrix (predicted vs actual star rating):")
    log.info("\n%s", conf_matrix.to_string())
    log.info("=" * 55)

    return report

def merge_with_existing(incoming_batch:pd.DataFrame):
    existing_df = pd.read_csv(OUTPUT_PATH)
    if len(existing_df) > 0:
        return pd.concat([existing_df,incoming_batch],ignore_index=True)
    return incoming_batch


async def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    log.info("=" * 55)
    log.info("EV Sentiment Pipeline — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 55)


    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    df = df[df['scraped_at'] == datetime.now().strftime("%d-%m-%Y")]
    log.info("Loaded %d reviews from %s", len(df), input_path.name)

    # Ensure rating column is numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    classifier = load_model()
    
    log.info("Translating to English...")
    translated = await translate_to_english(df['review_text'].fillna("").tolist())
    df['review_text_en'] = translated
    texts      = df["review_text_en"].tolist()
    preds      = run_inference(classifier, texts)

    # Attach model outputs
    df["model_label"] = [p["model_label"] for p in preds]
    df["model_conf"]  = [p["model_conf"]  for p in preds]
    df["score_pos"]   = [p["score_pos"]   for p in preds]
    df["score_neu"]   = [p["score_neu"]   for p in preds]
    df["score_neg"]   = [p["score_neg"]   for p in preds]


    log.info("Applying confidence blend...")
    blends = df.apply(
        lambda row: confidence_blend(
            row["model_label"],
            row["model_conf"],
            row["rating_sentiment"],
        ),
        axis=1,
    )
    df["final_label"] = [b[0] for b in blends]
    df["blend_tier"]  = [b[1] for b in blends]

    validate(df)

    col_order = [
        "review_id", "brand", "model", "source",
        "user_name", "review_text","review_text_en",
        "rating", "rating_rounded","rating_sentiment",
        "model_label", "model_conf",
        "score_pos", "score_neu", "score_neg",
        "final_label", "blend_tier",
        "posted_date", "scraped_at",
        "word_count", "is_short"
    ]
    df = df[[c for c in col_order if c in df.columns]]
    final_df = merge_with_existing(df)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Saved → %s (%d rows)", output_path, len(df))

    return final_df


if __name__ == "__main__":
    asyncio.run(run())