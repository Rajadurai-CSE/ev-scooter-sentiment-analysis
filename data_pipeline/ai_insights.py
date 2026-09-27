import re
import os
import json
import time
import log_manager.logger as logger
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from google import genai

warnings.filterwarnings("ignore")
log = logger.get_logger("ai_insights.py")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
INPUT_PATH  = DATA_DIR / "topic_reviews.csv"
OUTPUT_PATH = DATA_DIR / "ai_insights.json"
STOPWORDS_DIR = BASE_DIR/"resources"
ENGLISH_STOPWORDS = STOPWORDS_DIR/"english_stopwords.txt"
EV_STOPWORDS = STOPWORDS_DIR/"ev_stopwords.txt"


MODEL = "gemma-4-31b-it"   
MAX_TOKENS     =    3800
SAMPLES_PER_CLUSTER = 15    
API_DELAY      =   20 

SENTIMENT_COLORS = {
    "positive": "#1D9E75",
    "neutral":  "#BA7517",
    "negative": "#E24B4A",
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


def top_words(texts: pd.Series, n: int = 15) -> list[str]:
    EV_STOPWORDS_LI = load_stopwords()
    freq: dict[str, int] = {}
    for text in texts.dropna():
        for word in re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower()):
            if word not in EV_STOPWORDS_LI:
                freq[word] = freq.get(word, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]


def sentiment_dist(series: pd.Series) -> dict:
    counts = series.value_counts()
    total  = len(series)
    return {
        label: {
            "count": int(counts.get(label, 0)),
            "pct":   round(counts.get(label, 0) / max(total, 1) * 100, 1),
        }
        for label in ["positive", "neutral", "negative"]
    }


def clean_cluster_label(raw_label: str) -> str:
    """
    BERTopic default labels look like '0_rides_drive_speed_travel'.
    Convert to 'Rides / Drive / Speed / Travel' for readable display.
    """
    parts = raw_label.split("_")
    if parts[0].isdigit():
        parts = parts[1:]
    return " / ".join(w.title() for w in parts[:4])


def call_gemini(client: genai.Client,prompt: str) -> str:
    for attempt in range(3):
        try:
            resp = client.interactions.create(
                model=MODEL,
                input = prompt
                )
            return resp.output_text
        except Exception as e:
            log.warning("  API error (attempt %d): %s", attempt + 1, e)
            time.sleep((attempt+1)*5)
    return "Summary unavailable."



def cluster_summary_prompt(
    brand: str,
    model_name: str,
    cluster_label: str,
    n_reviews: int,
    sentiment: dict,
    top_words_list: list[str],
    samples: list[str],
) -> str:
    pos_pct = sentiment["positive"]["pct"]
    neg_pct = sentiment["negative"]["pct"]
    return f"""You are analyzing customer reviews for {brand} {model_name} electric scooters.

Topic cluster: "{cluster_label}"
Reviews in cluster: {n_reviews}
Sentiment: {pos_pct}% positive, {neg_pct}% negative
Top keywords: {', '.join(top_words_list[:10])}

Sample reviews from this cluster:
{chr(10).join(f'- {r[:300]}' for r in samples)}

Write a 3-sentence summary of what customers in this cluster are saying.
Focus on: (1) the main theme, (2) key praise or complaints, (3) one specific insight.
Be direct and specific. No generic statements."""


def overall_summary_prompt(
    brand: str,
    model_name: str,
    total: int,
    avg_rating: float,
    sentiment: dict,
    clusters: list[dict],
    pos_words: list[str],
    neg_words: list[str],
    pos_samples: list[str],
    neg_samples: list[str],
) -> str:
    cluster_overview = "\n".join(
        f"- {c['clean_label']} ({c['n_reviews']} reviews, "
        f"{c['sentiment_dist']['positive']['pct']}% positive): {c['summary']}"
        for c in clusters if c.get("summary")
    )
    return f"""You are an automotive market analyst producing a brand health report.

Brand: {brand} {model_name}
Total reviews analyzed: {total}
Average star rating: {avg_rating:.2f} / 5
Overall sentiment: {sentiment['positive']['pct']}% positive, \
{sentiment['negative']['pct']}% negative, \
{sentiment['neutral']['pct']}% neutral

Topic clusters identified:
{cluster_overview}

Top words in POSITIVE reviews: {', '.join(pos_words[:10])}
Top words in NEGATIVE reviews: {', '.join(neg_words[:10])}

Sample positive reviews:
{chr(10).join(f'- {r[:300]}' for r in pos_samples[:6])}

Sample negative reviews:
{chr(10).join(f'- {r[:300]}' for r in neg_samples[:6])}

Write a 200-word brand health report covering:
1. Overall brand health score (out of 10) and what drives it
2. Top 2 strengths customers consistently praise
3. Top 2 pain points requiring attention
4. One specific actionable recommendation for the product team
Be direct, data-driven, and specific to this brand."""

def check_prompt_tokens(client, prompt):
    result = client.models.count_tokens(
        model=MODEL,
        contents=prompt
    )
    tokens = result.total_tokens
    log.info(f"Prompt tokens: {tokens:,}")

    return tokens





def build_doc_viz_data(subset: pd.DataFrame) -> list[dict]:
    """
    Prepare data for the scatter plot / document visualiser in the dashboard.
    Uses cluster_prob as a proxy for distance from cluster centre
    (actual UMAP coordinates are not saved by default).

    Each point:  { review_id, x, y, cluster_label, final_label, review_snippet }
    x/y are synthesised from cluster_prob + jitter for visual spread.
    """
    rows = []
    rng  = np.random.default_rng(42)

    for _, row in subset.iterrows():
        cluster_id = row.get("cluster", -1)
        if cluster_id == -1:
            continue

        n_clusters = subset["cluster"].nunique()
        angle = (int(cluster_id) / max(n_clusters, 1)) * 2 * np.pi
        radius = 3 + (1 - float(row.get("cluster_prob", 0.5))) * 2

        x = float(np.cos(angle) * radius + rng.normal(0, 0.4))
        y = float(np.sin(angle) * radius + rng.normal(0, 0.4))

        text = str(row.get("review_text_en", ""))
        rows.append({
            "review_id":      str(row.get("review_id", "")),
            "x":              round(x, 3),
            "y":              round(y, 3),
            "cluster":        int(cluster_id),
            "cluster_label":  str(row.get("cluster_label", "unassigned")),
            "final_label":    str(row.get("final_label", "neutral")),
            "rating":         float(row.get("rating", 0)),
            "snippet":        text[:120] + ("…" if len(text) > 120 else ""),
            "user_name":      str(row.get("user_name", "")),
        })
    return rows



def run(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> dict:
    log.info("=" * 60)
    log.info("EV AI Insights — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    if not input_path.exists():
        log.error("Input not found: %s — run topic_modelling.py first", input_path)
        return {}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set. Export it before running.")
        return {}
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    df = pd.read_csv(input_path)
    # df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    log.info("Loaded %d rows", len(df))

    # Load existing insights if present — allows incremental updates
    if output_path.exists():
        with open(output_path,encoding="utf-8") as f:
            insights = json.load(f)
        log.info("Loaded existing insights (%d keys)", len(insights))
    else:
        insights = {}

    brand_model_map = {
        brand: df[df["brand"] == brand]["model"].unique().tolist()
        for brand in df["brand"].unique()
    }

    total_api_calls = 0

    for brand, models in brand_model_map.items():
        for model_name in models:
            key = f"{brand}_{model_name}"
            log.info("─" * 50)
            log.info("%s", key)

            mask   = (df["brand"] == brand) & (df["model"] == model_name) & (df["is_short"]==False)
            subset = df[mask].copy()

            if subset.empty:
                continue


            total      = len(subset)
            avg_rating = subset["rating"].mean()
            sent_dist  = sentiment_dist(subset["final_label"])


            cluster_data = []
            valid = subset[subset["cluster"] != -1]

            for cluster_id in sorted(valid["cluster"].unique()):
                cdf      = valid[valid["cluster"] == cluster_id]
                raw_label  = cdf["cluster_label"].iloc[0]
                clean_label = clean_cluster_label(raw_label)
                n_rev    = len(cdf)
                c_sent   = sentiment_dist(cdf["final_label"])
                words    = top_words(cdf["review_text_en"], n=15)
                samples  = (
                    cdf["review_text_en"].dropna()
                    .sample(min(SAMPLES_PER_CLUSTER, n_rev), random_state=42)
                    .tolist()
                )

                log.info("  Cluster %d: '%s' (%d reviews)", cluster_id, clean_label, n_rev)

                prompt  = cluster_summary_prompt(
                    brand, model_name, clean_label,
                    n_rev, c_sent, words, samples,
                )

                #Bug Fix
                log.info(f"Checking token size for {brand} {model_name} - Cluster {raw_label}")
                tokens = check_prompt_tokens(client,prompt)
                summary_unavailable = False
                while tokens > MAX_TOKENS:
                    log.info(f"Token exceed max limit for {brand} {model_name} - Cluster {raw_label}")
                    log.info("Truncating the prompt")
                    no_of_samples = max(0,len(samples) - 2)
                    if no_of_samples == 0:
                        log.info("Prompt with 0 review samples")
                        summary_unavailable = True
                        break
                    samples = samples[:no_of_samples]
                    prompt = cluster_summary_prompt(brand, model_name, clean_label,n_rev, c_sent, words, samples)
                    tokens = check_prompt_tokens(client,prompt)

                if summary_unavailable == False:
                    summary = call_gemini(client,prompt)
                    total_api_calls += 1
                    time.sleep(API_DELAY)
                else:
                    summary = "Summary Unavailable"

                cluster_data.append({
                    "cluster_id":     int(cluster_id),
                    "raw_label":      raw_label,
                    "clean_label":    clean_label,
                    "n_reviews":      n_rev,
                    "sentiment_dist": c_sent,
                    "top_words":      words,
                    "sample_reviews": samples,
                    "summary":        summary,
                })


            pos_words   = top_words(subset[subset["final_label"]=="positive"]["review_text_en"])
            neg_words   = top_words(subset[subset["final_label"]=="negative"]["review_text_en"])

            pos_samples = subset[subset["final_label"]=="positive"]["review_text_en"]
            neg_samples = subset[subset["final_label"]=="negative"]["review_text_en"]
            pos_samples = pos_samples.dropna().sample(min(len(pos_samples),10)).tolist()
            neg_samples = neg_samples.dropna().sample(min(len(neg_samples),10)).tolist()

            log.info(" Generating overall summary...")
            overall_prompt = overall_summary_prompt(
                brand, model_name, total, avg_rating,
                sent_dist, cluster_data,
                pos_words, neg_words,
                pos_samples, neg_samples,
            )

            log.info(f"Checking token size for {brand} {model_name} - Overall Summary")
            tokens = check_prompt_tokens(client,overall_prompt)

            summary_unavailable = False
            while tokens > MAX_TOKENS:
                    log.info(f"Token exceed max limit for {brand} {model_name} - Overall Summary")
                    log.info("Truncating the prompt")
                    no_of_pos_samples = max(0,len(pos_samples) - 2)
                    no_of_neg_samples = max(0,len(neg_samples) - 2)

                    if no_of_neg_samples == 0 and no_of_pos_samples == 0:
                        summary_unavailable = True
                        log.error("Prompt with 0 review samples")
                        break

                    pos_samples = pos_samples[:no_of_pos_samples]
                    neg_samples = neg_samples[:no_of_neg_samples]
                    overall_prompt = overall_summary_prompt(brand, model_name, total, avg_rating, sent_dist, cluster_data, pos_words, neg_words, pos_samples, neg_samples,)
                    tokens = check_prompt_tokens(client,overall_prompt)

            if summary_unavailable == False:
                overall_summary = call_gemini(client,overall_prompt)
                total_api_calls += 1
                time.sleep(API_DELAY)
            else:
                overall_summary = "Summary Unavailable"


            insights[key] = {
                "brand":           brand,
                "model":           model_name,
                "generated_at":    datetime.now().isoformat(),
                "total_reviews":   total,
                "avg_rating":      round(float(avg_rating), 2) if not pd.isna(avg_rating) else None,
                "sentiment_dist":  sent_dist,
                "overall_summary": overall_summary,
                "clusters":        cluster_data,
                # "doc_viz":         doc_viz,
            }

            log.info("  ✓ %d clusters | %d API calls so far", len(cluster_data), total_api_calls)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    log.info("=" * 60)
    log.info("Saved → %s", output_path)
    log.info("Total API calls: %d", total_api_calls)
    log.info("=" * 60)

    return insights


if __name__ == "__main__":
    run()