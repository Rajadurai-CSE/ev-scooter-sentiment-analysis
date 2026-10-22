# Indian EV Scooter Sentiment Analytics 

An end-to-end NLP and AI application that consolidates public electric-scooter reviews from multiple platforms and turns unstructured feedback into sentiment trends, recurring topics, and AI-generated summaries.

<p align="center">
  <a href="https://indian-ev-scooter-dashboard.duckdns.org/"><strong>🌐 Open the Live Dashboard</strong></a>
  &nbsp; | &nbsp;
  <a href="https://github.com/Rajadurai-CSE/ev-scooter-sentiment-analysis"><strong>💻 View the Source Code</strong></a>
</p>

## The Problem

Customer feedback is one of the most valuable sources of information for any product-driven company.
Understanding what customers appreciate, the issues they repeatedly encounter, and how their perception of a product changes over time can help businesses make better product decisions.
However, in India's electric scooter market, customer reviews are scattered across multiple platforms, making it difficult to consolidate feedback and identify recurring patterns.

## What if we could bring these scattered reviews together and turn them into actionable insights?

That was the idea behind this project.

## The Solution: Indian EV Scooter Sentiment Analytics

- **Multi-source review collection:** Collects public reviews from various review sites.
- **Interactive analytics dashboard:** Explore review volume, ratings, sentiment, topic clusters, and source-level differences. Filter by brand, model, and review source.
- **Sentiment analysis:** Classifies review text as positive, neutral, or negative, and retains model confidence and sentiment scores for further analysis.
- **Topic discovery with BERTopic:** Go beyond individual reviews by clustering similar feedback and identifying recurring customer concerns and discussion themes. 
- **AI-generated summaries:** Generate AI-based summaries of discovered topics and overall product sentiment, making large volumes of feedback easier to interpret.
- **Data quality and drift checks:** Checks incoming data and monitors rating-distribution changes between historical and newly collected reviews.
- **Cloud-based data delivery:** Publishes processed data to Amazon S3 for the dashboard to consume. AWS EventBridge, Systems Manager (SSM), and EC2 are used in the automated refresh/deployment workflow.

## Live Demo

**[Open the Indian EV Scooter Analytics Dashboard](https://indian-ev-scooter-dashboard.duckdns.org/)**

The dashboard currently covers these scooter models, subject to the latest data available in the pipeline:

- TVS iQube
- Ather Rizta
- Ola S1X
- Bajaj Chetak
- Hero Vida V2

## System Architecture

<img width="1672" height="940" alt="Ai_insights py Gemini 3 1B IT Brandmodel summaries (1)" src="https://github.com/user-attachments/assets/345b53da-2d80-4fb3-bb04-5a2ca4d86d5e" />


The main data artifacts consumed by the dashboard are published under the S3 `latest/` prefix:

- `latest/sentiment_reviews.csv`
- `latest/topic_reviews.csv`
- `latest/ai_insights.json`

In the automated workflow, a data-publication event/ready marker triggers an EventBridge rule and an SSM command to refresh the EC2-hosted dashboard service.

## How the Pipeline Works

### 1. Review collection — `data_pipeline/scraper.py`

Collects reviews from the configured pages for each supported scooter model. It standardizes fields such as brand, model, source, rating, review date, and scrape date. It also creates a stable review identifier, removes duplicates within a scrape batch, checks for reviews already present in the master dataset, and saves the newly collected batch for subsequent stages.

### 2. Data quality and drift — `data_pipeline/data_quality_ch.py`

Checks incoming rating values against expected bounds and compares rating distributions between historical and new data. The implementation uses **Population Stability Index (PSI)** for a source-level rating comparison and a **Kolmogorov–Smirnov (KS) two-sample test** for another source grouping. These checks are intended to flag data changes that may affect interpretation; they do not, by themselves, prove that customer sentiment or product quality has changed.

### 3. Text and metadata cleaning — `data_pipeline/cleaner.py`

Normalizes Unicode, decodes HTML entities, removes URLs and mentions, reduces repeated punctuation, and standardizes whitespace. It preserves the original review text in a separate field, converts rating and date fields into consistent formats, flags missing ratings, derives a rating-based sentiment reference, and marks very short reviews so they can be treated appropriately in topic analysis.

### 4. Sentiment scoring — `data_pipeline/sentiment.py`

Uses the Google Cloud Translation API to translate review text into English in batches, then applies Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment-latest` model. The output includes the original review, the English text, positive/neutral/negative scores, the model's predicted label, and its confidence.

The pipeline also applies a confidence-based blend: when model confidence is below the configured threshold, it can use the sentiment inferred from the star rating as a fallback. A validation report compares the final label with the rating-derived reference and records disagreements for inspection. A star rating is a useful reference signal, but it is not a perfect ground-truth sentiment label.

### 5. Semantic clustering and topic discovery — `data_pipeline/topic_modelling.py`

Generates text embeddings with `sentence-transformers/all-MiniLM-L12-v2` and uses **BERTopic Methods** to discover groups of related reviews. A domain-specific synonym map helps normalize common EV expressions (for example, variations of “mileage” into “range”), while stopword lists reduce generic words that would otherwise dominate topic labels. The output includes each review's cluster, readable cluster label, cluster probability, and visualization coordinates, along with a separate topic summary table.

### 6. AI-generated insights — `data_pipeline/ai_insights.py`

Combines topic labels, review samples, sentiment distributions, counts, and other calculated statistics into prompts for a Google GenAI-accessible Gemma model. The generated summaries are intended to make discovered themes easier to interpret; they are model-generated explanations and should be checked against the underlying reviews and metrics.

### 7. Dashboard — `app.py`

The Streamlit interface loads the latest sentiment, topic, and AI-insight artifacts from S3. Plotly charts support exploration of review ratings, sentiment, trends, source comparisons, and common topics. Sidebar filters allow users to compare brands, scooter models, and review platforms.

## Repository Guide

```text
.
├── .github/
│   └── workflows/             # GitHub Actions automation for data processing/deployment
├── data_pipeline/
│   ├── scraper.py              # Review collection, IDs, deduplication, batch/master updates
│   ├── cleaner.py              # Review text cleaning and metadata normalization
│   ├── data_quality_ch.py      # Feature checks and rating-distribution drift tests
│   ├── sentiment.py            # English translation, sentiment inference and validation
│   ├── topic_modelling.py      # Sentence embeddings and BERTopic topic discovery
│   ├── ai_insights.py          # LLM-based cluster summaries and product-level insights
│   └── log_manager/
│       └── logger.py           # Shared logging configuration and rotating log files
├── resources/
│   ├── english_stopwords.txt   # General English words excluded from topic keywords
│   └── ev_stopwords.txt        # EV-specific and low-information domain terms
├── utils/
│   ├── helpers.py              # Dashboard data preparation and reusable helper functions
│   ├── charts.py               # Plotly chart functions used by the dashboard
│   └── s3_loader.py            # Read CSV and JSON artifacts from Amazon S3
├── app.py                      # Streamlit dashboard entry point
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files and generated artifacts excluded from Git
└── README.md
```

### Generated data and model artifacts

The following files are created during processing and may live in a local `data/`, `models/`, `logs/`, or validation directory rather than being committed to Git:

| Artifact | Purpose |
|---|---|
| `master_reviews.csv` | Cumulative collection of known reviews used for deduplication and comparison with new data. |
| `batch.csv` | Newly collected, deduplicated reviews for the current pipeline run. |
| `clean_reviews.csv` | Cleaned text and normalized review metadata. |
| `sentiment_reviews.csv` | Review text, English translation, sentiment scores, final label, and confidence/blend metadata. |
| `topic_reviews.csv` | Review-level topic assignments, topic labels, probabilities, and visualization fields. |
| `topic_info.csv` | Topic-level summary information such as topic names, counts, and representative terms. |
| `ai_insights.json` | AI-generated summaries used by the dashboard. |
| `drift_stats.json` | Results from the data-drift checks. |
| `validation/blend_validation.csv` | Review-level disagreements between model sentiment and rating-derived sentiment. |
| `validation/validation_report.json` | Aggregate sentiment validation statistics. |
| `models/bertopic_*` | Serialized BERTopic model artifacts created for brand/model groups. |

## Multilingual Review Handling — In Progress 🌐

Reviews in the Indian market can contain English, Hindi, Hinglish, regional-language phrases, and code-switching within a single sentence. The current pipeline includes an **initial translation-based approach**, but multilingual sentiment quality is still an area of development.

Current approach:

1. The original review is retained in `review_text`.
2. `sentiment.py` sends review text to the Google Cloud Translation API with English as the target language, in batches of up to 10 reviews.
3. The translation is stored separately in `review_text_en` and used as input to the English-focused sentiment and topic-modelling pipeline.
4. If a translation batch fails, the pipeline logs the error and falls back to the original text for that batch so the processing run can continue.

**Current limitation:** translation does not make the downstream model truly multilingual. Hinglish in Latin script, regional slang, sarcasm, product-specific terms, and mixed-language phrases may be translated imperfectly. If translation fails, the original non-English review can reach an English-focused model, which may reduce reliability. Translation and sentiment outputs should therefore be treated as estimates, not definitive labels.

Planned areas for improvement include language-aware evaluation across Indian languages, measuring translation failure and sentiment-agreement rates by language, handling code-switched reviews more carefully, and benchmarking translation-based predictions against suitable multilingual models. The aim is to improve reliability without losing the context and meaning of the original review.

## Tech Stack

| Layer | Tools and technologies |
|---|---|
| Data collection | Python, Requests, Cloudscraper, Beautiful Soup, Pandas |
| Data quality and drift | Pandas, NumPy, SciPy, PSI, KS two-sample test |
| Text translation | Google Cloud Translation API |
| Sentiment analysis | Hugging Face Transformers, CardiffNLP Twitter-RoBERTa sentiment model |
| Embeddings and topic modelling | Sentence Transformers, BERTopic, UMAP, HDBSCAN, KeyBERTInspired |
| AI-generated summaries | Google GenAI SDK, Gemma model |
| Dashboard and charts | Streamlit, Plotly |
| Cloud and automation | Amazon S3, EventBridge, Systems Manager (SSM), EC2, GitHub Actions, AWS IAM/OIDC where configured |

## Run the Dashboard Locally

### Prerequisites

- Python version compatible with the packages pinned in `requirements.txt`.
- AWS credentials supplied through an appropriate IAM role, AWS profile, or environment configuration.

### Setup

```bash
git clone https://github.com/Rajadurai-CSE/ev-scooter-sentiment-analysis.git
cd ev-scooter-sentiment-analysis

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies and launch Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard reads the latest files from these S3 keys by default:

```text
latest/sentiment_reviews.csv
latest/topic_reviews.csv
latest/ai_insights.json
```

Set `BUCKET_NAME` to the S3 bucket used by your deployment and configure AWS authentication before launching the dashboard. Without access to the expected data files, the dashboard cannot display the complete live dataset.

## Run the Data Pipeline Locally

The full processing workflow requires additional external credentials and can incur translation/LLM API usage. Set the following environment variables before running the relevant stages:

- `GOOGLE_TRANSLATE_API_KEY` — Google Cloud Translation API key for translation in `sentiment.py`.
- `GEMINI_API_KEY` — key used by `ai_insights.py` to access the Google GenAI API.
- AWS credentials and `BUCKET_NAME` — required for S3 access or publishing, depending on the workflow being run.

With dependencies installed and the required credentials available, the main processing stages can be run in order:

```bash
python data_pipeline/scraper.py
python data_pipeline/cleaner.py
python data_pipeline/sentiment.py
python data_pipeline/topic_modelling.py
python data_pipeline/ai_insights.py
```

`data_quality_ch.py` is invoked by the scraping stage for checks on incoming data. Run the stages in order because each consumes artifacts produced by the previous step. Depending on the existing local data state and configuration, the processing scripts may also read or update cumulative files under `data/`.

This is an ongoing project focused on applying NLP, machine learning, large language models, data quality monitoring, and cloud deployment to a practical customer-feedback analytics problem.
