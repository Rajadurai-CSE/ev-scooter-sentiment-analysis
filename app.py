import json
import warnings
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

from utils.helpers import (preprocess_data, source_breakdown, pct)
from utils.charts import (source_kpi_bar, rating_bar, rating_histogram, rating_over_time, sentiment_pie, platform_bias_chart, word_treemap, source_comparison_bars, source_rating_box, source_volume_bar, brand_sentiment_heatmap, topic_bar, topic_sentiment_bar)
from utils.s3_loader import load_s3_csv,load_s3_json

# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="EV Scooter Review Analytics",page_icon="⚡",layout="wide",initial_sidebar_state="expanded",)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD THEME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>

    /* ================================================================
       PAGE
       ================================================================ */

    .stApp {
        background: #eef4fb;
    }

    .main .block-container {
        max-width: 1600px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        padding-left: 1.3rem;
        padding-right: 1.3rem;
    }

    body {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    /* ================================================================
       SIDEBAR
       ================================================================ */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dfe7f0;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] h2 {
        color: #18324b;
        font-size: 1.15rem;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] hr {
        border: 0;
        border-top: 1px solid #edf0f4;
        margin: 0.8rem 0 1.1rem;
    }

    section[data-testid="stSidebar"] label {
        color: #536273 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #f8fafc;
        border: 1px solid #dce5ef;
        border-radius: 8px;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 7px 2px 3px 2px;
    }

    .sidebar-icon {
        width: 39px;
        height: 39px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #0877c9,
            #6239c7
        );
        color: white;
        font-size: 21px;
        font-weight: 700;
        box-shadow: 0 5px 14px rgba(48, 91, 190, 0.20);
    }

    .sidebar-title {
        color: #172b3f;
        font-size: 17px;
        font-weight: 750;
        line-height: 1.1;
    }

    .sidebar-subtitle {
        color: #7c8b9b;
        font-size: 10px;
        margin-top: 2px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-context {
        background: #f5f8fc;
        border: 1px solid #e5ebf2;
        border-radius: 9px;
        padding: 10px 12px;
        margin-top: 14px;
    }

    .sidebar-context-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        padding: 4px 0;
        font-size: 11px;
    }

    .sidebar-context-key {
        color: #8a96a3;
    }

    .sidebar-context-value {
        color: #304357;
        font-weight: 600;
        text-align: right;
    }

    /* ================================================================
       HERO
       ================================================================ */

    .hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(
                105deg,
                #e9f4ff 0%,
                #dceeff 48%,
                #e9e3ff 100%
            );
        border: 1px solid #d5e3f1;
        border-radius: 18px;
        padding: 21px 25px 22px 25px;
        margin-bottom: 15px;
        box-shadow:
            0 5px 20px rgba(43, 78, 117, 0.07);
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -100px;
        width: 310px;
        height: 310px;
        border-radius: 50%;
        background:
            radial-gradient(
                circle,
                rgba(104, 76, 202, 0.16) 0%,
                rgba(104, 76, 202, 0) 70%
            );
    }

    .hero-inner {
        position: relative;
        z-index: 2;
    }

    .hero-kicker {
        color: #1775ba;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .hero-title {
        color: #182b3e;
        font-size: 29px;
        font-weight: 750;
        letter-spacing: -0.035em;
        line-height: 1.1;
        margin: 0;
    }

    .hero-subtitle {
        color: #65778a;
        font-size: 12px;
        margin-top: 7px;
    }

    .hero-pill {
        display: inline-block;
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(255,255,255,0.9);
        color: #4c5e70;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 10px;
        margin-top: 10px;
        margin-right: 5px;
    }

    /* ================================================================
       SECTION HEADERS
       ================================================================ */

    .section-header {
        color: #26394d;
        font-size: 16px;
        font-weight: 750;
        margin: 18px 2px 9px 2px;
        letter-spacing: -0.015em;
    }

    .section-header span {
        color: #8b98a7;
        font-size: 11px;
        font-weight: 500;
        margin-left: 7px;
    }

    /* ================================================================
       KPI CARDS
       ================================================================ */

    .kpi-card {
        min-height: 112px;
        border-radius: 15px;
        padding: 15px 17px;
        position: relative;
        overflow: hidden;
        color: white;
        box-shadow:
            0 7px 18px rgba(52, 78, 114, 0.13);
    }

    .kpi-card::after {
        content: "";
        position: absolute;
        right: -35px;
        bottom: -55px;
        width: 135px;
        height: 135px;
        border-radius: 50%;
        background: rgba(255,255,255,0.11);
    }

    .kpi-label {
        position: relative;
        z-index: 2;
        font-size: 11px;
        font-weight: 500;
        opacity: 0.88;
        margin-bottom: 5px;
    }

    .kpi-value {
        position: relative;
        z-index: 2;
        font-size: 25px;
        font-weight: 750;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }

    .kpi-foot {
        position: relative;
        z-index: 2;
        font-size: 10px;
        opacity: 0.78;
        margin-top: 7px;
    }

    .kpi-purple {
        background:
            linear-gradient(
                135deg,
                #5d3fc4,
                #6556dc
            );
    }

    .kpi-blue {
        background:
            linear-gradient(
                135deg,
                #1557a6,
                #137fc6
            );
    }

    .kpi-cyan {
        background:
            linear-gradient(
                135deg,
                #087fb8,
                #16a0c7
            );
    }

    .kpi-green {
        background:
            linear-gradient(
                135deg,
                #0bb68d,
                #12c99d
            );
    }

    .kpi-pink {
        background:
            linear-gradient(
                135deg,
                #dc4d7b,
                #e76687
            );
    }

    .kpi-violet {
        background:
            linear-gradient(
                135deg,
                #7142ae,
                #914ebc
            );
    }

    /* ================================================================
       CARD CONTAINERS
       ================================================================ */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #d9e3ed;
        border-radius: 14px;
        box-shadow:
            0 3px 12px rgba(55, 79, 106, 0.045);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 14px;
    }

    /* ================================================================
       PLOTLY
       ================================================================ */

    .js-plotly-plot {
        border-radius: 10px;
    }

    /* ================================================================
       INSIGHT CARDS
       ================================================================ */

    .insight-card {
        background: #ffffff;
        border: 1px solid #dce5ee;
        border-radius: 13px;
        padding: 17px 19px;
        color: #435467;
        font-size: 13px;
        line-height: 1.75;
        box-shadow:
            0 3px 12px rgba(55, 79, 106, 0.045);
    }

    .cluster-summary {
        background:
            linear-gradient(
                135deg,
                #f7fbff,
                #f7f5ff
            );
        border: 1px solid #e0e6f0;
        border-left: 4px solid #6570c9;
        border-radius: 9px;
        padding: 12px 15px;
        margin: 8px 0 14px;
        font-size: 13px;
        line-height: 1.7;
        color: #465668;
    }

    .bias-notice {
        background: #fffaf0;
        border: 1px solid #f2dfb2;
        border-left: 4px solid #e1ae38;
        border-radius: 9px;
        padding: 10px 13px;
        font-size: 11px;
        line-height: 1.55;
        color: #6e624d;
        margin-top: 7px;
    }

    /* ================================================================
       TABLE
       ================================================================ */

    [data-testid="stDataFrame"] {
        border: 1px solid #dbe4ed;
        border-radius: 10px;
        overflow: hidden;
        background: #fff;
    }

    /* ================================================================
       TABS
       ================================================================ */

    button[data-baseweb="tab"] {
        font-size: 12px;
        font-weight: 650;
    }

    /* ================================================================
       SELECT BOX
       ================================================================ */

       div[data-baseweb="select"] > div {
    background: linear-gradient(
        135deg,
        #d9edff,
        #c6e3ff
    ) !important;

    border: 1px solid #82baf0 !important;
    border-radius: 9px !important;

    box-shadow: 0 2px 5px rgba(60, 130, 200, 0.10) !important;
}

div[data-baseweb="select"] span {
    color: #075a9c !important;
    font-weight: 650 !important;
}

div[data-baseweb="select"] svg {
    fill: #075a9c !important;
}

div[data-baseweb="select"] > div:hover {
    background: linear-gradient(
        135deg,
        #cde7ff,
        #b9dcfa
    ) !important;

    border-color: #5fa5e8 !important;
}

    /* ================================================================
       SMALL TEXT
       ================================================================ */

    .subtle {
        color: #788797;
        font-size: 11px;
        line-height: 1.5;
    }

    /* ================================================================
       FOOTER
       ================================================================ */

    .footer {
        margin-top: 22px;
        border-radius: 11px;
        padding: 10px 16px;
        text-align: center;
        color: #ffffff;
        font-size: 11px;
        background:
            linear-gradient(
                90deg,
                #2f176d,
                #6930a7
            );
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(label, value, foot="", theme="purple"):
    """
    Rich KPI card inspired by the supplied dashboard reference.
    """

    return f"""
    <div class="kpi-card kpi-{theme}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-foot">{foot}</div>
    </div>
    """

def section(title, subtitle=""):
    if subtitle:
        html = f'<div class="section-header">' f'{title}' f'<span>{subtitle}</span>' f'</div>'
    else:
        html = f'<div class="section-header">' f'{title}' f'</div>'

    st.markdown(html,unsafe_allow_html=True)

def card_start():
    return st.container(border=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading reviews...")
def load_sentiment():
    #return load_s3_csv("data/sentiment_reviews.csv")
    return preprocess_data(load_s3_csv("latest/sentiment_reviews.csv"))

@st.cache_data(show_spinner="Loading topic data...")
def load_topics():

    try:
        topic_df = load_s3_csv("latest/topic_reviews.csv")
        #topic_df = pd.read_csv("data/topic_reviews.csv")
        numeric_columns = ["umap_x","umap_y","cluster_prob","rating","cluster",]
        for col in numeric_columns:
            if col in topic_df.columns:
                topic_df[col] = pd.to_numeric(topic_df[col],errors="coerce")
        return topic_df

    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_insights():

    try:
        return load_s3_json("latest/ai_insights.json")
        # with open("data/ai_insights.json",encoding="utf-8") as f:
        # with open(ai_insights,encoding="utf-8") as f:
        #     return json.load(f)

    except FileNotFoundError:
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

df_full = load_sentiment()
print("Dtypes",df_full.dtypes)
topic_full = load_topics()
insights = load_insights()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
           <!-- <div class="sidebar-icon">⚡</div> --!>
            <div>
                <div class="sidebar-title">EV Analytics</div>
                <div class="sidebar-subtitle">Review Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    brands = (["All Brands"]+ sorted(df_full["brand"].dropna().unique().tolist()))
    sel_brand = st.selectbox("Brand",brands)
    if sel_brand != "All Brands":
        pool = df_full[df_full["brand"] == sel_brand]["model"].dropna().unique().tolist()

    else:
        pool = df_full["model"].dropna().unique().tolist()

    sel_model = st.selectbox("Model",["All Models"] + sorted(pool))
    sources = (["Overall"]+ sorted(df_full["source"].dropna().unique().tolist()))
    sel_source = st.selectbox("Source",sources)

    st.markdown("---")
    st.html(
        f"""
        <div class="sidebar-context">

            <div class="sidebar-context-row">
                <span class="sidebar-context-key">
                    Brand
                </span>
                <span class="sidebar-context-value">
                    {sel_brand}
                </span>
            </div>

            <div class="sidebar-context-row">
                <span class="sidebar-context-key">
                    Model
                </span>
                <span class="sidebar-context-value">
                    {sel_model}
                </span>
            </div>

            <div class="sidebar-context-row">
                <span class="sidebar-context-key">
                    Source
                </span>
                <span class="sidebar-context-value">
                    {sel_source}
                </span>
            </div>

        </div>
        """
    )

# ══════════════════════════════════════════════════════════════════════════════
# FILTERS
# ══════════════════════════════════════════════════════════════════════════════

df = df_full.copy()

if sel_brand != "All Brands":
    df = df[df["brand"] == sel_brand]

if sel_model != "All Models":
    df = df[df["model"] == sel_model]

if sel_source != "Overall":
    df = df[df["source"] == sel_source]

tdf = topic_full.copy()

if not tdf.empty:
    if sel_brand != "All Brands":
        tdf = tdf[tdf["brand"] == sel_brand]

    if sel_model != "All Models":
        tdf = tdf[tdf["model"] == sel_model]

    if sel_source != "Overall":
        tdf = tdf[tdf["source"] == sel_source]

is_single_source = sel_source != "Overall"
is_single_brand = sel_brand != "All Brands"
is_single_model = sel_model != "All Models"

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════

context_text = f"{len(df):,} reviews"

if sel_brand != "All Brands":
    context_text += (f" · {sel_brand}")

if sel_model != "All Models":
    context_text += (f" · {sel_model}")

if sel_source != "Overall":
    context_text += (f" · {sel_source}")

st.html(
    f"""
    <div class="hero">

        <div class="hero-inner">

            <div class="hero-kicker">
                EV REVIEW INTELLIGENCE
            </div>

            <div class="hero-title">
                EV Scooter Review Analytics
            </div>

            <div class="hero-subtitle">
                Understand what riders like, dislike,
                and talk about across brands and platforms.
            </div>

            <span class="hero-pill">
                 {context_text}
            </span>

            <span class="hero-pill">
                NLP + Topic Modeling
            </span>

            <span class="hero-pill">
                Sentiment Analysis
            </span>

        </div>

    </div>
    """
)

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════

total = len(df)

n_pos = (df["final_label"] == "positive").sum()
n_neg = (df["final_label"] == "negative").sum()
n_neu = (df["final_label"] == "neutral").sum()
avg_rat = df["rating"].mean()

section("Overview","current selection")
k1, k2, k3, k4, k5 = st.columns(5,gap="small")

with k1:
    st.markdown(kpi_card("Total reviews",f"{total:,}","Reviews in current selection","purple"),unsafe_allow_html=True)

with k2:

    st.markdown(
        kpi_card(
            "Average rating",
            (
                f"{avg_rat:.2f} ★"
                if not pd.isna(avg_rat)
                else "N/A"
            ),
            "Out of 5 stars",
            "blue"
        ),
        unsafe_allow_html=True
    )

with k3:
    st.markdown(kpi_card("Positive",f"{pct(n_pos, total)}",f"{n_pos:,} reviews","green"),unsafe_allow_html=True)

with k4:
    st.markdown(kpi_card("Negative",f"{pct(n_neg, total)}",f"{n_neg:,} reviews","pink"),unsafe_allow_html=True)

with k5:
    st.markdown(kpi_card("Neutral",f"{pct(n_neu, total)}",f"{n_neu:,} reviews","violet"),unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE STRIP
# ══════════════════════════════════════════════════════════════════════════════

section("Review volume","by source")

with card_start():
    st.plotly_chart(source_kpi_bar(source_breakdown(df)),width="stretch",config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# RATINGS
# ══════════════════════════════════════════════════════════════════════════════

section("Ratings & sentiment","how riders are responding")

r1, r2 = st.columns(2,gap="small")

with r1:
    with card_start():
        st.plotly_chart(rating_bar(df),width="stretch",config={"displayModeBar": False})

with r2:
    with card_start():
        st.plotly_chart(rating_histogram(df),width="stretch",config={"displayModeBar": False})

with card_start():
    st.plotly_chart(rating_over_time(df),width="stretch",config={"displayModeBar": False})

s1, s2 = st.columns([1, 2],gap="small")

with s1:

    with card_start():
        st.markdown("**Sentiment distribution**")
        st.plotly_chart(sentiment_pie(df),width="stretch",config={"displayModeBar": False})

with s2:

    with card_start():
        st.plotly_chart(platform_bias_chart(df),width="stretch",config={"displayModeBar": False})
        st.markdown(
            """
            <div class="bias-notice">
            <b>Platform bias</b><br>
            91Wheels and Bikedekho tend to have
            higher ratings, while Bikewale contains
            more complaint-oriented reviews.
            Source weighting is applied in the
            sentiment pipeline.
            </div>
            """,
            unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# WORDS
# ══════════════════════════════════════════════════════════════════════════════

section("What riders talk about","frequent words")

with card_start():
    t1, t2 = st.tabs([" Positive reviews"," Negative reviews"])

    with t1:
        st.plotly_chart(word_treemap(df,"positive"),width="stretch",config={"displayModeBar": False})

    with t2:
        st.plotly_chart(word_treemap(df,"negative"),width="stretch",config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if not is_single_source:

    section("Source comparison","where the reviews come from")
    sc1, sc2 = st.columns(2,gap="small")

    with sc1:
        with card_start():
            st.plotly_chart(source_volume_bar(df),width="stretch",config={"displayModeBar": False})

        with card_start():
            st.plotly_chart(source_rating_box(df),width="stretch",config={"displayModeBar": False})

    with sc2:
        with card_start():
            st.plotly_chart(source_comparison_bars(df),width="stretch",config={"displayModeBar": False})

        with card_start():
            st.plotly_chart(brand_sentiment_heatmap(df),width="stretch",config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# TOPIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if is_single_brand and is_single_model:
    section("Topic analysis","what themes emerge from reviews")

    if (not tdf.empty and "cluster_label" in tdf.columns):
        tp1, tp2 = st.columns(2,gap="small")
        with tp1:
            with card_start():
                st.plotly_chart(topic_bar(tdf),width="stretch",config={"displayModeBar": False})

        with tp2:
            with card_start():
                st.plotly_chart(topic_sentiment_bar(tdf),width="stretch",config={"displayModeBar": False})

    else:
        st.info("Run `topic_modelling.py` " "to generate topic data.")

    # ══════════════════════════════════════════════════════════════════════════
    # UMAP MAP
    # ══════════════════════════════════════════════════════════════════════════

    section("Review map","topic clusters in semantic space")

    st.markdown(
        """
        <div class="subtle">
        Each dot represents one review. Reviews close together
        have similar semantic representations. Colours represent
        topic clusters, while the labels mark the approximate
        centre of each cluster.
        </div>
        """,
        unsafe_allow_html=True
    )

    has_umap = (
        not tdf.empty
        and "umap_x" in tdf.columns
        and "umap_y" in tdf.columns
        and "cluster" in tdf.columns
        and "cluster_label" in tdf.columns
        and tdf["umap_x"].notna().sum() > 10
        and tdf["umap_y"].notna().sum() > 10
    )

    if has_umap:
        viz = tdf[tdf["umap_x"].notna()& tdf["umap_y"].notna()& tdf["cluster"].notna()].copy()
        viz["cluster_num"] = pd.to_numeric(viz["cluster"],errors="coerce")

        def topic_number(value):
            if pd.isna(value):
                return "?"
            return str(int(value))

        def topic_name(row):
            cluster_num = row["cluster_num"]
            if cluster_num == -1:
                return "Outliers"
            label = row.get("cluster_label","Unknown")
            if pd.isna(label):
                label = "Unknown"

            return str(label)

        viz["topic_name"] = (viz.apply(topic_name,axis=1))

        viz["topic_display"] = (
            viz.apply(
                lambda row: (
                    "Outliers"
                    if row["cluster_num"] == -1
                    else (
                        f"Topic "
                        f"{topic_number(row['cluster_num'])}"
                        f" — "
                        f"{row['topic_name']}"
                    )
                ),
                axis=1
            )
        )

        if "review_text_en" in viz.columns:
            viz["snippet"] = (viz["review_text_en"].fillna("").astype(str).str.replace("\n"," ",regex=False).str[:180])

        else:
            viz["snippet"] = ""

        # ──────────────────────────────────────────────────────────────────────
        # COLOUR PALETTE
        # ──────────────────────────────────────────────────────────────────────

        topic_colors = [
            "#4267B2",
            "#F28E2B",
            "#24A148",
            "#E05263",
            "#7956B8",
            "#1BA3B7",
            "#D8A927",
            "#CC5C8A",
            "#527EAA",
            "#6C9C63",
            "#A36A45",
            "#4D77BE",
            "#9B63B6",
            "#329B86",
            "#D16B4C",
            "#6E7E91",
            "#B47CBA",
            "#3789A9",
            "#B79A30",
            "#788C65",
        ]

        outlier_color = "#B8C0C8"
        clusters = sorted(viz["cluster_num"].dropna().unique())
        normal_clusters = [c for c in clusters if c != -1]
        cluster_colors = {}

        for i, cluster_num in enumerate(normal_clusters):
            cluster_colors[cluster_num] = topic_colors[i % len(topic_colors)]

        cluster_colors[-1] = (outlier_color)

        # ──────────────────────────────────────────────────────────────────────
        # FIGURE
        # ──────────────────────────────────────────────────────────────────────

        fig_sc = go.Figure()

        for cluster_num in clusters:
            sub = viz[viz["cluster_num"]== cluster_num].copy()
            if sub.empty:
                continue

            if cluster_num == -1:
                legend_name = "Outliers"

            else:
                label = sub["topic_name"].iloc[0]
                legend_name = f"Topic " f"{topic_number(cluster_num)}" f" — {label}"

            custom_columns = ["snippet","topic_display",]
            optional_columns = ["rating","user_name","source","final_label","cluster_prob",]

            for col in optional_columns:
                if col in sub.columns:
                    custom_columns.append(col)

            custom = sub[custom_columns].values
            hover = "<b>%{customdata[1]}</b>" "<br><br>"
            data_index = 2

            if "rating" in sub.columns:
                hover += ("Rating: " f"%{{customdata[{data_index}]}} ★" "<br>")
                data_index += 1

            if "user_name" in sub.columns:
                hover += ("User: " f"%{{customdata[{data_index}]}}" "<br>")
                data_index += 1

            if "source" in sub.columns:
                hover += ("Source: " f"%{{customdata[{data_index}]}}" "<br>")
                data_index += 1

            if "final_label" in sub.columns:
                hover += ("Sentiment: " f"%{{customdata[{data_index}]}}" "<br>")
                data_index += 1

            if "cluster_prob" in sub.columns:
                hover += ("Cluster probability: " f"%{{customdata[{data_index}]:.2f}}" "<br>")
                data_index += 1

            hover += ("<br>" "%{customdata[0]}" "<extra></extra>")

            fig_sc.add_trace(
                go.Scatter(
                    x=sub["umap_x"],
                    y=sub["umap_y"],
                    mode="markers",
                    name=legend_name,
                    legendgroup=legend_name,

                    marker=dict(
                        color=cluster_colors[
                            cluster_num
                        ],
                        size=6,
                        opacity=(
                            0.32
                            if cluster_num == -1
                            else 0.58
                        ),
                        line=dict(
                            width=0.35,
                            color="white"
                        )
                    ),

                    customdata=custom,

                    hovertemplate=hover,
                )
            )

        # ──────────────────────────────────────────────────────────────────────
        # CENTRES
        # ──────────────────────────────────────────────────────────────────────

        centers = (

            viz[
                viz["cluster_num"] != -1
            ]

            .groupby(
                [
                    "cluster_num",
                    "topic_name"
                ],
                as_index=False
            )

            .agg(
                x=("umap_x", "mean"),
                y=("umap_y", "mean"),
                reviews=(
                    "umap_x",
                    "size"
                )
            )
        )

        for _, center in centers.iterrows():
            cluster_num = center["cluster_num"]
            color = cluster_colors[cluster_num]
            fig_sc.add_trace(
                go.Scatter(
                    x=[center["x"]],
                    y=[center["y"]],
                    mode="markers",

                    marker=dict(
                        size=13,
                        color="white",
                        line=dict(
                            color=color,
                            width=2.5
                        )
                    ),

                    showlegend=False,

                    hovertemplate=(
                        f"<b>Topic "
                        f"{topic_number(cluster_num)}"
                        f" — "
                        f"{center['topic_name']}"
                        f"</b><br>"
                        f"{int(center['reviews']):,} reviews"
                        "<extra></extra>"
                    )
                )
            )

        # ──────────────────────────────────────────────────────────────────────
        # LABELS
        # ──────────────────────────────────────────────────────────────────────

        for _, center in centers.iterrows():
            cluster_num = center["cluster_num"]
            label = str(center["topic_name"])
            if len(label) > 26:
                label = label[:23]+ "..."

            fig_sc.add_annotation(
                x=center["x"],
                y=center["y"],

                text=(
                    f"<b>"
                    f"{topic_number(cluster_num)}"
                    f"</b> "
                    f"{label}"
                ),

                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                xshift=8,
                yshift=7,
                font=dict(
                    size=10,
                    color="#36485b"
                ),
                bgcolor=( "rgba(255,255,255,0.88)"),
                bordercolor=("rgba(220,225,232,0.9)"),
                borderwidth=1,
                borderpad=3,
            )

        # ──────────────────────────────────────────────────────────────────────
        # LAYOUT
        # ──────────────────────────────────────────────────────────────────────

        fig_sc.update_layout(
            height=650,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fbfcfe",
            margin=dict(
                l=62,
                r=270,
                t=25,
                b=60
            ),
            hovermode="closest",
            font=dict(
                family="Inter, sans-serif",
                color="#465568"
            ),

            legend=dict(
                title=dict(
                    text="<b>Topics</b>",
                    font=dict(
                        size=12
                    )
                ),

                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor=("rgba(255,255,255,0.95)"),
                bordercolor="#dce4ec",
                borderwidth=1,
                font=dict(size=10)
            ),

            xaxis=dict(
                title=dict(text="UMAP 1",font=dict(size=11)),
                showgrid=True,
                gridcolor="#e8edf3",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="#cdd6df",
                showline=True,
                linecolor="#bcc7d2",
                ticks="outside",
                tickfont=dict(
                    size=9,
                    color="#7a8998"
                )
            ),

            yaxis=dict(
                title=dict(
                    text="UMAP 2",
                    font=dict(
                        size=11
                    )
                ),
                showgrid=True,
                gridcolor="#e8edf3",
                gridwidth=1,
                zeroline=True,
                zerolinecolor="#cdd6df",
                showline=True,
                linecolor="#bcc7d2",
                ticks="outside",
                tickfont=dict(size=9,color="#7a8998")
            )
        )

        with card_start():
            st.plotly_chart(fig_sc,width="stretch",config={"displayModeBar": True,"scrollZoom": True,"displaylogo": False})

        # ──────────────────────────────────────────────────────────────────────
        # TOPIC SUMMARY
        # ──────────────────────────────────────────────────────────────────────

        st.markdown("**Topic summary**")
        overview = centers.copy()
        overview["Topic"] = (overview["cluster_num"].apply(lambda x:f"Topic {topic_number(x)}"))
        overview["Label"] = (overview["topic_name"])
        overview["Reviews"] = (overview["reviews"].astype(int))
        overview["UMAP 1"] = (overview["x"].round(2))
        overview["UMAP 2"] = (overview["y"].round(2))
        st.dataframe(overview[["Topic","Label","Reviews","UMAP 1","UMAP 2"]],width="stretch",hide_index=True)

    else:
        st.info("Rerun `topic_modelling.py` — " "UMAP X/Y coordinates and cluster labels " "are required.")

    # ══════════════════════════════════════════════════════════════════════════
    # AI INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════

    ins_key = f"{sel_brand}_{sel_model}" if (sel_brand != "All Brands" and sel_model != "All Models" and sel_source == "Overall")else None
    entry = insights.get(ins_key)if ins_key else None

    if entry:

        section("Brand health","AI-generated interpretation")
        sd = entry["sentiment_dist"]
        a1, a2, a3, a4 = st.columns(4,gap="small")

        with a1:
            st.markdown(kpi_card("Reviews",f"{entry['total_reviews']:,}","In this model selection","purple"),unsafe_allow_html=True)

        with a2:
            st.markdown(
                kpi_card( "Avg. rating",( f"{entry['avg_rating']} ★" if entry["avg_rating"]else "N/A" ), "Overall customer rating", "blue"),
                unsafe_allow_html=True
            )

        with a3:
            st.markdown(kpi_card("Positive",f"{sd['positive']['pct']}%",f"{sd['positive']['count']} reviews","green" ), unsafe_allow_html=True )

        with a4:
            st.markdown(kpi_card("Negative",f"{sd['negative']['pct']}%",f"{sd['negative']['count']} reviews","pink"),unsafe_allow_html=True)

        st.markdown(f"""<div class="insight-card">{entry["overall_summary"]}</div>""",unsafe_allow_html=True)



        # ══════════════════════════════════════════════════════════════════════
        # CLUSTER EXPLORER
        # ══════════════════════════════════════════════════════════════════════

        section("Explore a topic","AI summary + real reviews")
        clusters = [c for c in entry.get("clusters",[])if c.get("n_reviews",0) > 0]

        if clusters:
            cluster_map = {c["clean_label"]: c for c in clusters}

            sel_cluster = st.selectbox("Topic",list(cluster_map.keys()),
                format_func=lambda x: (
                    f"{x} — "
                    f"{cluster_map[x]['n_reviews']} reviews · "
                    f"{cluster_map[x]['sentiment_dist']['positive']['pct']}% positive"
                )
            )

            cluster = cluster_map[sel_cluster]
            csd = cluster["sentiment_dist"]
            ck1, ck2, ck3, ck4 = (st.columns(4))

            with ck1:

                st.markdown(kpi_card("Reviews",f"{cluster['n_reviews']:,}","Reviews in topic","purple"),unsafe_allow_html=True)

            with ck2:
                st.markdown(kpi_card("Positive",f"{csd['positive']['pct']}%",f"{csd['positive']['count']} reviews","green"),unsafe_allow_html=True)

            with ck3:

                st.markdown( kpi_card( "Negative", f"{csd['negative']['pct']}%",f"{csd['negative']['count']} reviews","pink" ), unsafe_allow_html=True )

            with ck4:
                st.markdown(
                    kpi_card( "Neutral",f"{csd['neutral']['pct']}%",f"{csd['neutral']['count']} reviews","violet"),unsafe_allow_html=True)

            st.markdown(f"""<div class="cluster-summary"><b>AI summary:</b>{cluster["summary"]}</div>""",unsafe_allow_html=True)
            words = cluster.get("top_words",[])[:12]

            if words:
                fig_w = go.Figure(
                    go.Bar(x=words,y=list(range(len(words), 0,-1)),
                        marker_color="#6556c9",
                        hovertemplate=(
                            "%{x}"
                            "<extra></extra>"
                        )
                    )
                )
                fig_w.update_layout(
                    height=210,
                    showlegend=False,
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    margin=dict(
                        l=0,
                        r=0,
                        t=25,
                        b=5
                    ),
                    title=dict(
                        text="Top keywords",
                        font=dict(
                            size=13,
                            color="#4c5c6d"
                        )
                    ),
                    xaxis=dict(
                        tickangle=-30,
                        tickfont=dict(
                            size=10
                        )
                    ),
                    yaxis=dict(
                        visible=False
                    )
                )

                with card_start():
                    st.plotly_chart(fig_w,width="stretch",config={"displayModeBar": False})

            # ══════════════════════════════════════════════════════════════════
            # REVIEWS
            # ══════════════════════════════════════════════════════════════════

            st.markdown(f"**Reviews in '{sel_cluster}'**")
            cluster_reviews = tdf[tdf["cluster_label"]== cluster["raw_label"]]if not tdf.empty else pd.DataFrame()

            if not cluster_reviews.empty:
                show_cols = {
                    "user_name":"User",
                    "review_text_en": "Review",
                    "rating":"Stars",
                    "final_label":"Sentiment",
                    "model_conf": "Confidence",
                    "source": "Source",
                    "posted_date": "Date",
                }

                available_cols = [c for c in show_cols if c in cluster_reviews.columns]
                disp = cluster_reviews[available_cols].rename(columns=show_cols).head(50)
                st.dataframe(disp,width="stretch",hide_index=True)

            else:
                for rev in cluster.get("sample_reviews",[])[:5]:
                    st.markdown(f"""<div class="cluster-summary">{rev}</div>""",unsafe_allow_html=True)
        else:
            st.info("No clusters found for this selection.")

    # elif is_single_source:

    #     st.info(
    #         "Run `python ai_insights.py` "
    #         "to generate AI summaries for this selection."
    #     )
    # else:
    #             st.info(
    #         "Run `python ai_insights.py` "
    #         "to generate AI summaries for this selection."
    #     )

    # ══════════════════════════════════════════════════════════════════════════
    # ALL REVIEWS
    # ══════════════════════════════════════════════════════════════════════════

    section("Reviews","inspect the underlying data")
    sent_filter = st.selectbox("Filter by sentiment",["All","positive","neutral","negative"])

    # show_df = (

    #     df

    #     if sent_filter == "All"

    #     else df[
    #         df["final_label"]
    #         == sent_filter
    #     ]
    # )

    show_df = tdf if sent_filter == "All" else tdf[tdf["final_label"]== sent_filter]
    review_columns = ["user_name","review_text_en","rating","final_label","model_conf","blend_tier","posted_date","source"]
    available_review_columns = [c for c in review_columns if c in show_df.columns]
    st.dataframe(
        show_df[available_review_columns]
        .rename(
            columns={
                "user_name":"User",
                "review_text_en":"Review",
                "rating":"Stars",
                "final_label": "Sentiment",
                "model_conf":"Confidence",
                "blend_tier":"Tier",
                "posted_date": "Date",
                "source": "Source"
            }
        )
        .head(100),
        width="stretch",
        hide_index=True
    )

st.markdown("""<div class="footer"> EV Review Intelligence ·Sentiment Analysis · Topic Modeling · UMAP</div>""",unsafe_allow_html=True)