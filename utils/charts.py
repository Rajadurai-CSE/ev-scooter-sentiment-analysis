import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils.helpers import SENTIMENT_COLORS, SOURCE_COLORS, BRAND_COLORS, top_words

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=13),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _apply(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15, color="#444")),
                      **_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


def source_kpi_bar(source_counts: dict) -> go.Figure:
    sources = list(source_counts.keys())
    counts  = list(source_counts.values())
    colors  = [SOURCE_COLORS.get(s, "#888") for s in sources]
    fig = go.Figure(go.Bar(
        x=counts, y=sources, orientation="h",
        marker_color=colors, text=counts, textposition="auto",
    ))
    fig.update_layout(height=120, showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0),
                      **{k: v for k, v in _LAYOUT.items() if k not in ["margin","legend"]})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(tickfont=dict(size=12))
    return fig



def rating_bar(df: pd.DataFrame) -> go.Figure:
    counts = (df["rating_rounded"]
              .dropna()
              .astype(int)
              .value_counts()
              .reindex([1,2,3,4,5], fill_value=0)
              .reset_index())
    counts.columns = ["rating", "count"]
    colors = ["#E24B4A","#E8593C","#BA7517","#1D9E75","#0F6E56"]
    fig = go.Figure(go.Bar(
        x=counts["rating"].astype(str),
        y=counts["count"],
        marker_color=colors,
        text=counts["count"],
        textposition="outside",
    ))
    return _apply(fig, "Rating Distribution (1–5)")


def rating_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["rating"].dropna(),
        nbinsx=20,
        marker_color="#7F77DD",
        opacity=0.85,
    ))
    return _apply(fig, "Raw Rating Distribution")


def rating_over_time(df: pd.DataFrame) -> go.Figure:
    """Monthly average rating line graph."""
    monthly = (df.dropna(subset=["posted_date", "rating"])
               .set_index("posted_date")
               .resample("ME")["rating"]
               .agg(["mean", "count"])
               .reset_index())
    monthly.columns = ["date", "avg_rating", "count"]
    monthly = monthly[monthly["count"] >= 3]   # skip months with < 3 reviews

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["avg_rating"].round(2),
        mode="lines+markers",
        line=dict(color="#1D9E75", width=2.5),
        marker=dict(size=7),
        name="Avg Rating",
        hovertemplate="%{x|%b %Y}<br>Avg: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color="#aaa",
                  annotation_text="Neutral (3.0)", annotation_position="bottom right")
    fig.update_yaxes(range=[1, 5.2])
    return _apply(fig, "Monthly Avg Rating Trend")


def sentiment_pie(df: pd.DataFrame, label_col: str = "final_label") -> go.Figure:
    counts = df[label_col].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [SENTIMENT_COLORS.get(l, "#888") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.42,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
    ))
    fig.update_layout(showlegend=False, height=300, **_LAYOUT)
    return fig


def platform_bias_chart(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("source").agg(
        avg_rating=("rating", "mean"),
        pct_positive=("final_label", lambda x: (x == "positive").mean() * 100),
        n=("review_id", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Rating (×20 scale)",
        x=grp["source"],
        y=grp["avg_rating"] * 20,   
        marker_color=[SOURCE_COLORS.get(s, "#888") for s in grp["source"]],
        text=grp["avg_rating"].round(2),
        textposition="outside",
        customdata=grp["avg_rating"],
        hovertemplate="%{x}<br>Avg Rating: %{customdata:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="% Positive Sentiment",
        x=grp["source"],
        y=grp["pct_positive"].round(1),
        marker_color=["rgba(29,158,117,0.5)"] * len(grp),
        text=grp["pct_positive"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="%{x}<br>Positive: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(barmode="group")
    return _apply(fig, "Platform Bias — Avg Rating vs % Positive")



# def word_treemap(df: pd.DataFrame, sentiment: str = "positive") -> go.Figure:
#     subset = df[df["final_label"] == sentiment]["review_text_en"]
#     freq   = top_words(subset, n=40)
#     if not freq:
#         fig = go.Figure()
#         fig.update_layout(title=f"No {sentiment} reviews in selection", **_LAYOUT)
#         return fig

#     labels = list(freq.keys())
#     values = list(freq.values())
#     color  = SENTIMENT_COLORS.get(sentiment, "#888")

#     fig = go.Figure(go.Treemap(
#         labels=labels,
#         parents=[""] * len(labels),
#         values=values,
#         marker=dict(
#             colors=values,
#             colorscale=[[0, f"{color}44"], [1, color]],
#             showscale=False,
#         ),
#         textinfo="label+value",
#         hovertemplate="%{label}: %{value} mentions<extra></extra>",
#     ))
#     fig.update_layout(height=380, margin=dict(l=0,r=0,t=30,b=0),
#                       paper_bgcolor="rgba(0,0,0,0)",
#                       title=dict(text=f"Top Words — {sentiment.title()} Reviews",
#                                  font=dict(size=15, color="#444")))
#     return fig

def word_treemap(df: pd.DataFrame, sentiment: str = "positive") -> go.Figure:
    """Treemap of most frequent words for a given sentiment."""
    subset = df[df["final_label"] == sentiment]["review_text_en"]
    freq   = top_words(subset, n=40)
    if not freq:
        fig = go.Figure()
        fig.update_layout(title=f"No {sentiment} reviews in selection", **_LAYOUT)
        return fig

    labels = list(freq.keys())
    values = list(freq.values())
    color  = SENTIMENT_COLORS.get(sentiment, "#888")

    # Map sentiments to valid rgba strings for Plotly transparency
    transparent_map = {
        "positive": "rgba(29, 158, 117, 0.3)",
        "neutral":  "rgba(186, 117, 23, 0.3)",
        "negative": "rgba(232, 89, 60, 0.3)"
    }
    color_transparent = transparent_map.get(sentiment, "rgba(136, 136, 136, 0.3)")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        marker=dict(
            colors=values,
            colorscale=[[0, color_transparent], [1, color]],
            showscale=False,
        ),
        textinfo="label+value",
        hovertemplate="%{label}: %{value} mentions<extra></extra>",
    ))
    fig.update_layout(height=380, margin=dict(l=0,r=0,t=30,b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      title=dict(text=f"Top Words — {sentiment.title()} Reviews",
                                 font=dict(size=15, color="#444")))
    return fig



def source_comparison_bars(df: pd.DataFrame) -> go.Figure:
    """Stacked sentiment bar by source."""
    grp = (df.groupby(["source", "final_label"])
             .size()
             .reset_index(name="count"))
    total = grp.groupby("source")["count"].transform("sum")
    grp["pct"] = grp["count"] / total * 100

    fig = go.Figure()
    for sentiment in ["positive", "neutral", "negative"]:
        sub = grp[grp["final_label"] == sentiment]
        fig.add_trace(go.Bar(
            name=sentiment.title(),
            x=sub["source"],
            y=sub["pct"].round(1),
            marker_color=SENTIMENT_COLORS[sentiment],
            text=sub["pct"].round(0).astype(int).astype(str) + "%",
            textposition="inside",
            hovertemplate="%{x}<br>" + sentiment + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(barmode="stack")
    return _apply(fig, "Sentiment % by Source")


def source_rating_box(df: pd.DataFrame) -> go.Figure:
    """Box plot of rating distribution per source."""
    fig = go.Figure()
    for src in df["source"].unique():
        sub = df[df["source"] == src]["rating"].dropna()
        fig.add_trace(go.Box(
            y=sub,
            name=src,
            marker_color=SOURCE_COLORS.get(src, "#888"),
            boxmean="sd",
        ))
    return _apply(fig, "Rating Distribution by Source")


def source_volume_bar(df: pd.DataFrame) -> go.Figure:
    """Review volume by source."""
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["source", "count"]
    fig = go.Figure(go.Bar(
        x=counts["source"],
        y=counts["count"],
        marker_color=[SOURCE_COLORS.get(s, "#888") for s in counts["source"]],
        text=counts["count"],
        textposition="outside",
    ))
    return _apply(fig, "Review Volume by Source")


def brand_sentiment_heatmap(df: pd.DataFrame) -> go.Figure:
    """Brand × Sentiment matrix as a heatmap — shows which brand has most negative reviews."""
    pivot = (df.groupby(["brand", "final_label"])
               .size()
               .unstack(fill_value=0))
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["positive", "neutral", "negative"]]
    # Normalise to %
    row_total = pivot.sum(axis=1)
    pct_pivot = pivot.div(row_total, axis=0) * 100

    fig = go.Figure(go.Heatmap(
        z=pct_pivot.values,
        x=["Positive", "Neutral", "Negative"],
        y=pct_pivot.index.tolist(),
        colorscale=[[0,"#E24B4A"],[0.5,"#FAC775"],[1,"#1D9E75"]],
        text=pct_pivot.round(1).astype(str) + "%",
        texttemplate="%{text}",
        showscale=True,
        hovertemplate="Brand: %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(height=260)
    return _apply(fig, "Sentiment % by Brand")


# ── Topic section ─────────────────────────────────────────────────────────────

# def topic_bar(topic_df: pd.DataFrame) -> go.Figure:
#     """Topic distribution bar (requires topic_label column)."""
#     if "topic_label" not in topic_df.columns:
#         return go.Figure()
#     counts = topic_df["topic_label"].value_counts().head(12).reset_index()
#     counts.columns = ["topic", "count"]
#     fig = go.Figure(go.Bar(
#         y=counts["topic"], x=counts["count"],
#         orientation="h",
#         marker_color="#7F77DD",
#         text=counts["count"], textposition="auto",
#     ))
#     fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
#     return _apply(fig, "Topic Distribution")


# def topic_sentiment_bar(topic_df: pd.DataFrame) -> go.Figure:
#     """Stacked sentiment by topic."""
#     if "topic_label" not in topic_df.columns:
#         return go.Figure()
#     grp = (topic_df.groupby(["topic_label", "final_label"])
#                    .size().reset_index(name="count"))
#     top_topics = topic_df["topic_label"].value_counts().head(8).index
#     grp = grp[grp["topic_label"].isin(top_topics)]

#     fig = go.Figure()
#     for sentiment in ["positive", "neutral", "negative"]:
#         sub = grp[grp["final_label"] == sentiment]
#         fig.add_trace(go.Bar(
#             name=sentiment.title(),
#             x=sub["topic_label"],
#             y=sub["count"],
#             marker_color=SENTIMENT_COLORS[sentiment],
#         ))
#     fig.update_layout(barmode="stack", height=380)
#     return _apply(fig, "Sentiment by Topic")


def topic_bar(topic_df: pd.DataFrame) -> go.Figure:
    """Topic distribution bar (requires topic_label column)."""
    if "cluster_label" not in topic_df.columns:
        return go.Figure()
    
    counts = topic_df[topic_df["is_short"]==False]["cluster_label"].value_counts().head(12).reset_index()
    counts.columns = ["topic", "count"]
    fig = go.Figure(go.Bar(
        y=counts["topic"], x=counts["count"],
        orientation="h",
        marker_color="#7F77DD",
        text=counts["count"], textposition="auto",
    ))
    fig.update_layout(height=380, yaxis=dict(autorange="reversed"))
    return _apply(fig, "Topic Distribution")


def topic_sentiment_bar(topic_df: pd.DataFrame) -> go.Figure:
    """Stacked sentiment by topic."""
    if "cluster_label" not in topic_df.columns:
        return go.Figure()
    topic_df = topic_df[topic_df["is_short"] == False]
    grp = (topic_df.groupby(["cluster_label", "final_label"])
                   .size().reset_index(name="count"))
    top_topics = topic_df["cluster_label"].value_counts().head(10).index

    grp = grp[grp["cluster_label"].isin(top_topics)]

    grp["cluster_num"] = "Topic "+ grp["cluster_label"].str.split("_", n=1).str[0].astype(str)
    #grp.sort_values(by="cluster_num",inplace=True)
    topics = sorted(grp['cluster_num'].unique().tolist())


    fig = go.Figure()

    # for topic in topics:
    #     sub = grp[grp["cluster_label"] == topic]
    #     pos_val = len(sub[sub["final_label"] ==  "positive"])
    #     neg_val = len(sub[sub["final_label"] ==  "negative"])
    #     neu_val = len(sub[sub["final_label"] ==  "neutral"])
    #     fig.add_trace(go.Bar(
    #         x=["positive", "neutral", "negative"],
    #         y=[pos_val,neg_val,neu_val],
    #         name=topic,
    #         marker_color=SENTIMENT_COLORS,
    #         customdata=sub["cluster_label"],
    #         hovertemplate=(
    #             "<b>Topic %{customdata}</b><br>"
    #             "%{customdata}<br>"
    #             f"{sentiment.title()}: "
    #             "%{y} reviews"
    #             "<extra></extra>"
    #         )
    #     ))

    #     fig.update_layout(
    #     barmode='stack',
    #     title="Stacked Bar Chart Example",
    #     xaxis_title="Categories",
    #     yaxis_title="Values",
    #     legend_title="Series",
    #     template="plotly_white"
    #     )

    for sentiment in ["positive", "neutral", "negative"]:
        sub = grp[grp["final_label"] == sentiment]
        fig.add_trace(go.Bar(
            name=sentiment.title(),
            x=sub["cluster_num"],
            y=sub["count"],
            marker_color=SENTIMENT_COLORS[sentiment],
            customdata=sub["cluster_label"],

hovertemplate=(
    "<b>Topic %{x}</b><br>"
    "%{customdata}<br>"
    f"{sentiment.title()}: "
    "%{y} reviews"
    "<extra></extra>"
)
        ))
    fig.update_layout(barmode="stack", height=380)
    return _apply(fig, "Sentiment by Topic")