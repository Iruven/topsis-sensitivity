"""Plotly figures for the Streamlit app.

Each function takes analysis results (DataFrames) and returns a figure, so the
charts can be built and checked without running Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sensitivity import OneAtATimeResult


def ranking_bar(ranking: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of TOPSIS scores, best alternative on top."""
    data = ranking.sort_values("score").reset_index(names="alternative")
    fig = px.bar(
        data, x="score", y="alternative", orientation="h", text="rank",
        labels={"score": "TOPSIS score (closeness coefficient)", "alternative": ""},
    )
    fig.update_traces(texttemplate="#%{text}", textposition="outside")
    fig.update_layout(xaxis_range=[0, 1.05], height=max(300, 40 * len(data)))
    return fig


def oat_rank_lines(
    result: OneAtATimeResult, alternatives: Sequence[object] | None = None
) -> go.Figure:
    """Rank of each alternative as the weight of one criterion goes from 0 to 1.

    Args:
        result: Output of :func:`sensitivity.one_at_a_time`.
        alternatives: Alternatives to draw (default: all). Ranks are still
            computed among all alternatives.
    """
    table = result.table
    if alternatives is not None:
        table = table[table["alternative"].isin(list(alternatives))]
    fig = px.line(
        table, x="weight", y="rank", color="alternative", markers=True,
        # 24 distinct colours: the default palette has only 10, so with 11
        # laptops two lines would share a colour.
        color_discrete_sequence=px.colors.qualitative.Dark24,
        labels={"weight": f"Weight of {result.criterion}", "rank": "Rank"},
    )
    fig.update_yaxes(autorange="reversed", dtick=1)  # rank 1 at the top
    fig.add_vline(
        x=result.base_weight, line_dash="dash", line_color="gray",
        annotation_text="current weight", annotation_position="top",
    )
    return fig


def mc_share_bars(summary: pd.DataFrame) -> go.Figure:
    """Share of simulations in first place and in the top three, per alternative."""
    data = (
        summary.sort_values(["share_first", "share_top3"], ascending=False)
        .reset_index(names="alternative")
        .melt(id_vars="alternative", value_vars=["share_first", "share_top3"],
              var_name="measure", value_name="share")
        .replace({"measure": {"share_first": "1st place", "share_top3": "Top 3"}})
    )
    fig = px.bar(
        data, x="alternative", y="share", color="measure", barmode="group",
        labels={"share": "Share of simulations", "alternative": "", "measure": ""},
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def mc_rank_heatmap(rank_distribution: pd.DataFrame) -> go.Figure:
    """Heatmap: share of simulations in which each alternative got each rank."""
    fig = px.imshow(
        rank_distribution, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
        zmin=0, zmax=1, labels={"x": "Rank", "y": "", "color": "Share"},
    )
    fig.update_xaxes(dtick=1)
    return fig
