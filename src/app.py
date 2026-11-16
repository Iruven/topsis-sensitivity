"""Streamlit app: TOPSIS ranking with weight sensitivity analysis.

Run from the project root:  streamlit run src/app.py

All computation lives in topsis.py and sensitivity.py; this file only handles
the user interface: loading data, collecting settings and showing results.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

import charts
from sensitivity import MonteCarloResult, monte_carlo, one_at_a_time, stability_summary
from topsis import topsis

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "laptops_clean.csv"

# Criteria that are costs (lower is better) in the example data set.
# Any other numeric column defaults to "max".
DEFAULT_COST_CRITERIA = frozenset({"Price", "Weight"})

DIRECTION_LABELS = {"max": "max (benefit)", "min": "min (cost)"}


# --- Data loading --------------------------------------------------------------


@st.cache_data
def load_csv(source) -> pd.DataFrame:
    """Read a CSV from a path or an uploaded file (cached by content)."""
    return pd.read_csv(source)


def choose_data() -> pd.DataFrame:
    """Sidebar: pick the example data set or upload a CSV."""
    st.sidebar.header("Data")
    choice = st.sidebar.radio("Source", ["Example: laptops", "Upload CSV"])
    if choice == "Upload CSV":
        uploaded = st.sidebar.file_uploader("CSV file", type="csv")
        if uploaded is None:
            st.info("Upload a CSV file: one row per alternative, numeric columns as criteria.")
            st.stop()
        return load_csv(uploaded)
    if not EXAMPLE_PATH.exists():
        raise ValueError(
            f"Example file not found: {EXAMPLE_PATH.name}. Run `python src/clean_data.py` first."
        )
    return load_csv(EXAMPLE_PATH)


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sidebar: choose the ID column and the criteria.

    Returns:
        ``(matrix, info)``: the numeric decision matrix indexed by ID and the
        remaining (descriptive) columns, also indexed by ID.
    """
    columns = list(df.columns)
    default_id = columns.index("ID") if "ID" in columns else 0
    id_col = st.sidebar.selectbox("ID column", columns, index=default_id)

    ids = df[id_col].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"Column {id_col!r} has duplicate values; IDs must be unique.")

    numeric = [c for c in columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("The file has no numeric columns to use as criteria.")
    criteria = st.sidebar.multiselect("Criteria", numeric, default=numeric)
    if not criteria:
        raise ValueError("Select at least one criterion.")

    data = df.set_index(ids.rename("ID"))
    info = data.drop(columns=[id_col, *criteria], errors="ignore")
    return data[criteria], info


# --- Settings ------------------------------------------------------------------


def criteria_settings(criteria: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    """Sidebar: include checkbox, direction and weight slider for every criterion.

    An excluded criterion gets weight 0, which removes its influence on the
    ranking exactly as if the column were dropped. It stays in the matrix, so
    one-at-a-time can still vary its weight from 0 to 1, and Monte Carlo keeps
    it at 0 in every simulation.
    """
    st.sidebar.header("Criteria")
    weights: dict[str, float] = {}
    directions: dict[str, str] = {}
    for c in criteria:
        with st.sidebar.expander(c, expanded=False):
            included = st.checkbox("Include", value=True, key=f"use_{c}")
            default = "min" if c in DEFAULT_COST_CRITERIA else "max"
            directions[c] = st.radio(
                "Direction", ["max", "min"], index=["max", "min"].index(default),
                format_func=DIRECTION_LABELS.get, key=f"dir_{c}", horizontal=True,
            )
            weight = st.slider("Weight", 0.0, 1.0, 0.5, 0.05, key=f"w_{c}",
                               disabled=not included)
            weights[c] = weight if included else 0.0
    excluded = [c for c in criteria if not st.session_state.get(f"use_{c}", True)]
    if excluded:
        st.sidebar.caption("Excluded (weight 0): " + ", ".join(excluded))
    st.sidebar.caption("Weights are normalized to sum to 1; only their proportions matter.")
    return weights, directions


def simulation_settings() -> tuple[int, float, int]:
    """Sidebar: Monte Carlo parameters."""
    st.sidebar.header("Monte Carlo")
    n = st.sidebar.number_input("Simulations (n)", 100, 50_000, 5000, step=500)
    concentration = st.sidebar.number_input(
        "Concentration", min_value=0.1, value=50.0, step=5.0,
        help="Higher = sampled weights stay closer to the chosen ones.",
    )
    seed = st.sidebar.number_input("Seed", 0, 2**31 - 1, 0, step=1)
    return int(n), float(concentration), int(seed)


# --- Cached computations ---------------------------------------------------------


@st.cache_data
def run_monte_carlo(
    matrix: pd.DataFrame, weights: dict[str, float], directions: dict[str, str],
    n: int, concentration: float, seed: int,
) -> MonteCarloResult:
    """Monte Carlo is the slowest step, so its result is cached per settings."""
    return monte_carlo(matrix, weights, directions, n=n, concentration=concentration, seed=seed)


# --- Tabs ----------------------------------------------------------------------------


def ranking_tab(ranking: pd.DataFrame, info: pd.DataFrame, weights: dict[str, float]) -> None:
    st.plotly_chart(charts.ranking_bar(ranking), width="stretch")
    table = ranking.join(info).sort_values("rank")
    st.dataframe(table, column_config={"score": st.column_config.NumberColumn(format="%.4f")})
    total = sum(weights.values())
    st.caption("Normalized weights: " + ", ".join(
        f"{c} {w / total:.2f}" for c, w in weights.items()
    ))


def oat_tab(
    matrix: pd.DataFrame, weights: dict[str, float], directions: dict[str, str],
    ranking: pd.DataFrame,
) -> None:
    col1, col2 = st.columns(2)
    criterion = col1.selectbox("Criterion to vary", list(matrix.columns))
    grid = col2.slider("Grid points", 5, 101, 21, 2)
    # Many crossing lines are hard to follow, so show the current top 5 by default.
    by_rank = ranking.sort_values("rank").index.tolist()
    shown = st.multiselect("Alternatives on the chart", by_rank, default=by_rank[:5])
    result = one_at_a_time(matrix, weights, directions, criterion, grid=grid)

    st.plotly_chart(charts.oat_rank_lines(result, shown), width="stretch")
    if result.changes.empty:
        st.success(f"The winner does not change for any weight of {criterion} "
                   f"(always {result.winners.iloc[0]}).")
    else:
        st.write(f"Winner changes (the switch lies between *weight_from* and *weight_to*; "
                 f"current weight of {criterion}: {result.base_weight:.2f}):")
        st.dataframe(result.changes, hide_index=True)


def monte_carlo_tab(result: MonteCarloResult) -> None:
    st.text(stability_summary(result))
    st.plotly_chart(charts.mc_share_bars(result.summary), width="stretch")
    st.subheader("Rank distribution")
    st.plotly_chart(charts.mc_rank_heatmap(result.rank_distribution), width="stretch")
    st.dataframe(result.summary.sort_values("mean_rank"), column_config={
        "share_first": st.column_config.NumberColumn(format="percent"),
        "share_top3": st.column_config.NumberColumn(format="percent"),
        "mean_rank": st.column_config.NumberColumn(format="%.2f"),
    })


def export_tab(ranking: pd.DataFrame, info: pd.DataFrame) -> None:
    table = ranking.join(info).sort_values("rank")
    st.download_button(
        "Download ranking (CSV)", table.to_csv().encode("utf-8"),
        file_name="topsis_ranking.csv", mime="text/csv",
    )


@contextmanager
def show_errors() -> Iterator[None]:
    """Show a ValueError raised inside the block as an error message."""
    try:
        yield
    except ValueError as exc:
        st.error(str(exc))


# --- Main ------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="TOPSIS with sensitivity analysis", layout="wide")
    st.title("TOPSIS ranking with weight sensitivity analysis")
    try:
        df = choose_data()
        matrix, info = build_matrix(df)
        weights, directions = criteria_settings(list(matrix.columns))
        n, concentration, seed = simulation_settings()

        ranking = topsis(matrix, weights, directions)
    except ValueError as exc:
        # Validation errors are shown as messages, not as a stack trace.
        st.error(str(exc))
        return

    tabs = st.tabs(["Ranking", "One-at-a-time", "Monte Carlo", "Export"])
    # Each tab handles its own errors, so a problem in one analysis
    # (e.g. invalid one-at-a-time settings) does not hide the others.
    with tabs[0], show_errors():
        ranking_tab(ranking, info, weights)
    with tabs[1], show_errors():
        oat_tab(matrix, weights, directions, ranking)
    with tabs[2], show_errors():
        monte_carlo_tab(run_monte_carlo(matrix, weights, directions, n, concentration, seed))
    with tabs[3], show_errors():
        export_tab(ranking, info)


if __name__ == "__main__":
    main()
