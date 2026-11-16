"""Weight sensitivity analysis for TOPSIS.

Two complementary analyses:

- :func:`one_at_a_time` - change the weight of one criterion from 0 to 1
  (the other weights keep their proportions) and watch how the ranking moves.
- :func:`monte_carlo` - draw many random weight vectors around the chosen
  weights (Dirichlet distribution) and count how often each alternative wins.

Pure functions only: no Streamlit, no file I/O, no global state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topsis import DirectionsLike, WeightsLike, closeness, competition_rank, validate_inputs

TOP_K: int = 3  # "top three" used in the Monte Carlo summary


@dataclass(frozen=True)
class OneAtATimeResult:
    """Result of :func:`one_at_a_time`.

    Attributes:
        criterion: Name of the criterion whose weight was varied.
        base_weight: Its weight in the original (normalized) weighting.
        table: Long table with columns ``weight``, ``alternative``,
            ``score``, ``rank`` - one row per grid point and alternative.
        winners: Series indexed by grid weight; the rank-1 alternative(s),
            tied winners joined with " / ".
        changes: One row per change of winner between consecutive grid points,
            columns ``weight_from``, ``weight_to``, ``winner_from``, ``winner_to``.
            The true switching weight lies between ``weight_from`` and ``weight_to``.
    """

    criterion: str
    base_weight: float
    table: pd.DataFrame
    winners: pd.Series
    changes: pd.DataFrame


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of :func:`monte_carlo`.

    Attributes:
        summary: Indexed by alternative, columns ``share_first``,
            ``share_top3``, ``mean_rank``.
        rank_distribution: Indexed by alternative, one column per rank
            (1..m); each cell is the share of simulations with that rank.
        sampled_weights: The ``n`` weight vectors used, one row each.
        n: Number of simulations.
        concentration: Dirichlet concentration parameter.
        seed: Random seed.
    """

    summary: pd.DataFrame
    rank_distribution: pd.DataFrame
    sampled_weights: pd.DataFrame
    n: int
    concentration: float
    seed: int


def _winner_labels(ranks: np.ndarray, alternatives: pd.Index) -> list[str]:
    """For each row of ``ranks`` (shape (g, m)) return the rank-1 alternative(s)."""
    return [" / ".join(str(a) for a in alternatives[row == 1]) for row in ranks]


def one_at_a_time_weights(base: np.ndarray, j: int, grid: int) -> np.ndarray:
    """Build the weight vectors for the one-at-a-time analysis.

    Criterion ``j`` takes ``grid`` equally spaced values t from 0 to 1. The
    remaining weight (1 - t) is split among the other criteria in the same
    proportions as in ``base``, so every row sums to 1.

    Args:
        base: Normalized base weights, shape ``(k,)``.
        j: Index of the varied criterion.
        grid: Number of grid points (at least 2).

    Returns:
        Array of shape ``(grid, k)``.

    Raises:
        ValueError: if all other criteria have zero weight, because there are
            then no proportions to keep.
    """
    others = base.copy()
    others[j] = 0.0
    total = others.sum()
    if total == 0:
        raise ValueError(
            "All other criteria have zero weight, so the remaining weight "
            "cannot be split proportionally. Give another criterion a positive weight."
        )
    t = np.linspace(0.0, 1.0, grid)
    w = np.outer(1.0 - t, others / total)
    w[:, j] = t
    return w


def one_at_a_time(
    matrix: pd.DataFrame,
    weights: WeightsLike,
    directions: DirectionsLike,
    criterion: str,
    grid: int = 21,
) -> OneAtATimeResult:
    """Vary the weight of one criterion from 0 to 1 and recompute TOPSIS.

    Args:
        matrix, weights, directions: As in :func:`topsis.topsis`.
        criterion: Column whose weight is varied.
        grid: Number of equally spaced weights in [0, 1] (default 21, step 0.05).

    Returns:
        :class:`OneAtATimeResult`.

    Raises:
        ValueError: on invalid inputs.
    """
    x, w, is_max = validate_inputs(matrix, weights, directions)
    if criterion not in matrix.columns:
        raise ValueError(f"Unknown criterion {criterion!r}.")
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError("grid must be an integer of at least 2.")

    j = matrix.columns.get_loc(criterion)
    w_grid = one_at_a_time_weights(w, j, int(grid))
    scores = closeness(x, w_grid, is_max)  # shape (grid, m)
    ranks = competition_rank(scores)
    t = w_grid[:, j]

    m = len(matrix.index)
    table = pd.DataFrame(
        {
            "weight": np.repeat(t, m),
            "alternative": np.tile(matrix.index.to_numpy(), len(t)),
            "score": scores.ravel(),
            "rank": ranks.ravel(),
        }
    )

    winners = pd.Series(_winner_labels(ranks, matrix.index), index=pd.Index(t, name="weight"),
                        name="winner")
    changed = np.flatnonzero(winners.to_numpy()[1:] != winners.to_numpy()[:-1]) + 1
    changes = pd.DataFrame(
        {
            "weight_from": t[changed - 1],
            "weight_to": t[changed],
            "winner_from": winners.to_numpy()[changed - 1],
            "winner_to": winners.to_numpy()[changed],
        }
    )
    return OneAtATimeResult(str(criterion), float(w[j]), table, winners, changes)


def sample_weights(
    base: np.ndarray, n: int, concentration: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` weight vectors from Dirichlet(concentration * base).

    The mean of the distribution equals ``base``; a larger ``concentration``
    keeps the samples closer to it. Criteria with zero base weight stay at
    exactly 0 (a Dirichlet parameter must be positive, and a weight of 0 means
    the criterion was deliberately excluded).

    Returns:
        Array of shape ``(n, k)``; every row sums to 1.
    """
    active = base > 0
    w = np.zeros((n, base.size))
    w[:, active] = rng.dirichlet(concentration * base[active], size=n)
    return w


def monte_carlo(
    matrix: pd.DataFrame,
    weights: WeightsLike,
    directions: DirectionsLike,
    n: int = 5000,
    concentration: float = 50.0,
    seed: int = 0,
) -> MonteCarloResult:
    """Estimate how stable the ranking is under random perturbations of the weights.

    Args:
        matrix, weights, directions: As in :func:`topsis.topsis`.
        n: Number of simulations.
        concentration: Dirichlet concentration (> 0). Larger = weights closer
            to the given ones.
        seed: Random seed; the same seed gives the same result.

    Returns:
        :class:`MonteCarloResult`. Tied alternatives share a rank, so e.g.
        ``share_first`` can sum to more than 1 across alternatives.

    Raises:
        ValueError: on invalid inputs.
    """
    x, w, is_max = validate_inputs(matrix, weights, directions)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be a positive number.")

    rng = np.random.default_rng(seed)
    w_samples = sample_weights(w, int(n), float(concentration), rng)
    ranks = competition_rank(closeness(x, w_samples, is_max))  # shape (n, m)

    m = len(matrix.index)
    summary = pd.DataFrame(
        {
            "share_first": (ranks == 1).mean(axis=0),
            "share_top3": (ranks <= TOP_K).mean(axis=0),
            "mean_rank": ranks.mean(axis=0),
        },
        index=matrix.index,
    )
    rank_distribution = pd.DataFrame(
        np.column_stack([(ranks == r).mean(axis=0) for r in range(1, m + 1)]),
        index=matrix.index,
        columns=pd.Index(range(1, m + 1), name="rank"),
    )
    sampled = pd.DataFrame(w_samples, columns=matrix.columns)
    return MonteCarloResult(summary, rank_distribution, sampled, int(n), float(concentration), seed)


def _format_share(share: float) -> str:
    """Format a share as a percentage; tiny non-zero shares become '<1%'."""
    if 0 < share < 0.005:
        return "<1%"
    return f"{share:.0%}"


def stability_summary(
    result: MonteCarloResult, labels: Mapping[object, str] | None = None
) -> str:
    """Describe a Monte Carlo result in plain English.

    One line per alternative that was first at least once, most frequent first,
    e.g. "L03 is first in 82% of simulations (top 3 in 97%, mean rank 1.3)."

    Args:
        result: Output of :func:`monte_carlo`.
        labels: Optional display names keyed by alternative (e.g. short names).

    Returns:
        Multi-line text.
    """
    labels = labels or {}
    s = result.summary.sort_values(["share_first", "mean_rank"], ascending=[False, True])
    lines = [
        f"Based on {result.n} simulations (concentration = {result.concentration:g}, "
        f"seed = {result.seed}):"
    ]
    for alt, row in s[s["share_first"] > 0].iterrows():
        lines.append(
            f"- {labels.get(alt, alt)} is first in {_format_share(row['share_first'])} "
            f"of simulations (top {TOP_K} in {_format_share(row['share_top3'])}, "
            f"mean rank {row['mean_rank']:.1f})."
        )
    if s["share_first"].sum() > 1 + 1e-12:
        lines.append("Note: ties count as first place for every tied alternative, "
                     "so the shares add up to more than 100%.")
    return "\n".join(lines)
