"""Tests for src/sensitivity.py.

Hand-calculated values (one-at-a-time weight grid, which alternative wins at
small weights in the symmetric example) were supplied by the author. The other
tests check properties that must hold for any data and need no expected numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sensitivity import (
    _format_share,
    monte_carlo,
    one_at_a_time,
    one_at_a_time_weights,
    stability_summary,
)
from topsis import topsis

BASE = pd.DataFrame(
    [[4, 80, 10], [6, 90, 8], [3, 60, 12]], columns=["a", "b", "c"], index=["X", "Y", "Z"]
)
BASE_W = [0.5, 0.3, 0.2]
BASE_D = ["min", "max", "max"]

# Alternative "D" is best on every criterion (lowest a, highest b and c).
DOMINANT = pd.DataFrame(
    [[1, 9, 9], [2, 5, 3], [3, 4, 8]], columns=["a", "b", "c"], index=["D", "P", "Q"]
)


# --- Hand-calculated cases (author) -------------------------------------------


def test_oat_weights_hand_calculated():
    expected = np.array(
        [
            [0.00, 0.60, 0.40],
            [0.25, 0.45, 0.30],
            [0.50, 0.30, 0.20],
            [0.75, 0.15, 0.10],
            [1.00, 0.00, 0.00],
        ]
    )
    weights = one_at_a_time_weights(np.array(BASE_W), j=0, grid=5)
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-12)


def test_oat_symmetric_example_switches_winner_at_half():
    # A = [3, 1], B = [1, 3], both benefit criteria. By symmetry the two are tied
    # when both weights are 0.5; B wins for small weights of the first criterion
    # (author), so A wins for large ones.
    m = pd.DataFrame([[3, 1], [1, 3]], columns=["c1", "c2"], index=["A", "B"])
    result = one_at_a_time(m, [0.5, 0.5], ["max", "max"], "c1", grid=21)

    t = result.winners.index.to_numpy()
    winners = result.winners.to_numpy()
    assert (winners[t < 0.5 - 1e-9] == "B").all()
    assert (winners[np.isclose(t, 0.5)] == "A / B").all()
    assert (winners[t > 0.5 + 1e-9] == "A").all()

    assert result.changes["winner_from"].tolist() == ["B", "A / B"]
    assert result.changes["winner_to"].tolist() == ["A / B", "A"]
    np.testing.assert_allclose(result.changes["weight_from"], [0.45, 0.50])
    np.testing.assert_allclose(result.changes["weight_to"], [0.50, 0.55])


# --- One-at-a-time: properties ------------------------------------------------


def test_oat_weight_rows_sum_to_one_and_keep_proportions():
    base = np.array([0.1, 0.4, 0.3, 0.2])
    j = 2
    weights = one_at_a_time_weights(base, j=j, grid=11)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[:, j], np.linspace(0, 1, 11))
    others = np.delete(weights, j, axis=1)[:-1]  # last row: all others are 0
    expected = np.delete(base, j) / np.delete(base, j).sum()
    np.testing.assert_allclose(others / others.sum(axis=1, keepdims=True),
                               np.tile(expected, (len(others), 1)))


def test_oat_matches_topsis_at_every_grid_point():
    result = one_at_a_time(BASE, BASE_W, BASE_D, "b", grid=11)
    weights = one_at_a_time_weights(np.array(BASE_W), j=1, grid=11)
    for t, w in zip(np.linspace(0, 1, 11), weights):
        rows = result.table[np.isclose(result.table["weight"], t)].set_index("alternative")
        expected = topsis(BASE, w, BASE_D)
        np.testing.assert_allclose(rows["score"], expected["score"], rtol=0, atol=1e-12)
        assert rows["rank"].tolist() == expected["rank"].tolist()


def test_oat_table_shape_and_base_weight():
    result = one_at_a_time(BASE, [5, 3, 2], BASE_D, "c", grid=21)
    assert len(result.table) == 21 * len(BASE)
    assert result.criterion == "c"
    assert result.base_weight == pytest.approx(0.2)


@pytest.mark.parametrize("criterion", ["a", "b", "c"])
def test_oat_dominant_alternative_always_wins(criterion):
    result = one_at_a_time(DOMINANT, [1, 1, 1], BASE_D, criterion, grid=21)
    assert (result.winners == "D").all()
    assert result.changes.empty


# --- Monte Carlo: properties --------------------------------------------------


def test_mc_same_seed_gives_identical_result():
    first = monte_carlo(BASE, BASE_W, BASE_D, n=2000, seed=42)
    second = monte_carlo(BASE, BASE_W, BASE_D, n=2000, seed=42)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.rank_distribution, second.rank_distribution)
    pd.testing.assert_frame_equal(first.sampled_weights, second.sampled_weights)


def test_mc_different_seed_gives_different_weights():
    first = monte_carlo(BASE, BASE_W, BASE_D, n=2000, seed=1)
    second = monte_carlo(BASE, BASE_W, BASE_D, n=2000, seed=2)
    assert not first.sampled_weights.equals(second.sampled_weights)


def test_mc_zero_weight_stays_zero():
    result = monte_carlo(BASE, [0.5, 0.0, 0.5], BASE_D, n=1000)
    assert (result.sampled_weights["b"] == 0).all()


def test_mc_results_are_consistent():
    result = monte_carlo(BASE, BASE_W, BASE_D, n=3000, concentration=5)
    m = len(BASE)
    np.testing.assert_allclose(result.sampled_weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.rank_distribution.sum(axis=1), 1.0)
    assert (result.summary["share_first"] <= result.summary["share_top3"]).all()
    assert result.summary["mean_rank"].between(1, m).all()
    np.testing.assert_allclose(result.summary["share_first"], result.rank_distribution[1])
    np.testing.assert_allclose(
        result.summary["share_top3"], result.rank_distribution[[1, 2, 3]].sum(axis=1)
    )


def test_mc_dominant_alternative_wins_every_simulation():
    result = monte_carlo(DOMINANT, [1, 1, 1], BASE_D, n=2000, concentration=1)
    assert result.summary.loc["D", "share_first"] == 1.0
    assert result.summary.loc["D", "mean_rank"] == 1.0


def test_mc_mean_sampled_weights_close_to_base():
    # Std of the mean of a weight of 0.2 at n=5000, c=50 is about
    # 0.056 / sqrt(5000) = 0.0008, so 0.005 is about 6 standard deviations.
    result = monte_carlo(BASE, BASE_W, BASE_D, n=5000, concentration=50, seed=0)
    np.testing.assert_allclose(result.sampled_weights.mean(), BASE_W, rtol=0, atol=0.005)


def test_mc_huge_concentration_reproduces_topsis_winner():
    # With concentration 1e6 the sampled weights barely move from the base ones.
    winner = topsis(BASE, BASE_W, BASE_D)["rank"].idxmin()
    result = monte_carlo(BASE, BASE_W, BASE_D, n=1000, concentration=1e6)
    assert result.summary.loc[winner, "share_first"] == 1.0


# --- Text summary --------------------------------------------------------------


def test_summary_mentions_winner_and_uses_labels():
    result = monte_carlo(DOMINANT, [1, 1, 1], BASE_D, n=500)
    text = stability_summary(result, labels={"D": "Dominant laptop"})
    assert "Dominant laptop is first in 100% of simulations" in text
    assert "500 simulations" in text


def test_summary_notes_ties():
    tied = pd.DataFrame([[1, 9], [1, 9], [5, 2]], columns=["a", "b"], index=["T1", "T2", "L"])
    result = monte_carlo(tied, [1, 1], ["min", "max"], n=200)
    assert "ties count as first place" in stability_summary(result)


@pytest.mark.parametrize(
    "share, text", [(0.0, "0%"), (0.004, "<1%"), (0.005, "<1%"), (0.0099, "<1%"), (0.82, "82%"), (1.0, "100%")]
)
def test_format_share_never_shows_zero_for_a_winner(share, text):
    assert _format_share(share) == text


# --- Invalid input -------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"criterion": "nope"}, "Unknown criterion"),
        ({"criterion": "a", "grid": 1}, "grid must be"),
        ({"criterion": "a", "grid": 2.5}, "grid must be"),
    ],
    ids=["unknown_criterion", "grid_too_small", "grid_not_int"],
)
def test_oat_invalid_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        one_at_a_time(BASE, BASE_W, BASE_D, **kwargs)


def test_oat_all_other_weights_zero():
    with pytest.raises(ValueError, match="zero weight"):
        one_at_a_time(BASE, [1, 0, 0], BASE_D, "a")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n": 0}, "n must be"),
        ({"n": 10.5}, "n must be"),
        ({"concentration": 0}, "concentration must be"),
        ({"concentration": -1}, "concentration must be"),
        ({"concentration": float("nan")}, "concentration must be"),
    ],
    ids=["n_zero", "n_not_int", "concentration_zero", "concentration_negative",
         "concentration_nan"],
)
def test_mc_invalid_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        monte_carlo(BASE, BASE_W, BASE_D, **kwargs)
