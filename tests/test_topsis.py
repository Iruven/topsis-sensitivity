"""Tests for src/topsis.py.

Expected values in the hand-calculated cases were supplied by the author.
Property tests compare TOPSIS outputs with each other and need no
hand-calculated numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topsis import topsis


# Expected scores are given to 4 decimal places.
SCORE_TOL = 5e-5

HAND_CASES = [
    pytest.param(
        [[4, 80, 10], [6, 90, 8], [3, 60, 12]],
        [0.5, 0.3, 0.2],
        ["min", "max", "max"],
        [0.6578,0.2531,0.7469],
        [1, 2, 0],
        id="mixed_gain_cost",
    ),
    pytest.param(
        [[20, 15, 30], [10, 25, 20], [15, 10, 15]],
        [0.4, 0.4, 0.2],
        ["min", "min", "min"],
        [0.4198,0.4445,0.7494],
        [2, 1, 0],
        id="all_cost",
    ),
]




# --- Hand-calculated cases (author) -------------------------------------------


@pytest.mark.parametrize("data, weights, signs, expected_score, expected_rank0", HAND_CASES)
def test_hand_calculated_scores(data, weights, signs, expected_score, expected_rank0):
    result = topsis(pd.DataFrame(data), weights, signs)
    np.testing.assert_allclose(result["score"], expected_score, rtol=0, atol=SCORE_TOL)


@pytest.mark.parametrize("data, weights, signs, expected_score, expected_rank0", HAND_CASES)
def test_hand_calculated_ranks(data, weights, signs, expected_score, expected_rank0):
    # The author's ranks are 0-based (0 = best); topsis() returns 1-based ranks.
    result = topsis(pd.DataFrame(data), weights, signs)
    assert result["rank"].tolist() == [r + 1 for r in expected_rank0]


# --- Properties of the method -------------------------------------------------

BASE = pd.DataFrame([[4, 80, 10], [6, 90, 8], [3, 60, 12]], columns=["a", "b", "c"])
BASE_W = [0.5, 0.3, 0.2]
BASE_D = ["min", "max", "max"]


def test_invariant_to_column_scale():
    # Vector normalization removes units: e.g. price in PLN or in thousands of PLN.
    scaled = BASE.copy()
    scaled["b"] = scaled["b"] * 1000
    pd.testing.assert_frame_equal(topsis(BASE, BASE_W, BASE_D), topsis(scaled, BASE_W, BASE_D))


def test_invariant_to_weight_scale():
    # Weights are normalized to sum to 1, so only their proportions matter.
    pd.testing.assert_frame_equal(
        topsis(BASE, BASE_W, BASE_D), topsis(BASE, [w * 7 for w in BASE_W], BASE_D)
    )


def test_zero_weight_is_same_as_dropping_the_column():
    # The app's "Include" checkbox relies on this: an excluded criterion gets weight 0.
    with_zero = topsis(BASE, [0.5, 0.0, 0.2], BASE_D)
    dropped = topsis(BASE.drop(columns="b"), [0.5, 0.2], ["min", "max"])
    np.testing.assert_allclose(with_zero["score"], dropped["score"], rtol=0, atol=1e-12)
    assert with_zero["rank"].tolist() == dropped["rank"].tolist()


def test_dominant_alternative_is_ideal():
    # Best on every criterion -> equals the ideal solution -> d+ = 0 -> score 1.
    m = pd.DataFrame([[1, 9, 9], [2, 5, 3], [3, 4, 8]])
    result = topsis(m, [1, 1, 1], ["min", "max", "max"])
    assert result.loc[0, "score"] == pytest.approx(1.0)
    assert result.loc[0, "rank"] == 1


def test_identical_alternatives_share_score_and_rank():
    m = pd.DataFrame([[5, 7, 2], [5, 7, 2], [3, 9, 4]])
    result = topsis(m, [1, 1, 1], ["min", "max", "max"])
    assert result.loc[0, "score"] == result.loc[1, "score"]
    assert result.loc[0, "rank"] == result.loc[1, "rank"]


def test_ties_use_competition_ranking():
    # Two tied winners -> ranks 1, 1, 3 (author's convention).
    m = pd.DataFrame([[1, 9], [1, 9], [5, 2]])
    assert topsis(m, [1, 1], ["min", "max"])["rank"].tolist() == [1, 1, 3]


def test_index_is_preserved():
    m = BASE.set_axis(["X", "Y", "Z"])
    assert topsis(m, BASE_W, BASE_D).index.tolist() == ["X", "Y", "Z"]


def test_weights_as_mapping_equal_weights_as_list():
    as_dict = dict(zip(BASE.columns, BASE_W))
    pd.testing.assert_frame_equal(topsis(BASE, as_dict, BASE_D), topsis(BASE, BASE_W, BASE_D))


# --- Invalid input -------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix, weights, directions, message",
    [
        (BASE.astype(float).mask(BASE == 80), BASE_W, BASE_D, "Missing values"),
        (BASE.assign(b=["x", "y", "z"]), BASE_W, BASE_D, "Non-numeric"),
        (BASE.assign(c=0), BASE_W, BASE_D, "zero norm"),
        (BASE, [0.5, -0.3, 0.2], BASE_D, "non-negative"),
        (BASE, [0, 0, 0], BASE_D, "must not all be zero"),
        (BASE, [0.5, 0.5], BASE_D, "Expected 3 weights"),
        (BASE, BASE_W, ["min", "max", "up"], "'max' or 'min'"),
        (BASE, BASE_W, ["min", "max"], "Expected 3 directions"),
        (pd.DataFrame(), [], [], "at least one row"),
    ],
    ids=[
        "missing_value", "non_numeric", "zero_norm", "negative_weight",
        "zero_weights", "wrong_weight_count", "bad_direction",
        "wrong_direction_count", "empty_matrix",
    ],
)
def test_invalid_input_raises_value_error(matrix, weights, directions, message):
    with pytest.raises(ValueError, match=message):
        topsis(matrix, weights, directions)


def test_all_alternatives_identical_raises():
    # Score would be 0/0; author's decision: raise ValueError.
    m = pd.DataFrame([[5, 7], [5, 7]])
    with pytest.raises(ValueError, match="identical"):
        topsis(m, [1, 1], ["min", "max"])
