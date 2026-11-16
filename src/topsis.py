"""TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

The input is an already-numeric decision matrix (alternatives x criteria).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

VALID_DIRECTIONS: frozenset[str] = frozenset({"max", "min"})

WeightsLike = Mapping[str, float] | Sequence[float] | np.ndarray | pd.Series
DirectionsLike = Mapping[str, str] | Sequence[str] | pd.Series


def _align_to_columns(values: object, columns: pd.Index, name: str) -> list:
    """Return ``values`` as a list ordered like ``columns``.

    Accepts a mapping keyed by column name (order-independent) or a
    sequence that is assumed to follow the column order.
    """
    if isinstance(values, (Mapping, pd.Series)):
        keys = set(values.keys())
        missing = [c for c in columns if c not in keys]
        extra = [k for k in keys if k not in set(columns)]
        if missing or extra:
            raise ValueError(
                f"{name} keys must match the matrix columns. "
                f"Missing: {missing or 'none'}; unexpected: {extra or 'none'}."
            )
        return [values[c] for c in columns]
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence or a mapping, not a string.")
    values = list(values)
    if len(values) != len(columns):
        raise ValueError(
            f"Expected {len(columns)} {name} (one per criterion), got {len(values)}."
        )
    return values


def validate_inputs(
    matrix: pd.DataFrame,
    weights: WeightsLike,
    directions: DirectionsLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check the TOPSIS inputs and convert them to numpy arrays.

    Returns:
        ``(x, w, is_max)``: the decision matrix as floats, the weights
        normalized to sum to 1, and a boolean mask (True = benefit/max criterion).

    Raises:
        ValueError: with a human-readable message if any check fails.
    """
    if not isinstance(matrix, pd.DataFrame):
        raise ValueError("The decision matrix must be a pandas DataFrame.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("The decision matrix must have at least one row and one column.")

    non_numeric = [
        str(c)
        for c in matrix.columns
        if not pd.api.types.is_numeric_dtype(matrix[c])
        or pd.api.types.is_bool_dtype(matrix[c])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric values in column(s): {', '.join(non_numeric)}.")

    x = matrix.to_numpy(dtype=float)
    if np.isnan(x).any():
        bad = [str(c) for c in matrix.columns[np.isnan(x).any(axis=0)]]
        raise ValueError(f"Missing values in column(s): {', '.join(bad)}.")
    if not np.isfinite(x).all():
        bad = [str(c) for c in matrix.columns[~np.isfinite(x).all(axis=0)]]
        raise ValueError(f"Infinite values in column(s): {', '.join(bad)}.")

    norms = np.sqrt((x**2).sum(axis=0))
    zero_norm = [str(c) for c, n in zip(matrix.columns, norms) if n == 0]
    if zero_norm:
        raise ValueError(
            f"Column(s) with zero norm (all values are 0): {', '.join(zero_norm)}."
        )

    try:
        w = np.asarray(_align_to_columns(weights, matrix.columns, "weights"), dtype=float)
    except (TypeError, ValueError) as exc:
        if "weights" in str(exc):
            raise
        raise ValueError("Weights must be numbers.") from exc
    if not np.isfinite(w).all():
        raise ValueError("Weights must be finite numbers.")
    if (w < 0).any():
        raise ValueError("Weights must be non-negative.")
    if w.sum() == 0:
        raise ValueError("Weights must not all be zero.")

    dirs = _align_to_columns(directions, matrix.columns, "directions")
    invalid = [d for d in dirs if d not in VALID_DIRECTIONS]
    if invalid:
        raise ValueError(f"Directions must be 'max' or 'min', got: {invalid}.")
    is_max = np.array([d == "max" for d in dirs], dtype=bool)

    return x, w / w.sum(), is_max


def closeness(x: np.ndarray, w: np.ndarray, is_max: np.ndarray) -> np.ndarray:
    """Compute TOPSIS closeness coefficients for one or many weight vectors.

    This is the numeric core of TOPSIS, without input validation (call
    :func:`validate_inputs` first). Accepting a batch of weight vectors lets
    the sensitivity analysis evaluate thousands of weightings in one
    vectorized call while using exactly the same formulas as :func:`topsis`.

    Args:
        x: Decision matrix, shape ``(m, k)`` (m alternatives, k criteria).
        w: Weights summing to 1, shape ``(k,)`` or ``(n, k)`` for n weightings.
        is_max: Boolean mask of shape ``(k,)``; True = benefit (max) criterion.

    Returns:
        Closeness coefficients, shape ``(m,)`` or ``(n, m)``.

    Raises:
        ValueError: if C_i is 0/0 (all alternatives identical on every
            criterion with a positive weight).
    """
    w = np.asarray(w, dtype=float)

    # 1. Vector normalization: r_ij = x_ij / sqrt(sum_i x_ij^2)
    r = x / np.sqrt((x**2).sum(axis=0))

    # 2. Weighted normalized matrix: v_ij = w_j * r_ij
    #    w[..., None, :] has shape (1, k) or (n, 1, k), so v is (m, k) or (n, m, k).
    v = r * w[..., None, :]

    # 3. Ideal (A+) and anti-ideal (A-) solutions, depending on direction.
    #    Max/min are taken over alternatives (axis -2), separately per weighting.
    col_max = v.max(axis=-2, keepdims=True)
    col_min = v.min(axis=-2, keepdims=True)
    ideal = np.where(is_max, col_max, col_min)
    anti_ideal = np.where(is_max, col_min, col_max)

    # 4. Euclidean distances to A+ and A- (summing over criteria, axis -1)
    d_plus = np.sqrt(((v - ideal) ** 2).sum(axis=-1))
    d_minus = np.sqrt(((v - anti_ideal) ** 2).sum(axis=-1))

    # 5. Closeness coefficient C_i = d-_i / (d+_i + d-_i)
    denom = d_plus + d_minus
    if (denom == 0).any():
        # Happens only when all alternatives are identical on every criterion
        # with a positive weight, so A+ == A- and C_i is 0/0.
        # Decision (author): raise an error instead of inventing a score.
        raise ValueError(
            "All alternatives are identical on every weighted criterion; "
            "the TOPSIS score is undefined."
        )
    return d_minus / denom


def competition_rank(scores: np.ndarray) -> np.ndarray:
    """Rank scores in descending order along the last axis ("1-1-3" ranking).

    rank_i = 1 + number of alternatives with a strictly higher score, so tied
    scores share the best rank and the following rank is skipped
    (scores 0.9, 0.9, 0.5 -> ranks 1, 1, 3).

    Args:
        scores: Shape ``(m,)`` or ``(n, m)``.

    Returns:
        Integer ranks with the same shape as ``scores``.
    """
    scores = np.asarray(scores, dtype=float)
    # higher[..., i, j] is True when alternative j scores strictly above i.
    higher = scores[..., np.newaxis, :] > scores[..., :, np.newaxis]
    return 1 + higher.sum(axis=-1)


def topsis(
    matrix: pd.DataFrame,
    weights: WeightsLike,
    directions: DirectionsLike,
) -> pd.DataFrame:
    """Rank alternatives with TOPSIS.

    Args:
        matrix: Numeric decision matrix; rows are alternatives (the index is
            kept as their labels), columns are criteria.
        weights: Non-negative criterion weights, either a mapping
            ``{column: weight}`` or a sequence in column order. They are
            normalized to sum to 1, so only their proportions matter.
        directions: ``"max"`` (benefit) or ``"min"`` (cost) per criterion,
            as a mapping or a sequence in column order.

    Returns:
        DataFrame with the same index as ``matrix`` and columns:
        ``score`` - closeness coefficient C_i in [0, 1] (higher is better),
        ``rank`` - 1 = best; ties use competition ranking (1, 1, 3).

    Raises:
        ValueError: if the inputs are invalid (see :func:`validate_inputs`)
            or the score is undefined (see :func:`closeness`).
    """
    x, w, is_max = validate_inputs(matrix, weights, directions)
    score = closeness(x, w, is_max)
    return pd.DataFrame(
        {"score": score, "rank": competition_rank(score)}, index=matrix.index
    )
