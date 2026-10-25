"""Exact computations behind every answer. The LLM only phrases what these return."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap


@dataclass
class Explanation:
    prediction: float
    base_value: float  # where the contributions start: the average prediction, or the "before" prediction
    contributions: dict[str, float]  # feature or group name -> contribution; base + sum == prediction
    feature_values: dict = field(default_factory=dict)
    reference_values: dict = field(default_factory=dict)  # "before" values, for change explanations

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predictions as a flat float array, for a model with .predict() or a plain function."""
    f = model.predict if hasattr(model, "predict") else model
    return np.asarray(f(X), dtype=float).reshape(-1)


def group_matrix(values: pd.DataFrame, feature_groups: dict[str, list[str]] | None) -> pd.DataFrame:
    """Sum SHAP values of correlated features into named concepts, so credit isn't split arbitrarily."""
    if not feature_groups:
        return values
    out = {name: values[cols].sum(axis=1) for name, cols in feature_groups.items()}
    used = {c for cols in feature_groups.values() for c in cols}
    for col in values.columns:
        if col not in used:
            out[col] = values[col]
    return pd.DataFrame(out, index=values.index)


def _tree_shap(model, X: pd.DataFrame, reference: pd.DataFrame | None):
    """Fast exact TreeSHAP. Returns None if the model isn't a supported tree model, or if its
    attributions don't add up to the prediction (e.g. a log-link objective explained in raw space)."""
    if reference is not None and not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        return None  # shap's interventional TreeSHAP can't handle categorical columns
    try:
        if reference is None:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model, data=reference, feature_perturbation="interventional")
        values = np.asarray(explainer.shap_values(X, check_additivity=False), dtype=float)
        base = float(np.asarray(explainer.expected_value, dtype=float).reshape(-1)[0])
    except Exception:
        return None
    if values.shape != X.shape:
        return None
    if not np.allclose(base + values.sum(axis=1), predict(model, X), rtol=1e-4, atol=1e-6):
        return None
    return values, np.full(len(X), base)


def _model_agnostic_shap(model, X: pd.DataFrame, background: pd.DataFrame):
    """Works for any model; slower. Permutation/exact SHAP always adds up to the prediction."""
    columns, dtypes = X.columns, X.dtypes
    # SHAP needs a numeric matrix: encode non-numeric columns as category codes, decode before predicting.
    levels = {c: pd.unique(pd.concat([X[c], background[c]]).astype(object))
              for c in columns if not pd.api.types.is_numeric_dtype(X[c])}

    def encode(df):
        out = df.copy()
        for c, lv in levels.items():
            out[c] = pd.Categorical(df[c].astype(object), categories=lv).codes
        return out.to_numpy(dtype=float)

    def f(arr):
        frame = pd.DataFrame(arr, columns=columns)
        for c, lv in levels.items():
            frame[c] = lv[frame[c].astype(int).to_numpy()]
        try:
            frame = frame.astype(dtypes)
        except (TypeError, ValueError):
            pass
        return predict(model, frame)

    masker = shap.maskers.Independent(encode(background), max_samples=len(background))
    explainer = shap.Explainer(f, masker, feature_names=list(columns))
    exp = explainer(encode(X), max_evals=max(500, 2 * X.shape[1] + 1), silent=True)
    return np.asarray(exp.values, dtype=float), np.asarray(exp.base_values, dtype=float).reshape(-1)


def shap_values(model, X: pd.DataFrame, reference: pd.DataFrame | None = None,
                background: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values for each row of X.

    reference: a single row; attributions explain f(x) - f(reference). Features equal to the
        reference get exactly zero.
    background: rows defining "average", used when the model isn't a tree model.
    """
    result = _tree_shap(model, X, reference)
    if result is None:
        bg = reference if reference is not None else background
        if bg is None:
            raise ValueError("background data is required for non-tree models")
        result = _model_agnostic_shap(model, X, bg)
    return result


def _check_row(row: pd.DataFrame, name: str = "row") -> None:
    if not isinstance(row, pd.DataFrame) or len(row) != 1:
        raise ValueError(f"{name} must be a single-row DataFrame")


def explain_prediction(model, row: pd.DataFrame, feature_groups: dict[str, list[str]] | None = None,
                       background: pd.DataFrame | None = None) -> Explanation:
    """Why is this prediction what it is? Contributions relative to the average prediction."""
    _check_row(row)
    values, base = shap_values(model, row, background=background)
    grouped = group_matrix(pd.DataFrame(values, columns=row.columns), feature_groups)
    return Explanation(
        prediction=float(predict(model, row)[0]),
        base_value=float(base[0]),
        contributions={k: float(v) for k, v in grouped.iloc[0].items()},
        feature_values=row.iloc[0].to_dict(),
    )


def explain_change(model, before: pd.DataFrame, after: pd.DataFrame,
                   feature_groups: dict[str, list[str]] | None = None) -> Explanation:
    """Why did the prediction move from f(before) to f(after)?

    Baseline Shapley with `before` as the single reference: contributions sum exactly to
    f(after) - f(before), and inputs that didn't change get exactly zero credit. That property
    is what makes it a "change" explanation rather than a difference of two SHAP explanations,
    where interactions can hand credit to inputs that never moved.
    """
    _check_row(before, "before")
    _check_row(after, "after")
    if list(before.columns) != list(after.columns):
        raise ValueError("before and after must have the same columns")
    values, _ = shap_values(model, after, reference=before)
    grouped = group_matrix(pd.DataFrame(values, columns=after.columns), feature_groups)
    return Explanation(
        prediction=float(predict(model, after)[0]),
        base_value=float(predict(model, before)[0]),
        contributions={k: float(v) for k, v in grouped.iloc[0].items()},
        feature_values=after.iloc[0].to_dict(),
        reference_values=before.iloc[0].to_dict(),
    )


def what_if(model, row: pd.DataFrame, changes: dict, feature_groups: dict[str, list[str]] | None = None
            ) -> tuple[pd.DataFrame, Explanation]:
    """Re-predict with some inputs changed; explain the difference."""
    _check_row(row)
    new = row.copy()
    for col, value in changes.items():
        if col not in new.columns:
            raise ValueError(f"unknown feature {col!r}")
        new[col] = _cast_like(value, row[col])
    return new, explain_change(model, row, new, feature_groups)


def _cast_like(value, column: pd.Series) -> pd.Series:
    """One-value Series with the column's dtype, without truncating decimals or inventing categories."""
    s = pd.Series([value], index=column.index)
    dtype = column.dtype
    if pd.api.types.is_integer_dtype(dtype) and not float(value).is_integer():
        return s.astype(float)
    if isinstance(dtype, pd.CategoricalDtype) and value not in dtype.categories:
        raise ValueError(f"{value!r} is not a known category of {column.name!r}: {list(dtype.categories)}")
    try:
        return s.astype(dtype)
    except (TypeError, ValueError):
        return s
