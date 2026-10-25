"""Explainer: attach to a model, ask questions in plain language."""

from __future__ import annotations

import math
import operator

import numpy as np
import pandas as pd

from explainkit import tools
from explainkit.agent import build_tool_specs, run_agent, system_prompt
from explainkit.grounding import unsupported_numbers
from explainkit.llm import get_llm
from explainkit.narrate import Result
from explainkit.report import Answer, Report

_OPS = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
        "<": operator.lt, "<=": operator.le}

_UNANSWERABLE = {
    "causal": ("The model learned patterns from historical data, not cause and effect. It can show how its "
               "prediction responds if an input is changed (a what-if), but not whether making that change "
               "in the real world would cause the outcome."),
    "future_or_external_events": ("The model only knows the inputs it was given. It can't foresee events or "
                                  "information outside those inputs."),
    "not_in_data": "This information isn't among the model's inputs or the data it was given.",
    "other": "This question can't be answered reliably from the model and its data.",
}


def _r(x):
    """Round for display: whole numbers from 1,000 up, otherwise 4 significant figures."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x) if abs(x) >= 1000 else float(f"{x:.4g}")


def _j(v):
    """JSON-friendly scalar."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and math.isnan(v) else (_r(v) if isinstance(v, float) else v)
    return str(v)


class Explainer:
    """Plain-language, tool-grounded explanations for a regression or forecasting model.

    Args:
        model: anything with .predict(DataFrame), or a function DataFrame -> predictions.
        data: the data the model was trained or is used on. Used as the "average" reference, for
            segment questions, and for accuracy (with y). May hold extra columns (e.g. "station") for filtering.
        target_name: what the model predicts, with units, e.g. "bike rentals per hour".
        y: actual outcomes, as an array or the name of a column in data. Enables accuracy questions.
        features: the model's input columns. Default: all columns of data except y.
        feature_groups: {"concept": [columns]} to report correlated inputs as one factor.
        feature_descriptions: {"column": "plain-language meaning"} to help the LLM phrase answers.
        value_labels: {"column": {code: "label"} or function} to show values as words,
            e.g. {"weekday": {0: "Sunday", ...}, "hour": lambda h: f"{h}:00"}.
        llm: a LiteLLM model string (default "gemini/gemini-2.5-flash") or an object with .chat().
    """

    def __init__(self, model, data: pd.DataFrame, target_name: str = "prediction", *, y=None,
                 features: list[str] | None = None, feature_groups: dict[str, list[str]] | None = None,
                 feature_descriptions: dict[str, str] | None = None, value_labels: dict | None = None,
                 llm="gemini/gemini-2.5-flash",
                 background_size: int = 100, random_state: int = 0):
        self.model = model
        self.data = data.reset_index(drop=True)
        self.target_name = target_name
        if isinstance(y, str):
            self.y = self.data[y].to_numpy(dtype=float)
            default_features = [c for c in self.data.columns if c != y]
        else:
            self.y = None if y is None else np.asarray(y, dtype=float).reshape(-1)
            default_features = list(self.data.columns)
        if self.y is not None and len(self.y) != len(self.data):
            raise ValueError("y must have one value per row of data")
        self.features = list(features or default_features)
        missing = set(self.features) - set(self.data.columns)
        if missing:
            raise ValueError(f"features not in data: {sorted(missing)}")
        self.feature_groups = feature_groups or {}
        for name, cols in self.feature_groups.items():
            bad = set(cols) - set(self.features)
            if bad:
                raise ValueError(f"feature group {name!r} has unknown features: {sorted(bad)}")
        self.feature_descriptions = feature_descriptions or {}
        self.value_labels = value_labels or {}
        self.llm = llm
        self.random_state = random_state
        X = self.data[self.features]
        self.background = X.sample(min(background_size, len(X)), random_state=random_state)
        self._predictions: np.ndarray | None = None

    # ---------- helpers ----------

    def _row(self, row) -> pd.DataFrame:
        if isinstance(row, pd.Series):
            row = row.to_frame().T
        if not isinstance(row, pd.DataFrame) or len(row) != 1:
            raise ValueError("pass exactly one row: a one-row DataFrame or a Series")
        missing = set(self.features) - set(row.columns)
        if missing:
            raise ValueError(f"row is missing features: {sorted(missing)}")
        x = row[self.features].copy()
        for c in self.features:
            try:
                x[c] = x[c].astype(self.data[c].dtype)
            except (TypeError, ValueError):
                pass
        return x

    def _all_predictions(self) -> np.ndarray:
        if self._predictions is None:
            self._predictions = tools.predict(self.model, self.data[self.features])
        return self._predictions

    def _label(self, feature: str, v):
        v = _j(v)
        lab = self.value_labels.get(feature)
        if callable(lab):
            return lab(v)
        if isinstance(lab, dict):
            return lab.get(v, lab.get(str(v), v))
        return v

    def _value(self, factor: str, row: pd.Series):
        if factor in self.feature_groups:
            return {c: self._label(c, row[c]) for c in self.feature_groups[factor]}
        return self._label(factor, row[factor])

    def _group(self, raw: dict) -> dict:
        return tools.group_matrix(pd.DataFrame([raw]), self.feature_groups).iloc[0].to_dict()

    def _drivers(self, raw: dict, row: pd.Series, before: pd.Series | None = None,
                 total_change: float | None = None, top_k: int = 8) -> list[dict]:
        """raw: per-input contributions. Grouped here, keeping a group's parts when they disagree."""
        contributions = self._group(raw)
        scale = max([1.0] + [abs(v) for v in contributions.values()])
        items = [(k, v) for k, v in contributions.items() if abs(v) > 1e-9 * scale]
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        out = []
        for name, effect in items[:top_k]:
            d = {"factor": name, "effect": _r(effect)}
            if before is None:
                d["value"] = self._value(name, row)
            else:
                d["before"], d["after"] = self._value(name, before), self._value(name, row)
            if total_change:
                d["share_of_change_pct"] = _r(100 * effect / total_change)
            if name in self.feature_groups:
                parts = {c: raw[c] for c in self.feature_groups[name] if abs(raw[c]) > 1e-9 * scale}
                total = sum(abs(v) for v in parts.values())
                material = [v for v in parts.values() if abs(v) >= 0.1 * total]
                if len(material) >= 2:  # show what's inside when more than one part really matters
                    d["parts"] = {c: _r(v) for c, v in sorted(parts.items(), key=lambda kv: -abs(kv[1]))}
            out.append(d)
        rest = items[top_k:]
        if rest:
            effect = sum(v for _, v in rest)
            d = {"factor": f"all other factors ({len(rest)})", "effect": _r(effect)}
            if total_change:
                d["share_of_change_pct"] = _r(100 * effect / total_change)
            out.append(d)
        return out

    def _cast(self, column: str, value):
        col = self.data[column]
        if pd.api.types.is_bool_dtype(col):
            return str(value).strip().lower() in ("true", "1", "yes")
        if pd.api.types.is_numeric_dtype(col):
            return float(value)
        return value.strip() if isinstance(value, str) else value

    def _mask(self, filter) -> np.ndarray:
        """filter: None, {"col": value or [values]}, or [{"column", "op", "value"}]."""
        mask = np.ones(len(self.data), dtype=bool)
        if not filter:
            return mask
        if isinstance(filter, dict):
            filter = [{"column": c, "op": "in" if isinstance(v, (list, tuple, set)) else "==", "value": v}
                      for c, v in filter.items()]
        for cond in filter:
            column, op, value = cond["column"], cond.get("op", "=="), cond["value"]
            if column not in self.data.columns:
                raise ValueError(f"unknown column {column!r}")
            col = self.data[column]
            if op == "in":
                values = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
                mask &= col.isin([self._cast(column, v) for v in values]).to_numpy()
            elif op in _OPS:
                mask &= _OPS[op](col, self._cast(column, value)).to_numpy(dtype=bool)
            else:
                raise ValueError(f"unknown operator {op!r}")
        if not mask.any():
            raise ValueError("no rows match this filter")
        return mask

    @staticmethod
    def _describe(filter) -> str:
        if not filter:
            return "all data"
        if isinstance(filter, dict):
            return " and ".join(f"{c} = {v}" for c, v in filter.items())
        return " and ".join(f"{c['column']} {c.get('op', '==')} {c['value']}" for c in filter)

    # ---------- tools: each returns exact, JSON-ready numbers ----------

    def explain_prediction(self, row) -> dict:
        """Why is this prediction what it is, relative to the average prediction?"""
        x = self._row(row)
        e = tools.explain_prediction(self.model, x, background=self.background)
        return Result("explain_prediction", {
            "target": self.target_name,
            "prediction": _r(e.prediction),
            "average_prediction": _r(e.base_value),
            "difference_from_average": _r(e.prediction - e.base_value),
            "drivers": self._drivers(e.contributions, x.iloc[0]),
            "how_to_read": "average_prediction + sum of driver effects = prediction",
        })

    def explain_change(self, before, after) -> dict:
        """Why did the prediction move from `before` to `after`?"""
        a, b = self._row(before), self._row(after)
        e = tools.explain_change(self.model, a, b)
        change = e.prediction - e.base_value
        return Result("explain_change", {
            "target": self.target_name,
            "before": _r(e.base_value),
            "after": _r(e.prediction),
            "change": _r(change),
            "pct_change": _r(100 * change / e.base_value) if e.base_value else None,
            "drivers": self._drivers(e.contributions, b.iloc[0], before=a.iloc[0], total_change=change or None),
            "how_to_read": "before + sum of driver effects = after; inputs that didn't change have zero effect",
        })

    def what_if(self, row, changes) -> dict:
        """Re-predict with inputs changed. changes: {"feature": new_value} or
        [{"feature", "operation": "set" | "add" | "multiply", "value"}]."""
        x = self._row(row)
        if isinstance(changes, dict):
            changes = [{"feature": f, "operation": "set", "value": v} for f, v in changes.items()]
        new_values, applied, warnings = {}, [], []
        for ch in changes:
            f, op, v = ch["feature"], ch.get("operation", "set"), ch["value"]
            if f not in self.features:
                raise ValueError(f"unknown feature {f!r}")
            old = x[f].iloc[0]
            if op == "set":
                new = self._cast(f, v)
            elif op == "add":
                new = old + float(v)
            elif op == "multiply":
                new = old * float(v)
            else:
                raise ValueError(f"unknown operation {op!r}")
            new_values[f] = new
            applied.append({"feature": f, "from": self._label(f, old), "to": self._label(f, new)})
            col = self.data[f]
            if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
                lo, hi = col.min(), col.max()
                if new < lo or new > hi:
                    warnings.append(f"{f} = {_j(new)} is outside the range seen in the data "
                                    f"({_j(lo)} to {_j(hi)}), so the model is extrapolating; treat with caution.")
        _, e = tools.what_if(self.model, x, new_values)
        change = e.prediction - e.base_value
        new_row = x.copy()
        for f, v in new_values.items():
            new_row[f] = v
        return Result("what_if", {
            "target": self.target_name,
            "changes_applied": applied,
            "prediction_before": _r(e.base_value),
            "prediction_after": _r(e.prediction),
            "change": _r(change),
            "pct_change": _r(100 * change / e.base_value) if e.base_value else None,
            "drivers": self._drivers(e.contributions, new_row.iloc[0], before=x.iloc[0],
                                     total_change=change or None),
            "warnings": warnings,
            "note": "This is how the model's prediction responds, not proof the change would cause this in reality.",
        })

    def prediction_context(self, row) -> dict:
        """Is this prediction high or low, and how unusual are its inputs?"""
        x = self._row(row)
        p = float(tools.predict(self.model, x)[0])
        preds = self._all_predictions()
        inputs = []
        for c in self.features[:30]:
            v = x[c].iloc[0]
            entry = {"input": c, "value": self._label(c, v)}
            col = self.data[c]
            if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
                entry["percentile_in_data"] = _r(100 * float((col <= v).mean()))
            if c in self.feature_descriptions:
                entry["description"] = self.feature_descriptions[c]
            inputs.append(entry)
        return Result("prediction_context", {
            "target": self.target_name,
            "prediction": _r(p),
            "percentile_among_all_predictions": _r(100 * float((preds <= p).mean())),
            "average_prediction": _r(preds.mean()),
            "median_prediction": _r(np.median(preds)),
            "typical_range_10th_to_90th_percentile": [_r(np.percentile(preds, 10)), _r(np.percentile(preds, 90))],
            "inputs": inputs,
        })

    def top_drivers(self, filter=None, top_k: int = 10, max_rows: int = 300) -> dict:
        """Which factors matter most, overall or in a segment."""
        mask = self._mask(filter)
        seg = self.data.loc[mask, self.features]
        sample = seg.sample(min(len(seg), max_rows), random_state=self.random_state)
        values, _ = tools.shap_values(self.model, sample, background=self.background)
        grouped = tools.group_matrix(pd.DataFrame(values, columns=self.features), self.feature_groups)
        impact, direction = grouped.abs().mean(), grouped.mean()
        order = impact.sort_values(ascending=False).index[:top_k]
        preds = self._all_predictions()
        return Result("top_drivers", {
            "target": self.target_name,
            "segment": self._describe(filter),
            "rows_in_segment": int(mask.sum()),
            "rows_analyzed": len(sample),
            "average_prediction_in_segment": _r(preds[mask].mean()),
            "average_prediction_overall": _r(preds.mean()),
            "drivers": [{"factor": f, "typical_impact_size": _r(impact[f]), "average_effect": _r(direction[f])}
                        for f in order],
            "how_to_read": ("typical_impact_size: how much this factor typically moves a single prediction, in "
                            "either direction. average_effect: whether it pushes predictions in this segment up "
                            "(+) or down (-) on average, relative to the overall average prediction."),
        })

    def model_errors(self, filter=None, by: str | None = None) -> dict:
        """How accurate is the model, overall or in a segment, optionally broken down by a column."""
        if self.y is None:
            return Result("model_errors", {"error": "No actual outcomes were provided, so accuracy can't be measured."})
        mask = self._mask(filter)
        y, p = self.y[mask], self._all_predictions()[mask]
        out = {"target": self.target_name, "segment": self._describe(filter), **_metrics(y, p)}
        if by:
            col = self.data.loc[mask, by]
            if col.nunique() > 50:
                raise ValueError(f"{by!r} has too many distinct values to break down by")
            rows = [{by: self._label(by, k), **_metrics(y[(col == k).to_numpy()], p[(col == k).to_numpy()])}
                    for k in col.unique()]
            out["breakdown_worst_first"] = sorted(rows, key=lambda r: r["average_error"], reverse=True)[:15]
        out["how_to_read"] = ("average_error: typical size of a miss. bias: positive means the model "
                              "over-predicts on average, negative means it under-predicts.")
        return Result("model_errors", out)

    def flag_unanswerable(self, category: str, reason: str = "") -> dict:
        """Mark a question the model can't honestly answer."""
        return Result("flag_unanswerable", {
            "status": "cannot_answer_reliably",
            "category": category,
            "reason": reason,
            "explanation": _UNANSWERABLE.get(category, _UNANSWERABLE["other"]),
            "can_answer_instead": ["why a prediction is what it is", "why a prediction changed",
                                   "how the prediction responds to changed inputs (what-if)",
                                   "which factors matter most", "how accurate the model is"],
        })

    # ---------- questions ----------

    def _filter_columns(self) -> list[str]:
        return [str(c) for c in self.data.columns]

    def _by_columns(self) -> list[str]:
        return [str(c) for c in self.data.columns if self.data[c].nunique() <= 50]

    def ask(self, question: str, row=None, compare_to=None, rows: dict | None = None,
            max_steps: int = 8) -> Answer:
        """Answer a question in plain language.

        row: the prediction the question is about. compare_to: an earlier/other prediction, for
        "why did it change?" questions. rows: any other named rows, e.g. {"monday_8am": ..., "saturday_2pm": ...}.
        """
        named = {}
        if row is not None:
            named["row"] = row
        if compare_to is not None:
            named["compare_to"] = compare_to
        named.update(rows or {})
        named = {k: self._row(v) for k, v in named.items()}

        def resolve(name):
            if name not in named:
                raise ValueError(f"unknown row {name!r}; available: {list(named)}")
            return named[name]

        dispatch = {
            "explain_prediction": lambda row: self.explain_prediction(resolve(row)),
            "prediction_context": lambda row: self.prediction_context(resolve(row)),
            "explain_change": lambda before, after: self.explain_change(resolve(before), resolve(after)),
            "what_if": lambda row, changes: self.what_if(resolve(row), changes),
            "top_drivers": lambda filter=None: self.top_drivers(filter),
            "model_errors": lambda filter=None, by=None: self.model_errors(filter, by),
            "flag_unanswerable": lambda category, reason="": self.flag_unanswerable(category, reason),
        }
        has_actuals = self.y is not None
        system = system_prompt(self.target_name, self.features, self.feature_descriptions, self.feature_groups,
                               list(named), self._filter_columns(), has_actuals)
        specs = build_tool_specs(list(named), self.features, self._filter_columns(), self._by_columns(), has_actuals)
        text, calls = run_agent(get_llm(self.llm), system, question, specs, dispatch, max_steps)
        sources = [question] + [str(c["result"]) for c in calls]
        return Answer(question, text, calls, unsupported_numbers(text, sources))

    def answer_all(self, questions: list[str], row=None, compare_to=None, rows: dict | None = None,
                   title: str = "Model explanation report") -> Report:
        """Answer a list of questions; returns a Report (.to_markdown(), .to_dataframe(), .to_json())."""
        return Report([self.ask(q, row=row, compare_to=compare_to, rows=rows) for q in questions], title=title)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    out = {
        "rows": int(len(y)),
        "average_actual": _r(y.mean()),
        "average_error": _r(np.abs(err).mean()),
        "root_mean_squared_error": _r(np.sqrt((err ** 2).mean())),
        "bias": _r(err.mean()),
    }
    if y.mean():
        out["average_error_pct_of_average_actual"] = _r(100 * np.abs(err).mean() / abs(y.mean()))
    return out
