import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from explainkit import explain_change, explain_prediction, what_if


def test_change_adds_up_and_unchanged_inputs_get_zero(demand):
    model, X, _ = demand
    before, after = X.iloc[[0]].copy(), X.iloc[[0]].copy()
    after["month"], after["rain"] = 11, 1
    e = explain_change(model, before, after)
    assert e.base_value + sum(e.contributions.values()) == pytest.approx(e.prediction, abs=1e-6)
    for col in ["wind", "temp", "station"]:
        assert e.contributions[col] == pytest.approx(0, abs=1e-9)


def test_change_with_groups(demand):
    model, X, _ = demand
    e = explain_change(model, X.iloc[[1]], X.iloc[[2]], feature_groups={"weather": ["temp", "wind"]})
    assert "weather" in e.contributions and "temp" not in e.contributions
    assert e.base_value + sum(e.contributions.values()) == pytest.approx(e.prediction, abs=1e-6)


def test_what_if_matches_model(demand):
    model, X, _ = demand
    row = X.iloc[[3]]
    new, e = what_if(model, row, {"wind": int(row["wind"].iloc[0]) + 7})
    assert e.prediction == pytest.approx(model.predict(new)[0])
    assert e.prediction - e.base_value == pytest.approx(e.contributions["wind"], abs=1e-6)


def test_what_if_keeps_decimals_on_integer_columns(demand):
    model, X, _ = demand
    new, _ = what_if(model, X.iloc[[3]], {"wind": 7.5})
    assert new["wind"].iloc[0] == 7.5


def test_what_if_rejects_unknown_category(demand):
    model, X, _ = demand
    with pytest.raises(ValueError, match="not a known category"):
        what_if(model, X.iloc[[3]], {"station": "Airport"})


def test_non_tree_model_is_supported():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = 3 * X["a"] - 2 * X["b"] + X["c"] * X["d"]
    model = Ridge().fit(X, y)
    exp = explain_prediction(model, X.iloc[[0]], background=X.iloc[:50])
    assert exp.base_value + sum(exp.contributions.values()) == pytest.approx(exp.prediction, abs=1e-6)
    after = X.iloc[[0]].copy()
    after["a"] += 1
    ch = explain_change(model, X.iloc[[0]], after)
    assert ch.contributions["a"] == pytest.approx(ch.prediction - ch.base_value, abs=1e-6)
    assert ch.contributions["b"] == pytest.approx(0, abs=1e-9)


def test_plain_function_as_model():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 0.0]})
    f = lambda df: 2 * df["a"].to_numpy() + df["b"].to_numpy()
    ch = explain_change(f, X.iloc[[0]], X.iloc[[1]])
    assert ch.contributions == pytest.approx({"a": 2.0, "b": 1.0})
