import lightgbm as lgb
import pytest
from sklearn.datasets import make_regression
import pandas as pd

from explainkit import explain_prediction


@pytest.fixture(scope="module")
def model_and_data():
    X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    model = lgb.LGBMRegressor(n_estimators=50, random_state=0, verbose=-1).fit(X, y)
    return model, X


def test_contributions_add_up_to_prediction(model_and_data):
    # Faithfulness check: base + sum(contributions) must equal the model output.
    model, X = model_and_data
    exp = explain_prediction(model, X.iloc[[0]])
    assert exp.base_value + sum(exp.contributions.values()) == pytest.approx(exp.prediction, abs=1e-6)


def test_grouping_preserves_total(model_and_data):
    model, X = model_and_data
    row = X.iloc[[3]]
    flat = explain_prediction(model, row)
    grouped = explain_prediction(model, row, feature_groups={"g": ["f0", "f1"]})
    assert set(grouped.contributions) == {"g", "f2", "f3", "f4"}
    assert sum(grouped.contributions.values()) == pytest.approx(sum(flat.contributions.values()))


def test_rejects_multiple_rows(model_and_data):
    model, X = model_and_data
    with pytest.raises(ValueError):
        explain_prediction(model, X.iloc[:2])
