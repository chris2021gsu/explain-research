import pytest

from conftest import ScriptedLLM, last_tool_result, tool_call
from explainkit import Explainer


@pytest.fixture
def ex(demand):
    model, X, y = demand
    return Explainer(model, X, "bike rentals per hour", y=y, feature_groups={"weather": ["temp", "wind"]},
                     llm=None)


def test_explain_prediction_adds_up(ex, demand):
    _, X, _ = demand
    r = ex.explain_prediction(X.iloc[0])
    total = r["average_prediction"] + sum(d["effect"] for d in r["drivers"])
    assert total == pytest.approx(r["prediction"], abs=0.05)
    assert {d["factor"] for d in r["drivers"]} <= {"weather", "month", "rain", "station"}


def test_explain_change_lists_only_changed_factors(ex, demand):
    _, X, _ = demand
    before, after = X.iloc[[0]].copy(), X.iloc[[0]].copy()
    after["rain"] = 1 - after["rain"]
    after["temp"] = after["temp"] + 10
    r = ex.explain_change(before, after)
    assert {d["factor"] for d in r["drivers"]} <= {"rain", "weather"}
    assert sum(d["share_of_change_pct"] for d in r["drivers"]) == pytest.approx(100, abs=0.5)


def test_what_if_warns_when_extrapolating(ex, demand):
    _, X, _ = demand
    r = ex.what_if(X.iloc[0], [{"feature": "wind", "operation": "add", "value": "100"}])
    assert r["warnings"] and "outside the range" in r["warnings"][0]


def test_segments_and_errors(ex):
    seg = ex.top_drivers([{"column": "station", "op": "==", "value": "Downtown"}])
    assert seg["rows_in_segment"] < 600 and seg["drivers"]
    err = ex.model_errors(by="station")
    assert err["rows"] == 600 and len(err["breakdown_worst_first"]) == 3
    assert ex.model_errors({"station": ["Downtown", "Campus"]})["rows"] < 600


def test_prediction_context(ex, demand):
    _, X, _ = demand
    r = ex.prediction_context(X.iloc[5])
    assert 0 <= r["percentile_among_all_predictions"] <= 100


def test_ask_runs_tools_and_grounds_answer(ex, demand):
    _, X, _ = demand
    before, after = X.iloc[[0]].copy(), X.iloc[[0]].copy()
    after["rain"] = 1 - after["rain"]

    def final(messages):
        r = last_tool_result(messages)
        return {"content": f"The forecast moved by {r['change']:,} to {r['after']:,}.", "tool_calls": []}

    ex.llm = ScriptedLLM([tool_call("explain_change", before="compare_to", after="row"), final])
    ans = ex.ask("Why did it change?", row=after, compare_to=before)
    assert [c["tool"] for c in ans.tool_calls] == ["explain_change"]
    assert ans.grounded, ans.unsupported_numbers
    tool_names = {t["function"]["name"] for t in ex.llm.calls[0]["tools"]}
    assert {"explain_change", "what_if", "model_errors", "flag_unanswerable"} <= tool_names


def test_ask_flags_invented_numbers(ex, demand):
    _, X, _ = demand
    ex.llm = ScriptedLLM([tool_call("explain_prediction", row="row"),
                          {"content": "It's 123,456 because of magic.", "tool_calls": []}])
    ans = ex.ask("Why?", row=X.iloc[0])
    assert not ans.grounded and "123,456" in ans.unsupported_numbers


def test_tool_errors_go_back_to_the_llm(ex, demand):
    _, X, _ = demand

    def final(messages):
        assert "error" in last_tool_result(messages)
        return {"content": "I couldn't find that row.", "tool_calls": []}

    ex.llm = ScriptedLLM([tool_call("explain_prediction", row="nope"), final])
    assert ex.ask("Why?", row=X.iloc[0]).text == "I couldn't find that row."


def test_answer_all_builds_report(ex, demand, tmp_path):
    _, X, _ = demand
    replies = []
    for _ in range(2):
        replies += [tool_call("flag_unanswerable", category="causal", reason="causal question"),
                    {"content": "The model shows patterns, not causes.", "tool_calls": []}]
    ex.llm = ScriptedLLM(replies)
    report = ex.answer_all(["Will adding bike lanes cause more rentals?", "Will cutting price help?"], row=X.iloc[0])
    md = report.to_markdown(tmp_path / "r.md")
    assert len(report) == 2 and "## 2." in md and (tmp_path / "r.md").exists()
    assert list(report.to_dataframe().columns)[:2] == ["question", "answer"]
