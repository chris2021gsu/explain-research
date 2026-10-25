# explainkit

Ask questions about your model's predictions in plain language, and get answers whose numbers are computed, not made up.

```python
from explainkit import Explainer

ex = Explainer(model, data, target_name="bike rentals per hour", y="count",
               feature_groups={"day type": ["holiday", "workingday", "weekday"]},
               llm="gemini/gemini-2.5-flash")   # or any LiteLLM model: "anthropic/claude-sonnet-5-5", "gpt-4.1", ...

print(ex.explain_change(week_before, thanksgiving))
# The prediction went from 425 to 62.8 bike rentals per hour: down by 362 (−85.2%).
# What caused it:
#   • day type (not a holiday → holiday, working day → day off): −304, 84% of the change
#   • weather conditions (misty → clear, 12°C → 10°C, ...): −57.9, 16% of the change
# Inputs that didn't change contributed nothing.

ex.ask("Why was Thanksgiving morning so quiet?", row=thanksgiving, compare_to=week_before)   # any question, via an LLM
```

SHAP is the engine; explainkit is the product. An LLM picks the right tool for each question, the tools compute
exact numbers, and the LLM only phrases them. Every answer records the tool calls behind it and flags any
number that didn't come from a tool.

## What you can ask

| Question | Tool |
|---|---|
| Why is this prediction X? | `explain_prediction` |
| Is X high or low? How unusual are the inputs? | `prediction_context` |
| Why did it change from A to B? | `explain_change`: inputs that didn't change get exactly zero credit |
| What if an input changes? | `what_if`: warns when the model is extrapolating |
| What matters most, overall or for a segment? | `top_drivers` |
| How accurate is the model, and where is it worst? | `model_errors` (needs `y`) |
| Will doing X *cause* Y? | `flag_unanswerable`: says honestly that the model can't answer causal questions |

All tools also work without an LLM: `ex.explain_change(last_week, this_week)` returns the numbers directly.

## A list of questions, as a report

```python
report = ex.answer_all(["Why is the forecast 525?", "What changed since last week?", "How accurate is it?"],
                       row=this_week, compare_to=last_week)
report.to_markdown("report.md")   # also .to_dataframe(), .to_json()
```

## Works with

- **Any model**: anything with `.predict(DataFrame)`, or a plain function. Tree models (LightGBM, XGBoost,
  sklearn forests) use fast exact TreeSHAP; everything else uses model-agnostic SHAP.
- **Any LLM** supported by [LiteLLM](https://docs.litellm.ai/docs/providers): set the matching key
  (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, ...) in `.env`. Or pass your own object with `.chat(messages, tools)`.

## Install

```bash
pip install "git+https://github.com/chris2021gsu/explain-research.git"
```

Requires Python 3.10+. To use `ex.ask()`, set the API key for your LLM (e.g. `GEMINI_API_KEY`) in your environment.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your LLM key
pytest
python examples/bike_demand.py
```

## License

MIT. See [LICENSE](LICENSE).
