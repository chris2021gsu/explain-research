"""explainkit on a real forecasting problem: hourly bike rentals (UCI Bike Sharing, Washington DC, 2011-2012).

Run:  python examples/bike_demand.py
With GEMINI_API_KEY in .env, it also asks questions in plain language. Without a key it shows the exact
numbers the LLM would be given.
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

from bike_setup import ex, last_week, this_week  # noqa: E402

if os.getenv("GEMINI_API_KEY"):
    report = ex.answer_all(
        [
            "Why is the forecast for this hour what it is?",
            "Why is it different from the same hour last week?",
            "What if it were 5 degrees warmer?",
            "What matters most for rentals overall?",
            "How accurate is the model, and when is it worst?",
            "Will lowering prices increase rentals?",
        ],
        row=this_week, compare_to=last_week, title="Bike rentals: Friday 5pm",
    )
    print(report.to_markdown("results/bike_demand_report.md"))
else:
    print("No GEMINI_API_KEY found, so showing the exact numbers the LLM would phrase.\n")
    show = lambda title, r: print(f"--- {title}\n{json.dumps(r, indent=2)}\n")
    show("explain_change(last week -> this week)", ex.explain_change(last_week, this_week))
    show("what_if(+5 degrees)", ex.what_if(this_week, [{"feature": "temp", "operation": "add", "value": 5},
                                                      {"feature": "feel_temp", "operation": "add", "value": 5}]))
    show("model_errors(by season)", ex.model_errors(by="season"))
