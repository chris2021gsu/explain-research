"""LLM tool-calling loop: the LLM picks tools, tools compute numbers, the LLM phrases the answer."""

from __future__ import annotations

import json

SYSTEM_PROMPT = """\
You explain a machine-learning model's predictions to a business user who does not know machine learning.

The model predicts: {target}
Model inputs: {features}
{groups}{rows}Data columns usable in filters: {filter_columns}
{has_actuals}

Rules:
1. Every number in your answer must come from a tool result in this conversation. Never estimate, \
calculate or invent numbers yourself; if you need a number, call a tool. You may round sensibly \
(4,987 -> "about 5,000").
2. Call as many tools as you need to answer completely. Typical choices:
   - "Why is the prediction X?" -> explain_prediction, plus prediction_context to say whether X is high or low.
   - "Why did it change / why is A different from B?" -> explain_change.
   - "What if ...?" -> what_if.
   - "What matters most?" -> top_drivers.
   - "How accurate is it / where is it wrong?" -> model_errors.
3. The model learned patterns from historical data, not causes. If the user asks whether doing something \
in the real world will cause an outcome ("will raising price reduce demand?"), call flag_unanswerable with \
category "causal". You may also run what_if, but present it only as how the model's prediction responds, \
never as what will happen. If a question needs information the model and data don't have, call \
flag_unanswerable and say plainly what you can't answer.
4. Write for a business reader: start with a one-sentence direct answer, then the main drivers with their \
size in the target's units, then any caveat. Use the input names as written, explained in plain words. \
Never mention SHAP, tools, or technical terms.
5. Driver effects add up exactly: starting value + effects = prediction. Group small effects as \
"other factors".
6. If tool results include warnings (for example inputs outside the range seen in the data), mention them.
"""


def system_prompt(target: str, features: list[str], descriptions: dict[str, str],
                  groups: dict[str, list[str]], row_names: list[str], filter_columns: list[str],
                  has_actuals: bool) -> str:
    feats = ", ".join(f"{f} ({descriptions[f]})" if f in descriptions else f for f in features[:80])
    if len(features) > 80:
        feats += f", ... ({len(features)} total)"
    group_text = "".join(f"  - {g}: {', '.join(cols)}\n" for g, cols in groups.items())
    group_text = f"Factor groups (each is reported as one factor):\n{group_text}" if groups else ""
    if row_names:
        row_text = "Rows the user provided, referred to by name: " + ", ".join(row_names) + "."
        if "row" in row_names and "compare_to" in row_names:
            row_text += (' "row" is the prediction the user is asking about and "compare_to" is the earlier or '
                         'other prediction to compare against: to explain a change, use before="compare_to", '
                         'after="row".')
        row_text += "\n"
    else:
        row_text = ("No specific rows were provided, so you can only answer questions about the model overall. "
                    "If asked about a specific prediction, say which prediction the user should pass in.\n")
    return SYSTEM_PROMPT.format(
        target=target, features=feats, groups=group_text, rows=row_text,
        filter_columns=", ".join(filter_columns[:80]),
        has_actuals=("Actual outcomes are available, so accuracy questions can be answered." if has_actuals
                     else "Actual outcomes are NOT available, so accuracy questions cannot be answered."),
    )


def _fn(name: str, description: str, properties: dict | None = None, required: tuple = ()) -> dict:
    params: dict = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = list(required)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


def build_tool_specs(row_names: list[str], features: list[str], filter_columns: list[str],
                     by_columns: list[str], has_actuals: bool) -> list[dict]:
    specs = []
    filt = {
        "type": "array",
        "description": "Optional conditions selecting a segment of the data (all must hold). Omit for all data.",
        "items": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "enum": filter_columns},
                "op": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<=", "in"]},
                "value": {"type": "string", "description": "Value to compare to; for 'in', comma-separated values."},
            },
            "required": ["column", "op", "value"],
        },
    }
    if row_names:
        row = {"type": "string", "enum": row_names, "description": "Name of a row the user provided."}
        specs += [
            _fn("explain_prediction",
                "Why a prediction is what it is: the average prediction, and how much each factor pushes this "
                "prediction above or below it.", {"row": row}, ("row",)),
            _fn("prediction_context",
                "Whether a prediction is high or low compared with all predictions, and where each input value "
                "sits compared with the data.", {"row": row}, ("row",)),
            _fn("what_if",
                "Re-run the model with some inputs changed and explain how the prediction moves.",
                {"row": row, "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {"type": "string", "enum": features},
                            "operation": {"type": "string", "enum": ["set", "add", "multiply"]},
                            "value": {"type": "string", "description": "A number, or a category name for 'set'."},
                        },
                        "required": ["feature", "operation", "value"],
                    },
                }}, ("row", "changes")),
        ]
        if len(row_names) >= 2:
            specs.append(_fn(
                "explain_change",
                "Why the prediction differs between two rows (e.g. last week vs this week): the change, and how "
                "much of it each changed factor caused. Factors that didn't change get zero.",
                {"before": row, "after": row}, ("before", "after")))
    specs.append(_fn(
        "top_drivers",
        "Which factors matter most to the model's predictions overall or in a segment, and in which direction.",
        {"filter": filt}))
    if has_actuals:
        by = {"by": {"type": "string", "enum": by_columns,
                     "description": "Optional column to break accuracy down by."}} if by_columns else {}
        specs.append(_fn(
            "model_errors",
            "How accurate the model is, overall or in a segment, optionally broken down by a column.",
            {"filter": filt, **by}))
    specs.append(_fn(
        "flag_unanswerable",
        "Use when the question can't be answered honestly from this model: causal questions, future or "
        "external events, or information not in the data.",
        {"category": {"type": "string", "enum": ["causal", "future_or_external_events", "not_in_data", "other"]},
         "reason": {"type": "string", "description": "One sentence on why."}},
        ("category", "reason")))
    return specs


def run_agent(llm, system: str, question: str, tools: list[dict], dispatch: dict,
              max_steps: int = 8) -> tuple[str, list[dict]]:
    """Returns (answer text, [{"tool", "arguments", "result"}])."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    calls: list[dict] = []
    for _ in range(max_steps):
        reply = llm.chat(messages, tools)
        if not reply["tool_calls"]:
            return (reply["content"] or "").strip(), calls
        messages.append({
            "role": "assistant",
            "content": reply["content"],
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                           for c in reply["tool_calls"]],
        })
        for c in reply["tool_calls"]:
            try:
                result = dispatch[c["name"]](**c["arguments"])
            except Exception as e:  # report to the LLM so it can recover or explain
                result = {"error": f"{type(e).__name__}: {e}"}
            calls.append({"tool": c["name"], "arguments": c["arguments"], "result": result})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, default=str)})
    final = messages + [{"role": "user", "content": "Write your final answer now, using only the tool results above."}]
    reply = llm.chat(final, None)
    return (reply["content"] or "").strip(), calls
