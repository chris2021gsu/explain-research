import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def demand():
    """Small synthetic bike-rental dataset with a categorical column and an interaction."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "month": rng.integers(1, 13, n),
        "wind": rng.integers(5, 30, n),
        "temp": rng.uniform(10, 50, n).round(2),
        "rain": rng.integers(0, 2, n),
        "station": pd.Categorical(rng.choice(["Downtown", "Campus", "Riverside"], n)),
    })
    y = (100 + 8 * (X["month"] >= 10) - 1.5 * X["wind"] + 0.8 * X["temp"]
         - 15 * X["rain"] * (X["month"] >= 10) + 10 * (X["station"] == "Downtown") + rng.normal(0, 2, n))
    model = lgb.LGBMRegressor(n_estimators=80, random_state=0, verbose=-1).fit(X, y)
    return model, X, y.to_numpy()


class ScriptedLLM:
    """Fake LLM: replays scripted replies. A reply can be a function of the messages so far."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


def tool_call(name, **arguments):
    return {"content": None, "tool_calls": [{"id": f"call_{name}", "name": name, "arguments": arguments}]}


def last_tool_result(messages):
    return json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])
