"""LLM connection. Any provider LiteLLM supports, or your own object with a .chat() method."""

from __future__ import annotations

import json
from typing import Protocol


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """Return {"content": str | None, "tool_calls": [{"id": str, "name": str, "arguments": dict}]}."""


class LiteLLM:
    """Any LiteLLM model string, e.g. "gemini/gemini-2.5-flash", "anthropic/claude-sonnet-5-5", "gpt-4.1".

    API keys come from the usual environment variables (GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = {"temperature": 0, **kwargs}

    def chat(self, messages, tools):
        import litellm

        resp = litellm.completion(model=self.model, messages=messages, tools=tools or None, **self.kwargs)
        msg = resp.choices[0].message
        calls = [
            {"id": tc.id or f"call_{i}", "name": tc.function.name,
             "arguments": json.loads(tc.function.arguments or "{}")}
            for i, tc in enumerate(msg.tool_calls or [])
        ]
        return {"content": msg.content, "tool_calls": calls}

    def __repr__(self):
        return f"LiteLLM({self.model!r})"


def get_llm(llm) -> LLM:
    if isinstance(llm, str):
        return LiteLLM(llm)
    if hasattr(llm, "chat"):
        return llm
    raise TypeError("llm must be a LiteLLM model string or an object with a .chat(messages, tools) method")
