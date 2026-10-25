"""Answer and Report objects: the text plus everything needed to audit it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Answer:
    question: str
    text: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"tool", "arguments", "result"}]
    unsupported_numbers: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True if every number in the answer was found in tool results or the question."""
        return not self.unsupported_numbers

    def __str__(self):
        return self.text

    def _repr_markdown_(self):
        return self.to_markdown()

    def to_markdown(self) -> str:
        tools = ", ".join(dict.fromkeys(c["tool"] for c in self.tool_calls)) or "none"
        md = f"{self.text}\n\n<sub>Computed with: {tools}"
        if self.unsupported_numbers:
            md += f". ⚠ Numbers not found in tool results: {', '.join(self.unsupported_numbers)}"
        return md + "</sub>"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    answers: list[Answer]
    title: str = "Model explanation report"

    def __iter__(self):
        return iter(self.answers)

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, i):
        return self.answers[i]

    def to_markdown(self, path: str | Path | None = None) -> str:
        parts = [f"# {self.title}"]
        for i, a in enumerate(self.answers, 1):
            parts.append(f"## {i}. {a.question}\n\n{a.to_markdown()}")
        md = "\n\n".join(parts) + "\n"
        if path:
            Path(path).write_text(md)
        return md

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"question": a.question, "answer": a.text,
             "tools": [c["tool"] for c in a.tool_calls], "grounded": a.grounded,
             "unsupported_numbers": a.unsupported_numbers}
            for a in self.answers
        ])

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps({"title": self.title, "answers": [a.to_dict() for a in self.answers]}, indent=2, default=str)
        if path:
            Path(path).write_text(s)
        return s
