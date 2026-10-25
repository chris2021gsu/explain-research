"""Check that every number in an answer appears in the tool results it was written from."""

from __future__ import annotations

import math
import re

_NUMBER = re.compile(r"(?<![\w.])[-+−]?\$?(\d[\d,]*(?:\.\d+)?)\s*(%|[kKmMbB]\b)?")
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


def _parse(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, significant digits written)."""
    out = []
    for m in _NUMBER.finditer(text):
        digits, suffix = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            value = float(digits) * _SUFFIX.get(suffix, 1)
        except ValueError:
            continue
        sig = digits.replace(".", "").lstrip("0")
        if "." not in digits:
            sig = sig.rstrip("0")
        out.append((m.group(0).strip(), value, max(len(sig), 1)))
    return out


def _round_sig(x: float, sig: int) -> float:
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def _matches(value: float, sig: int, sources: list[float]) -> bool:
    for s in sources:
        s = abs(s)
        if s == value or _round_sig(s, sig) == value:
            return True
        if s and abs(value - s) / s <= 0.01:
            return True
    return False


def unsupported_numbers(answer: str, sources: list[str], ignore_small_integers: bool = True) -> list[str]:
    """Numbers in `answer` that can't be traced to any of `sources` (tool results, the question).

    Sensible rounding is allowed ("about 5k" for 4,987). Integers 0-10 are ignored by default since
    they're mostly counts ("the top 3 factors").
    """
    known = [abs(v) for src in sources for _, v, _ in _parse(src)]
    bad = []
    for written, value, sig in _parse(answer):
        if ignore_small_integers and value.is_integer() and value <= 10:
            continue
        if not _matches(value, sig, known):
            bad.append(written)
    return bad
