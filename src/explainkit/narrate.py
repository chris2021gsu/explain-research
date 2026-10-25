"""Plain-English rendering of tool results, so the library speaks in words even without an LLM.

Each tool returns a Result: a dict of exact numbers (what the LLM and the grounding check use) that
prints as sentences.
"""

from __future__ import annotations


def _n(x) -> str:
    """Readable number: 1,234 / 52.3 / 0.62."""
    if x is None:
        return "n/a"
    x = float(x)
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:,.1f}".rstrip("0").rstrip(".")
    return f"{x:.2g}"


def _move(x) -> str:
    return f"up by {_n(abs(x))}" if x >= 0 else f"down by {_n(abs(x))}"


def _signed(x) -> str:
    return f"+{_n(x)}" if x >= 0 else f"−{_n(abs(x))}"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _value(v, name: str | None = None) -> str:
    """A value in words. Bare numbers get their input's name ("temp 12.3"); labels speak for themselves."""
    if isinstance(v, dict):
        return ", ".join(_value(x, k) for k, x in list(v.items())[:3]) + (", ..." if len(v) > 3 else "")
    if _is_num(v):
        return f"{name} {_n(v)}" if name else _n(v)
    return str(v)


def _change_value(before, after, name: str | None = None) -> str:
    if isinstance(before, dict):
        moved = [_change_value(before[k], after[k], k) for k in before if before[k] != after[k]]
        return ", ".join(moved[:3]) + (", ..." if len(moved) > 3 else "")
    prefix = f"{name} " if name and _is_num(before) else ""
    return f"{prefix}{_value(before)} → {_value(after)}"


def _parts(d: dict) -> str:
    """For a group whose parts pull in opposite directions: say what's inside."""
    parts = d.get("parts")
    if not parts:
        return ""
    fmt = lambda items: ", ".join(f"{k} {_signed(v)}" for k, v in items[:3])
    total = sum(abs(v) for v in parts.values())
    parts = {k: v for k, v in parts.items() if abs(v) >= 0.1 * total}  # skip parts too small to matter
    ups = [(k, v) for k, v in parts.items() if v > 0]
    downs = [(k, v) for k, v in parts.items() if v < 0]
    if not ups or not downs:
        return f"\n      (made up of: {fmt(list(parts.items()))})"
    if d["effect"] < 0:
        return f"\n      ({fmt(ups)} pushed it up, but {fmt(downs)} pulled it down more)"
    return f"\n      ({fmt(downs)} pulled it down, but {fmt(ups)} pushed it up more)"


def _share(pct) -> str:
    if pct is None:
        return ""
    if pct < 0:
        return ", working against the change"
    if pct > 100:
        return ", more than the whole change (others offset part of it)"
    return f", {pct:.0f}% of the change"


def _drivers(drivers: list[dict], change: bool) -> list[str]:
    lines = []
    for d in drivers:
        if d["factor"].startswith("all other factors"):
            lines.append(f"  • {d['factor']}: {_signed(d['effect'])}")
            continue
        val = _change_value(d["before"], d["after"]) if change else _value(d["value"])
        share = _share(d.get("share_of_change_pct")) if change else ""
        lines.append(f"  • {d['factor']} ({val}): {_signed(d['effect'])}{share}{_parts(d)}")
    return lines


def explain_prediction(r: dict) -> str:
    t = r["target"]
    diff = r["difference_from_average"]
    head = (f"The model predicts {_n(r['prediction'])} {t}, {_n(abs(diff))} "
            f"{'above' if diff >= 0 else 'below'} the average of {_n(r['average_prediction'])}.")
    return "\n".join([head, "Why:"] + _drivers(r["drivers"], change=False))


def explain_change(r: dict) -> str:
    t, ch = r["target"], r["change"]
    pct = f" ({_signed(r['pct_change'])}%)" if r.get("pct_change") is not None else ""
    head = f"The prediction went from {_n(r['before'])} to {_n(r['after'])} {t}: {_move(ch)}{pct}."
    if not r["drivers"]:
        return head + "\nNo inputs changed, so nothing contributed."
    return "\n".join([head, "What caused it:"] + _drivers(r["drivers"], change=True)
                     + ["Inputs that didn't change contributed nothing."])


def what_if(r: dict) -> str:
    t = r["target"]
    changes = " and ".join(_change_value(c["from"], c["to"], c["feature"]) for c in r["changes_applied"])
    pct = f" ({_signed(r['pct_change'])}%)" if r.get("pct_change") is not None else ""
    lines = [f"Changing {changes} moves the prediction from {_n(r['prediction_before'])} to "
             f"{_n(r['prediction_after'])} {t}: {_move(r['change'])}{pct}."]
    if len(r["drivers"]) > 1 or any(d.get("parts") for d in r["drivers"]):
        lines += _drivers(r["drivers"], change=True)
    lines += [f"⚠ {w}" for w in r["warnings"]]
    lines.append("This is how the model responds, not proof the change would cause this in reality.")
    return "\n".join(lines)


def prediction_context(r: dict) -> str:
    lo, hi = r["typical_range_10th_to_90th_percentile"]
    pct = r["percentile_among_all_predictions"]
    level = ("unusually high" if pct >= 90 else "unusually low" if pct <= 10 else
             "above typical" if pct >= 60 else "below typical" if pct <= 40 else "typical")
    return (f"{_n(r['prediction'])} {r['target']} is {level}: higher than {pct:.0f}% of all predictions. "
            f"Most predictions fall between {_n(lo)} and {_n(hi)} (average {_n(r['average_prediction'])}).")


def top_drivers(r: dict) -> str:
    lines = [f"What matters most for {r['target']} ({r['segment']}, {r['rows_in_segment']:,} rows):"]
    drivers = [d for d in r["drivers"] if d["typical_impact_size"]]
    for i, d in enumerate(drivers[:6], 1):
        direction = (f", and pushes this segment {_move(d['average_effect'])} on average"
                     if r["segment"] != "all data" and abs(d["average_effect"]) >= 0.2 * d["typical_impact_size"] else "")
        lines.append(f"  {i}. {d['factor']}: typically moves a prediction by {_n(d['typical_impact_size'])}{direction}")
    if r["segment"] != "all data":
        lines.append(f"Average prediction here: {_n(r['average_prediction_in_segment'])} "
                     f"vs {_n(r['average_prediction_overall'])} overall.")
    return "\n".join(lines)


def model_errors(r: dict) -> str:
    if "error" in r:
        return r["error"]
    bias = r["bias"]
    tendency = (f"It tends to under-predict, by {_n(abs(bias))} on average." if bias < -0.1 * r["average_error"] else
                f"It tends to over-predict, by {_n(bias)} on average." if bias > 0.1 * r["average_error"] else
                "It doesn't systematically over- or under-predict.")
    pct = r.get("average_error_pct_of_average_actual")
    lines = [f"On {r['segment']} ({r['rows']:,} rows), the model is typically off by {_n(r['average_error'])} "
             f"{r['target']}" + (f", {_n(pct)}% of the average actual ({_n(r['average_actual'])})." if pct else "."),
             tendency]
    rows = r.get("breakdown_worst_first")
    if rows:
        by = next(k for k in rows[0] if k not in ("rows", "average_actual", "average_error",
                                                   "root_mean_squared_error", "bias",
                                                   "average_error_pct_of_average_actual"))
        lines.append(f"Worst {by}: {rows[0][by]} (off by {_n(rows[0]['average_error'])}). "
                     f"Best: {rows[-1][by]} (off by {_n(rows[-1]['average_error'])}).")
    return "\n".join(lines)


def flag_unanswerable(r: dict) -> str:
    return f"I can't answer that reliably. {r['explanation']}"


NARRATORS = {f.__name__: f for f in [explain_prediction, explain_change, what_if, prediction_context,
                                     top_drivers, model_errors, flag_unanswerable]}


class Result(dict):
    """Exact numbers (as a dict) that print as plain English."""

    def __init__(self, kind: str, data: dict):
        super().__init__(data)
        self.kind = kind

    @property
    def text(self) -> str:
        return NARRATORS[self.kind](self)

    def __str__(self):
        return self.text

    def _repr_pretty_(self, p, cycle):  # notebooks / IPython show words, not a dict
        p.text(self.text)
