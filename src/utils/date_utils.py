"""Shared date/time parsing utilities."""

from datetime import datetime


def parse_chat_time(time_str: str) -> datetime:
    """
    Parse chat time string, e.g. 2023/04/10 (Mon) 17:50.

    Used by both benchmark (LME) and memory storage for time-based sorting.
    """
    raw = (time_str or "").strip()
    if not raw:
        return datetime.max

    formats = [
        "%Y/%m/%d (%a) %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d (%A) %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass

    # Fallback: strip content in parentheses and retry
    compact = []
    in_bracket = False
    for ch in raw:
        if ch == "(":
            in_bracket = True
            continue
        if ch == ")":
            in_bracket = False
            continue
        if not in_bracket:
            compact.append(ch)
    cleaned = " ".join("".join(compact).split())

    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass

    return datetime.max
