"""Deterministic date rules. LLMs are bad at business-day arithmetic, so code does it.

Business days: Monday-Saturday, skipping data/holidays.csv. Return windows: calendar days.
"""
import csv
from datetime import date, datetime, timedelta

from .config import DATA_DIR, TODAY


def _holidays() -> set:
    with open(DATA_DIR / "holidays.csv", encoding="utf-8", newline="") as fh:
        return {date.fromisoformat(r["date"]) for r in csv.DictReader(fh)}


HOLIDAYS = _holidays()

# REF-2: (min, max) business days after initiation, per refund method.
REFUND_BUSINESS_DAYS = {
    "upi": (1, 3),
    "card": (5, 7), "credit card": (5, 7), "debit card": (5, 7),
    "netbanking": (3, 5), "net banking": (3, 5),
    "wallet": (0, 0), "cod_wallet": (0, 0),
    "cod_bank": (5, 7),
}

# RET-2 return windows in calendar days (None = not returnable).
RETURN_WINDOWS = {
    "electronics": 7, "fashion": 15, "home_kitchen": 10, "beauty_sealed": 7,
    "books_media": 10, "grocery": None, "innerwear": None, "digital": None,
}


def parse_date(value) -> date:
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value).strip()[:19]).date()


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in HOLIDAYS


def add_business_days(start: date, n: int) -> date:
    d = start
    while n > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            n -= 1
    return d


def business_days_between(start: date, end: date) -> int:
    """Business days elapsed after `start` up to and including `end`."""
    count, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if is_business_day(d):
            count += 1
    return count


def refund_eta(initiated_on, method: str) -> dict:
    """When a refund initiated on `initiated_on` via `method` should arrive (REF-2)."""
    key = method.strip().lower().replace("_", " ") if method else ""
    key = {"cod wallet": "cod_wallet", "cod bank": "cod_bank"}.get(key, key)
    if key not in REFUND_BUSINESS_DAYS:
        return {"error": f"unknown refund method '{method}'",
                "known_methods": sorted(REFUND_BUSINESS_DAYS)}
    start = parse_date(initiated_on)
    lo, hi = REFUND_BUSINESS_DAYS[key]
    earliest, latest = add_business_days(start, lo), add_business_days(start, hi)
    out = {"method": key, "initiated_on": start.isoformat(), "business_days": f"{lo} to {hi}",
           "earliest_date": earliest.isoformat(), "latest_date": latest.isoformat(),
           "instant": hi == 0}
    if hi == 0:
        out["note"] = "Wallet refunds are instant, within 2 hours."
    overdue = business_days_between(latest, TODAY) if TODAY > latest else 0
    out["business_days_past_maximum_as_of_today"] = overdue
    out["escalate_overdue"] = overdue > 3  # ESC-1 item 5
    return out


def return_window(return_category: str, delivered_on=None, placed_at=None) -> dict:
    """Return window + deadline for a category, applying RET-3 and RET-8 date rules."""
    cat = (return_category or "").strip().lower()
    if cat not in RETURN_WINDOWS:
        return {"error": f"unknown return_category '{return_category}'",
                "known_categories": sorted(RETURN_WINDOWS)}
    days = RETURN_WINDOWS[cat]
    if days is None:
        return {"return_category": cat, "returnable": False,
                "rule": "RET-2/RET-5: not returnable for change of mind; damaged/wrong/expired "
                        "items follow the damaged-wrong-missing policy (DWM-1: report within 48 hours)."}
    rule = "RET-2"
    delivered = parse_date(delivered_on) if delivered_on else None
    placed = parse_date(placed_at) if placed_at else None
    if cat == "electronics" and delivered and delivered < date(2026, 8, 1):
        days, rule = 10, "RET-3: electronics delivered before 1 August 2026 keep the 10-day window"
    if cat == "fashion" and placed and date(2026, 10, 1) <= placed <= date(2026, 10, 15):
        days, rule = 30, "RET-8: festive sale (order placed 1-15 October 2026) gives 30 days"
    out = {"return_category": cat, "returnable": True, "window_days": days, "rule": rule}
    if delivered:
        deadline = delivered + timedelta(days=days)
        out.update(delivered_on=delivered.isoformat(), deadline=deadline.isoformat(),
                   days_left=(deadline - TODAY).days, within_window=TODAY <= deadline)
    else:
        out["note"] = "No delivery date given, so no deadline can be computed yet."
    if cat == "fashion" and not placed:
        out["note_festive"] = ("Order date unknown: if placed 1-15 October 2026 the window is 30 "
                               "days (RET-8), otherwise 15.")
    return out
