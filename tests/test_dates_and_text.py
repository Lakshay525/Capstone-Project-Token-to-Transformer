import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import dates  # noqa: E402
from src.textutil import normalize_spoken, speakable  # noqa: E402


def test_refund_eta_matches_practice_answers():
    # values taken from practice/questions.csv must_include
    assert dates.refund_eta("2026-10-17", "card")["latest_date"] == "2026-10-27"
    assert dates.refund_eta("2026-10-17", "upi")["latest_date"] == "2026-10-22"
    assert dates.refund_eta("2026-10-12", "upi")["latest_date"] == "2026-10-15"
    assert dates.refund_eta("2026-10-14", "netbanking")["latest_date"] == "2026-10-21"
    assert dates.refund_eta("2026-10-17", "netbanking")["latest_date"] == "2026-10-24"
    assert dates.refund_eta("2026-10-12", "credit card")["latest_date"] == "2026-10-21"


def test_wallet_is_instant_and_overdue_flag():
    assert dates.refund_eta("2026-10-17", "wallet")["instant"] is True
    r = dates.refund_eta("2026-07-20", "upi")
    assert r["latest_date"] == "2026-07-23" and r["escalate_overdue"] is True
    assert dates.refund_eta("2026-10-17", "upi")["escalate_overdue"] is False


def test_return_windows():
    w = dates.return_window("fashion", "2026-10-12", "2026-09-30")
    assert w["window_days"] == 15 and w["deadline"] == "2026-10-27"
    w = dates.return_window("fashion", "2026-10-14", "2026-10-06")
    assert w["window_days"] == 30 and w["deadline"] == "2026-11-13"
    assert dates.return_window("electronics", "2026-07-20")["window_days"] == 10
    assert dates.return_window("electronics", "2026-09-01")["window_days"] == 7
    assert dates.return_window("innerwear")["returnable"] is False


def test_business_days_skip_sundays_and_holidays():
    assert dates.add_business_days(date(2026, 10, 17), 1) == date(2026, 10, 20)  # Sun + holiday skipped


def test_spoken_digits_and_speakable():
    assert normalize_spoken("order number eight nine seven eight two eight three seven") == "order number 89782837"
    assert normalize_spoken("pincode double four zero zero one zero") == "pincode 440010"
    assert normalize_spoken("I have one coupon") == "I have one coupon"
    assert normalize_spoken("आख़िरी चार अंक ४५३३ हैं") == "आख़िरी चार अंक 4533 हैं"
    s = speakable("**Done.** Order 89782837 refund of ₹1,499 arrives by 2026-10-27.")
    assert "8978 2837" in s and "1499 rupees" in s and "27th of October 2026" in s and "*" not in s
