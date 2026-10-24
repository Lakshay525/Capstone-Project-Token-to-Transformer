"""Scoring rules from DATA_DICTIONARY.md for practice/questions.csv and conversations.csv.

must_include : every ` ; ` fact is required; ` / ` separates equivalent alternatives; ignore case;
               accept equivalent dates, spoken numbers and native-script digits.
must_not_include : a forbidden claim; any listed phrase (or paraphrase) means a wrong answer.
"""
import re
from datetime import date

DEV = str.maketrans("०१२३४५६७८९", "0123456789")
MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}
HI_MONTHS = {"जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7,
             "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12,
             "दिसम्बर": 12}
NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUM_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "twenty-four": 24, "forty-eight": 48,
                  "seventy-nine": 79, "forty-nine": 49, "twenty-five": 25})
HI_NUM = {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9,
          "दस": 10, "पंद्रह": 15, "तीस": 30, "अड़तालीस": 48}


def norm(text: str) -> str:
    t = (text or "").translate(DEV).lower().replace("’", "'").replace("‑", "-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)            # 10,000 -> 10000
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)        # 24th -> 24
    for w, n in sorted(NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{w}\b", str(n), t)
    for w, n in HI_NUM.items():
        t = t.replace(w, str(n)) if len(w) > 2 else t
    return t


def dates_in(text: str) -> set:
    """All dates mentioned in a reply, as ISO strings (year defaults to 2026)."""
    t = norm(text)
    found = set()
    for y, m, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", t):
        found.add(f"{y}-{m}-{d}")
    names = {**MONTHS, **HI_MONTHS}
    pat = "|".join(sorted(names, key=len, reverse=True))
    for d, mo, y in re.findall(rf"\b(\d{{1,2}})\s*(?:of\s+)?({pat})\s*,?\s*(20\d\d)?", t):
        found.add(_iso(int(d), names[mo], int(y) if y else 2026))
    for mo, d, y in re.findall(rf"\b({pat})\s+(\d{{1,2}})\s*,?\s*(20\d\d)?", t):
        found.add(_iso(int(d), names[mo], int(y) if y else 2026))
    return {f for f in found if f}


def _iso(d, m, y):
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _present(alt: str, reply_norm: str, reply_dates: set) -> bool:
    alt = alt.strip()
    if not alt:
        return True
    if re.fullmatch(r"20\d\d-\d\d-\d\d", alt):
        return alt in reply_dates
    a = norm(alt)
    if re.fullmatch(r"\d+", a):  # whole-number fact: don't match inside a longer number
        return re.search(rf"(?<![\d.]){a}(?![\d]|\.\d)", reply_norm) is not None
    if len(a) <= 3 and re.fullmatch(r"[a-z]+", a):  # short words ("ai", "no"): whole word only
        return re.search(rf"\b{a}\b", reply_norm) is not None
    if a in reply_norm:
        return True
    # tolerate "₹49" vs "49"
    return a.lstrip("₹") in reply_norm


def must_include_ok(reply: str, spec: str) -> bool:
    rn, rd = norm(reply), dates_in(reply)
    for fact in [f for f in (spec or "").split(" ; ") if f.strip()]:
        if not any(_present(alt, rn, rd) for alt in fact.split(" / ")):
            return False
    return True


def must_not_ok(reply: str, spec: str) -> bool:
    rn = norm(reply)
    for phrase in [p for p in (spec or "").split(" ; ") if p.strip()]:
        if any(norm(a) in rn for a in phrase.split(" / ") if a.strip()):
            return False
    return True


def evidence_recall(gold: str, used: list) -> float:
    g = [x.strip() for x in (gold or "").split(";") if x.strip()]
    if not g:
        return 1.0
    return sum(1 for x in g if x in used) / len(g)


def parse_tool_calls(spec: str) -> list:
    """'lookup_order(order_id=89782837, verification_value=4533)' -> [("lookup_order", {...})]"""
    out = []
    for name, body in re.findall(r"(\w+)\(([^)]*)\)", spec or ""):
        args = dict((k.strip(), v.strip()) for k, v in (kv.split("=", 1) for kv in body.split(",") if "=" in kv))
        out.append((name, args))
    return out


def tool_calls_ok(expected: list, actual: list) -> bool:
    """Every expected call appears (order id / verification digits compared as digits)."""
    digits = lambda v: re.sub(r"\D", "", str(v))  # noqa: E731
    for name, args in expected:
        match = False
        for a in actual:
            if a["name"] != name:
                continue
            if all(digits(a["args"].get(k, "")) == digits(v) for k, v in args.items()
                   if k in ("order_id", "verification_value") and (k != "verification_value" or v)):
                match = True
        if not match:
            return False
    return True
