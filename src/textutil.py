"""Text clean-up around the model: ASR-text normalisation in, speakable text out."""
import re
from datetime import date

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
    "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "ate": "8", "nine": "9",
    # Hindi, romanised
    "shunya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4", "paanch": "5",
    "panch": "5", "chhe": "6", "chheh": "6", "cheh": "6", "saat": "7", "sat": "7", "aath": "8",
    "nau": "9", "no": "9",
}
# words that count as digits only when part of a long run (avoids "do" / "to" / "for" in prose)
AMBIGUOUS = {"o", "to", "too", "for", "ate", "do", "no", "oh", "sat", "one", "ek"}
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
MULTIPLIER = {"double": 2, "triple": 3, "dabal": 2}


def normalize_spoken(text: str) -> str:
    """Turn ASR-style spelled-out digits into numerals so IDs and pincodes reach the tools.

    "order number eight nine seven eight two eight three seven" -> "order number 89782837".
    Runs shorter than 3 digit words are left alone, so ordinary words are not rewritten.
    """
    text = text.translate(DEVANAGARI_DIGITS)
    tokens = re.findall(r"\S+", text)
    out, run, run_orig = [], [], []

    def flush():
        confident = sum(1 for o in run_orig
                        if re.sub(r"[^\w]", "", o.lower()) not in AMBIGUOUS)
        if len(run) >= 3 and (confident >= 2 or all(o.isdigit() for o in run_orig)):
            out.append("".join(run))
        else:
            out.extend(run_orig)
        run.clear()
        run_orig.clear()

    i = 0
    while i < len(tokens):
        raw = tokens[i]
        word = re.sub(r"[^\w]", "", raw.lower())
        if word in MULTIPLIER and i + 1 < len(tokens):
            nxt = re.sub(r"[^\w]", "", tokens[i + 1].lower())
            if nxt in DIGIT_WORDS or nxt.isdigit():
                d = DIGIT_WORDS.get(nxt, nxt)
                run.append(d * MULTIPLIER[word]); run_orig.extend([raw, tokens[i + 1]])
                i += 2
                continue
        if word in DIGIT_WORDS:
            run.append(DIGIT_WORDS[word]); run_orig.append(raw)
        elif word.isdigit() and run:
            run.append(word); run_orig.append(raw)
        else:
            if run:
                flush()
            out.append(raw)
        i += 1
    if run:
        flush()
    return re.sub(r"\s+", " ", " ".join(out)).strip()


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _group_digits(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return " ".join(digits[i:i + 4] for i in range(0, len(digits), 4))


def speakable(text: str) -> str:
    """Prepare a reply for TTS (STY-3 and the Text normalization guide): no markdown, dates with
    the month name and year, amounts in words-friendly form, long IDs grouped in fours."""
    t = re.sub(r"[*_`#>]+", "", text)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)
    # ISO dates -> "the 24th of October 2026"
    def iso(m):
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"the {_ordinal(d.day)} of {MONTHS[d.month - 1]} {d.year}"
        except ValueError:
            return m.group(0)
    t = re.sub(r"\b(20\d\d)-(\d\d)-(\d\d)\b", iso, t)
    # "24 October 2026" / "24 October" -> "the 24th of October 2026"
    t = re.sub(r"\b(\d{1,2}) (%s)\b" % "|".join(MONTHS),
               lambda m: f"the {_ordinal(int(m.group(1)))} of {m.group(2)}", t)
    # amounts: Rs / ₹ / INR 1,500 -> "1500 rupees"
    t = re.sub(r"(?:₹|Rs\.?|INR)\s?([\d,]+(?:\.\d+)?)",
               lambda m: f"{m.group(1).replace(',', '')} rupees", t)
    # long numeric IDs (order numbers, RRNs): groups of four digits
    t = re.sub(r"(?<![\d,.])\d{7,}(?![\d,.])", _group_digits, t)
    # already-spaced order IDs like 8978 2837 stay as they are
    return re.sub(r"\s+", " ", t).strip()


def script_of(text: str) -> str:
    """'devanagari' if the text is mostly Devanagari, else 'latin'."""
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "devanagari" if dev > lat else "latin"
