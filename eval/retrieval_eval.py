"""Retrieval-only evaluation (no LLM, no API key needed).

For every practice question, search the KB with the question text (or the ASR text) and report
recall of gold_evidence section IDs at k, MRR and hard-negative leakage.

    python eval/retrieval_eval.py [--k 5] [--asr] [-v]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PRACTICE_DIR  # noqa: E402
from src.kb import Retriever  # noqa: E402


def split(v: str) -> list:
    return [x.strip() for x in v.split(";") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--asr", action="store_true", help="query with asr_text instead of question")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    r = Retriever()
    rows = list(csv.DictReader(open(PRACTICE_DIR / "questions.csv", encoding="utf-8")))
    n = hit_all = 0
    recall_sum = mrr_sum = leak = 0.0
    for row in rows:
        gold = split(row["gold_evidence"])
        if not gold:
            continue
        q = row["asr_text"] if args.asr else row["question"]
        got = [c.id for c, _ in r.search(q, k=args.k)]
        found = [g for g in gold if g in got]
        n += 1
        recall_sum += len(found) / len(gold)
        hit_all += len(found) == len(gold)
        ranks = [got.index(g) + 1 for g in gold if g in got]
        mrr_sum += 1 / min(ranks) if ranks else 0
        negs = split(row["hard_negatives"])
        leak += any(x in got for x in negs)
        if args.verbose and len(found) < len(gold):
            print(f"{row['id']}: missing {[g for g in gold if g not in got]} | got {got} | {q[:70]}")
    print(f"questions with gold evidence: {n}  (k={args.k}, {'ASR' if args.asr else 'text'} queries)")
    print(f"mean recall@k      : {recall_sum / n:.3f}")
    print(f"all-gold-found rate: {hit_all / n:.3f}")
    print(f"MRR (first gold)   : {mrr_sum / n:.3f}")
    print(f"hard-negative hit  : {leak / n:.3f}")


if __name__ == "__main__":
    main()
