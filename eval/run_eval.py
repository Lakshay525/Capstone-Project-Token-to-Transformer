"""Practice-set evaluation: Step 1 (prompt only) vs Step 2 (RAG) vs Step 3 (conversation agent).

    python eval/run_eval.py questions --mode baseline     # Step 1
    python eval/run_eval.py questions --mode rag          # Step 2
    python eval/run_eval.py questions --mode full         # Step 3 agent on single questions
    python eval/run_eval.py conversations                 # Step 3, 23 conversations / 96 turns
    python eval/run_eval.py questions --mode rag --asr    # query with the ASR-style text

Needs an LLM key in .env (see .env.example). Results go to eval/results/.
A reply counts as correct when must_include passes, must_not_include is clean and the decision is acceptable.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from checks import (evidence_recall, must_include_ok, must_not_ok, parse_tool_calls,  # noqa: E402
                    tool_calls_ok)
from src.agent import SupportAgent  # noqa: E402
from src.config import PRACTICE_DIR  # noqa: E402
from src.kb import Retriever  # noqa: E402
from src.llm import LLM  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def read(name):
    with open(PRACTICE_DIR / name, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _frac(rs, key):
    return round(sum(bool(r[key]) for r in rs) / max(len(rs), 1), 3)


def run_questions(args, llm, retriever):
    rows = read("questions.csv")
    unanswerable_ids = {r["id"] for r in rows if r["answerable"] == "True" and not r["gold_evidence"]}

    def one(row):
        agent = SupportAgent(llm, retriever, args.mode, channel="voice" if args.asr else "chat")
        q = row["asr_text"] if args.asr else row["question"]
        try:
            rep = agent.respond(q)
        except Exception as exc:
            return {"id": row["id"], "error": str(exc), "correct": False}
        acceptable = [d.strip() for d in (row["acceptable_decisions"] or row["expected_decision"]).split(";")]
        inc, notinc = must_include_ok(rep.text, row["must_include"]), must_not_ok(rep.text, row["must_not_include"])
        dec = rep.decision in acceptable
        return {"id": row["id"], "question": q, "reply": rep.text, "decision": rep.decision,
                "expected": row["expected_decision"], "must_include": inc, "must_not": notinc,
                "decision_ok": dec, "correct": inc and notinc and dec,
                "chunks": rep.chunks, "evidence_recall": evidence_recall(row["gold_evidence"], rep.chunks),
                "no_document_answer": row["id"] in unanswerable_ids, "language": row["language"]}

    done = [0]

    def tracked(row):
        r = one(row)
        done[0] += 1
        print(f"[{done[0]}/{len(rows)}] {r['id']} {'OK' if r.get('correct') else 'x'}", flush=True)
        return r

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(tracked, rows))
    ok = [r for r in res if "error" not in r]
    gold_rows = [r for r in ok if not r["no_document_answer"]]
    summary = {
        "mode": args.mode, "asr": args.asr, "n": len(res), "errors": len(res) - len(ok),
        "correct": sum(r["correct"] for r in res), "accuracy": _frac(res, "correct"),
        "must_include_rate": _frac(ok, "must_include"),
        "must_not_clean_rate": _frac(ok, "must_not"),
        "decision_rate": _frac(ok, "decision_ok"),
        "mean_evidence_recall": round(sum(r["evidence_recall"] for r in gold_rows) / max(len(gold_rows), 1), 3),
        "dont_know_questions_correct": "%d/%d" % (
            sum(r["correct"] for r in ok if r["no_document_answer"]), sum(r["no_document_answer"] for r in ok)),
    }
    return summary, res


def run_conversations(args, llm, retriever):
    rows = read("conversations.csv")
    convs: dict = {}
    for r in rows:
        convs.setdefault(r["conversation_id"], []).append(r)

    def one(item):
        cid, turns = item
        agent = SupportAgent(llm, retriever, "full", channel="voice" if args.asr else "chat")
        out = []
        for t in sorted(turns, key=lambda r: int(r["turn"])):
            try:
                rep = agent.respond(t["customer_message"])
            except Exception as exc:
                out.append({"conversation": cid, "turn": int(t["turn"]), "error": str(exc), "correct": False})
                continue
            exp_calls = parse_tool_calls(t["expected_tool_calls"])
            wants_ticket = any(n == "create_ticket" for n, _ in exp_calls)
            made_ticket = any(c["name"] == "create_ticket" for c in rep.tool_calls)
            calls_ok = (tool_calls_ok([c for c in exp_calls if c[0] != "create_ticket"], rep.tool_calls)
                        and made_ticket == wants_ticket)
            inc, notinc = must_include_ok(rep.text, t["must_include"]), must_not_ok(rep.text, t["must_not_include"])
            dec = rep.decision == t["expected_decision"]
            out.append({"conversation": cid, "turn": int(t["turn"]), "message": t["customer_message"],
                        "reply": rep.text, "decision": rep.decision, "expected": t["expected_decision"],
                        "must_include": inc, "must_not": notinc, "decision_ok": dec, "tools_ok": calls_ok,
                        "tool_calls": rep.tool_calls, "chunks": rep.chunks,
                        "evidence_recall": evidence_recall(t["gold_evidence"], rep.chunks),
                        "correct": inc and notinc and dec and calls_ok})
        return out

    done = [0]

    def tracked(item):
        out = one(item)
        done[0] += 1
        print(f"[{done[0]}/{len(convs)}] {item[0]} {sum(t['correct'] for t in out)}/{len(out)} turns OK", flush=True)
        return out

    with ThreadPoolExecutor(args.workers) as ex:
        res = [t for turns in ex.map(tracked, convs.items()) for t in turns]
    ok = [r for r in res if "error" not in r]
    per_conv: dict = {}
    for r in res:
        per_conv.setdefault(r["conversation"], []).append(r["correct"])
    summary = {
        "mode": "full", "asr": args.asr, "turns": len(res), "errors": len(res) - len(ok),
        "turn_accuracy": _frac(res, "correct"),
        "conversations_fully_correct": f"{sum(all(v) for v in per_conv.values())}/{len(per_conv)}",
        "decision_rate": _frac(ok, "decision_ok"), "tool_call_rate": _frac(ok, "tools_ok"),
        "must_include_rate": _frac(ok, "must_include"), "must_not_clean_rate": _frac(ok, "must_not"),
        "mean_evidence_recall": round(sum(r["evidence_recall"] for r in ok) / max(len(ok), 1), 3),
    }
    return summary, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["questions", "conversations"])
    ap.add_argument("--mode", default="full", choices=["baseline", "rag", "full"])
    ap.add_argument("--asr", action="store_true", help="use the ASR-style text instead of the written question")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    llm, retriever = LLM(), Retriever()
    fn = run_questions if args.target == "questions" else run_conversations
    summary, res = fn(args, llm, retriever)
    tag = f"{args.target}_{args.mode if args.target == 'questions' else 'full'}{'_asr' if args.asr else ''}"
    (OUT / f"{tag}.json").write_text(json.dumps({"summary": summary, "results": res}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps(summary, indent=2))
    bad = [r for r in res if not r.get("correct")]
    print(f"\n{len(bad)} not fully correct (details in eval/results/{tag}.json):")
    for r in bad[:40]:
        why = "error" if "error" in r else ",".join(k for k in ("must_include", "must_not", "decision_ok", "tools_ok")
                                                    if k in r and not r[k])
        print(f"  {r.get('id') or str(r['conversation']) + '/' + str(r['turn'])}: {why}")


if __name__ == "__main__":
    main()
