"""Build report.pdf (max 4 pages) from eval/results/*.json and the retrieval eval.

    python eval/make_report.py

Numbers come only from files produced by eval/run_eval.py; anything not run yet is shown as "not run".
Edit the NARRATIVE section below to add what went wrong / what you would fix after you have run it.
"""
import json
import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"

NARRATIVE = {
    "went_wrong": ("Fill in after running the evaluation: list the failing practice cases from "
                   "eval/results/*.json (the runner prints them) and the cause of each."),
    "fix_next": ("Fill in: what you would change first (retrieval misses, Hindi phrasing, voice latency, ...)."),
    "bonus": ("Hybrid BM25 + optional multilingual dense retrieval; code-enforced verification and date arithmetic; "
              "spoken-digit normalisation for ASR text; scripted-LLM offline tests; web demo with browser recording."),
}


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8"))["summary"] if p.exists() else None


def row(label, s, keys):
    if not s:
        return f"<tr><td>{label}</td>" + "<td colspan='%d'><i>not run</i></td>" % len(keys) + "</tr>"
    return f"<tr><td>{label}</td>" + "".join(f"<td>{s.get(k, '-')}</td>" for k in keys) + "</tr>"


def main():
    retr = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "eval" / "retrieval_eval.py"), "--k", "6"],
                          capture_output=True, text=True, encoding="utf-8").stdout.strip().replace("\n", "<br>")
    keys = ["accuracy", "must_include_rate", "decision_rate", "mean_evidence_recall", "dont_know_questions_correct"]
    q1, q2, q3 = load("questions_baseline"), load("questions_rag"), load("questions_full")
    qa, cv = load("questions_rag_asr"), load("conversations_full")
    ckeys = ["turn_accuracy", "conversations_fully_correct", "decision_rate", "tool_call_rate", "must_include_rate"]
    html = f"""
<h1>ZentroMart VoiceOps agent: report</h1>
<h2>What I built</h2>
<p><b>Step 1.</b> One LLM call with a system prompt (role, limits, AI-assistant introduction). No documents.<br>
<b>Step 2.</b> RAG over kb/*.md (one chunk per [SECTION-ID], status and effective_from kept) and the product catalog;
BM25 with a Hinglish-to-English query bridge, SKU matching, catalog-to-policy joins and KB-00.1 precedence weighting.
Answers use only retrieved excerpts, say "I don't have that information" otherwise, and log the chunk IDs used.<br>
<b>Step 3.</b> A tool-calling conversation agent: history, order/ticket tools, code-enforced identity check with one retry,
code-computed business-day and return-window dates, escalation flags, and checks that a handover is only claimed after
create_ticket returned created=True.<br>
<b>Step 4.</b> Gnani STT (ITN on, Hindi fallback) and TTS (timbre-v2.5), spoken-digit normalisation, ear-friendly
reply text, batch runner (run.py) and a browser demo (app.py).</p>
<h2>Step 1 vs Step 2 (practice questions, 52)</h2>
<table border=1 cellpadding=3 cellspacing=0>
<tr><th>run</th><th>accuracy</th><th>facts included</th><th>decision ok</th><th>evidence recall</th><th>"don't know" cases</th></tr>
{row("Step 1: prompt only", q1, keys)}{row("Step 2: RAG", q2, keys)}{row("Step 3 agent, single questions", q3, keys)}
{row("Step 2 on ASR-style text", qa, keys)}
</table>
<p>Accuracy = must_include satisfied, no must_not_include phrase, acceptable decision.</p>
<h2>Retrieval only (no LLM, text queries, k=6)</h2><p><tt>{retr}</tt></p>
<h2>Step 3 conversations (23 conversations, 96 turns)</h2>
<table border=1 cellpadding=3 cellspacing=0>
<tr><th>run</th><th>turn accuracy</th><th>conversations fully correct</th><th>decision ok</th><th>tool calls ok</th><th>facts included</th></tr>
{row("Step 3 agent", cv, ckeys)}
</table>
<h2>What went wrong</h2><p>{NARRATIVE['went_wrong']}</p>
<h2>What I would fix</h2><p>{NARRATIVE['fix_next']}</p>
<h2>Bonus work</h2><p>{NARRATIVE['bonus']}</p>
"""
    story = fitz.Story(html=html, user_css="body{font-family:sans-serif;font-size:10pt} h1{font-size:16pt} "
                                           "h2{font-size:12pt;margin-bottom:2pt} th,td{font-size:9pt}")
    writer = fitz.DocumentWriter(str(ROOT / "report.pdf"))
    more, page = 1, 0
    while more and page < 4:
        dev = writer.begin_page(fitz.paper_rect("a4"))
        more, _ = story.place(fitz.Rect(40, 40, 555, 800))
        story.draw(dev)
        writer.end_page()
        page += 1
    writer.close()
    print(f"report.pdf written ({page} page(s))")


if __name__ == "__main__":
    main()
