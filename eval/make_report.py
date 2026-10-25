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
    "went_wrong": (
        "<b>Evaluation limits (please read).</b> I used Groq's free tier (gpt-oss-120b, 200,000 tokens/day). When it ran "
        "out, calls rotated to gpt-oss-20b and qwen3.8-27b, and --resume merged runs made on different days, so the "
        "Step 2 and Step 3 rows mix models and two prompt versions. The Step 1 row used gpt-oss-120b only. 3 Step 1 "
        "and 3 full-agent questions errored (rate limit / malformed tool call) and count as wrong. The quota ran out "
        "before I could run the 23-conversation evaluation, so Step 3 conversation handling is <b>not scored</b>, and my "
        "end-to-end voice batch got only quota-error apologies from the LLM (Gnani STT/TTS themselves worked: my clips "
        "were transcribed and replies were synthesised).<br>"
        "<b>Step 1.</b> Without documents the model guesses: 14% of required facts present, and 0 of 5 questions the "
        "documents do not cover were answered with 'I don't know'. Step 2 got 5 of 5.<br>"
        "<b>Step 2 failures (14).</b> 6 wrong refund deadlines: the LLM miscounts business days (Sundays and the 19 "
        "October holiday). 3 warranty answers omitted the 12-month period. Hinglish questions answered in Devanagari. "
        "'Clarify' chosen after already answering. Swimsuit/briefs not stated as 'cannot be returned'.<br>"
        "<b>Full agent on single questions (14 wrong, 3 errors).</b> Date maths fixed by the refund_eta tool, but new "
        "problems: wrong decision label on escalation/safety questions (PQ0023, PQ0030, PQ0031), and 2 questions failed "
        "because the model sent null for an optional tool argument."),
    "fix_next": (
        "Fixed in code after those runs but <b>not re-scored</b> (quota): null tool arguments are tolerated; the "
        "'I am an AI assistant' opening is enforced in code; empty replies are retried; retry-after is honoured. Next "
        "(in order): re-run all evaluations on one model with a larger quota; tighten escalation prompts for "
        "safety/unauthorised-activity cases; add multilingual embeddings for Hindi; shorten prompts to cut tokens per call."),
    "bonus": ("Hybrid BM25 + optional multilingual dense retrieval; query bridge for Hinglish; catalog-to-policy joins; "
              "code-enforced identity check with one retry and ticket honesty; spoken-digit normalisation for ASR text; "
              "offline tests with a scripted LLM (9 passing); resumable evaluation runner; browser demo with microphone recording."),
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
    keys = ["correct", "accuracy", "errors", "must_include_rate", "decision_rate", "mean_evidence_recall", "dont_know_questions_correct"]
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
<tr><th>run</th><th>correct /52</th><th>accuracy</th><th>errors (rate limit)</th><th>facts included</th><th>decision ok</th><th>evidence recall</th><th>"I don't know" cases</th></tr>
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
