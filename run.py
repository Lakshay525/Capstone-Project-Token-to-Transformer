"""Batch runner: a folder of customer audio files in, replies (text + speech) and sources out.

    python run.py <audio_dir> [--out outputs] [--stt-language en-IN]

For every audio file it runs  Gnani STT -> agent -> Gnani TTS  and writes, per file:
    <name>.json        transcript, reply text, decision, retrieved chunks (id + text), tool calls, timings
    <name>.reply.txt   the reply text
    <name>.reply.wav   the spoken reply (Gnani TTS)
plus results.jsonl and results.csv for the whole batch.

Files are independent calls by default. Name files `<conversation>__<turn>.<ext>` (for example
`call7__01.wav`, `call7__02.wav`) to play them as consecutive turns of one conversation.
`.txt` files are treated as already-transcribed text (no STT) so the agent can be tested without audio.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agent import SupportAgent  # noqa: E402
from src.config import STT_LANGUAGE  # noqa: E402
from src.kb import Retriever  # noqa: E402
from src.llm import LLM, LLMError  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
GROUP_RE = re.compile(r"^(?P<conv>.+?)__(?P<turn>\d+)$")
APOLOGY = "Sorry, I'm having trouble right now. Let me connect you with a human colleague."


def collect(folder: Path):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT | {".txt"})
    groups: dict = {}
    for p in files:
        m = GROUP_RE.match(p.stem)
        key, turn = (m.group("conv"), int(m.group("turn"))) if m else (p.stem, 0)
        groups.setdefault(key, []).append((turn, p))
    return [(k, [p for _, p in sorted(v)]) for k, v in groups.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--stt-language", default=STT_LANGUAGE, help="Gnani language_code, e.g. en-IN, hi-IN, gu-IN")
    ap.add_argument("--mode", default="full", choices=["baseline", "rag", "full"])
    ap.add_argument("--no-tts", action="store_true", help="skip speech synthesis (text results only)")
    args = ap.parse_args()

    if not args.audio_dir.is_dir():
        sys.exit(f"{args.audio_dir} is not a folder")
    args.out.mkdir(parents=True, exist_ok=True)

    retriever, llm = Retriever(), LLM()
    voice = None
    needs_voice = not args.no_tts or any(p.suffix.lower() in AUDIO_EXT for p in args.audio_dir.iterdir())
    if needs_voice:
        from src.voice import Voice
        voice = Voice(stt_language=args.stt_language)

    rows = []
    for conv, paths in collect(args.audio_dir):
        agent = SupportAgent(llm, retriever, args.mode, channel="voice")
        for path in paths:
            t0 = time.time()
            rec = {"file": path.name, "conversation": conv, "transcript": "", "stt_language": args.stt_language,
                   "reply": "", "decision": "", "chunks": [], "chunk_text": {}, "tool_calls": [], "ticket_id": None,
                   "errors": []}
            if path.suffix.lower() == ".txt":
                rec["transcript"] = path.read_text(encoding="utf-8").strip()
            else:
                heard = voice.listen(path, args.stt_language)
                rec["transcript"], rec["stt_language"] = heard["transcript"], heard["language"]
                if heard["error"]:
                    rec["errors"].append("stt: " + heard["error"])
            t1 = time.time()

            if not rec["transcript"].strip():  # unclear / silent audio: recover, never guess
                rec["reply"], rec["decision"] = (voice.recovery_text(rec["stt_language"]) if voice else
                                                 "Sorry, I didn't catch that. Could you say it again?"), "clarify"
            else:
                try:
                    out = agent.respond(rec["transcript"])
                    rec.update(reply=out.text, decision=out.decision, chunks=out.chunks,
                               tool_calls=out.tool_calls, ticket_id=out.ticket_id)
                    rec["chunk_text"] = {cid: retriever.get(cid).render() for cid in out.chunks if retriever.get(cid)}
                except LLMError as exc:
                    rec["errors"].append(f"llm: {exc}")
                    rec["reply"], rec["decision"] = APOLOGY, "answer"
            t2 = time.time()

            stem = args.out / path.stem
            (args.out / f"{path.stem}.reply.txt").write_text(rec["reply"], encoding="utf-8")
            if voice and not args.no_tts:
                sp = voice.speak(rec["reply"], args.out / f"{path.stem}.reply.wav")
                rec["reply_audio"], rec["spoken_text"] = sp["audio"], sp["spoken_text"]
                if sp["error"]:
                    rec["errors"].append("tts: " + sp["error"])
            rec["timing_s"] = {"stt": round(t1 - t0, 2), "agent": round(t2 - t1, 2), "total": round(time.time() - t0, 2)}
            stem.with_suffix(".json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(rec)
            print(f"[{path.name}] {rec['decision']:9} heard: {rec['transcript'][:60]!r} -> {rec['reply'][:70]!r}")

    with open(args.out / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "transcript", "reply", "decision", "chunks", "ticket_id", "reply_audio", "errors"])
        for r in rows:
            w.writerow([r["file"], r["transcript"], r["reply"], r["decision"], " ; ".join(r["chunks"]),
                        r["ticket_id"] or "", r.get("reply_audio") or "", " | ".join(r["errors"])])
    print(f"\n{len(rows)} files processed -> {args.out}/")


if __name__ == "__main__":
    main()
