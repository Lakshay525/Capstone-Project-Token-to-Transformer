"""Text chat with the agent in your terminal (no audio, handy for debugging).

    python chat.py [--mode full|rag|baseline] [--show-sources]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.agent import SupportAgent  # noqa: E402
from src.kb import Retriever  # noqa: E402
from src.llm import LLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["baseline", "rag", "full"])
    ap.add_argument("--show-sources", action="store_true")
    ap.add_argument("--voice-style", action="store_true", help="short spoken-style replies")
    args = ap.parse_args()
    agent = SupportAgent(LLM(), Retriever(), args.mode, channel="voice" if args.voice_style else "chat")
    print("ZentroMart support agent. Ctrl-C to quit.\n")
    while True:
        try:
            text = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        r = agent.respond(text)
        print(f"agent > {r.text}")
        if args.show_sources:
            print(f"        [{r.decision}] sources: {', '.join(r.chunks)}"
                  + (f" | tools: {[c['name'] for c in r.tool_calls]}" if r.tool_calls else ""))


if __name__ == "__main__":
    main()
