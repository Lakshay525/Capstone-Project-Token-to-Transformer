# ZentroMart VoiceOps agent

A customer-support agent that answers from company documents, holds a conversation, and takes phone calls.
Capstone for *From Tokens To Transformers* x Gnani.ai VoiceOps 2026.

```
customer audio -> Gnani STT -> agent (LLM + RAG + tools) -> Gnani TTS -> spoken reply
```

Speech is **Gnani only** (STT and TTS). The LLM, retrieval and framework are mine: plain Python, a pure-Python
BM25 index, and either Anthropic or any OpenAI-compatible endpoint.

**Repository:** https://github.com/Lakshay525/Capstone-Project-Token-to-Transformer

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill in GNANI_API_KEY and an LLM key
```

The dataset is not in this repo. Extract `Capstone_Project_Dataset_FTTT_2026.zip` next to `run.py`
(or set `DATASET_DIR` in `.env`) and check it with `python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py`.

## Run command (batch, what the organisers use)

```bash
python run.py <folder_of_audio> --out outputs        # or: ./run.sh <folder_of_audio> outputs
```

For each audio file (`wav mp3 flac ogg m4a aac`, up to 60 s) it writes to `outputs/`:

| file | content |
| --- | --- |
| `<name>.json` | transcript, reply, decision, **retrieved chunks (id + text)**, tool calls, ticket id, timings |
| `<name>.reply.txt` | reply text |
| `<name>.reply.wav` | spoken reply from Gnani TTS |
| `results.jsonl`, `results.csv` | the whole batch |

* Each file is its own call. Name files `<call>__<turn>.wav` (`call7__01.wav`, `call7__02.wav`) to play them as
  consecutive turns of one conversation, so follow-ups and verification work across files.
* `--stt-language gu-IN` etc. sets the listening language (default `en-IN`, which also accepts Hindi-mixed English).
  If the first language returns nothing, the runner retries with Hindi. If nothing is heard, the agent says
  "Sorry, I didn't catch that" instead of guessing.
* `.txt` files are read as already-transcribed text (no STT), which is handy for testing without audio.
* A failure in one file (bad audio, LLM or TTS error) is logged in its JSON and never stops the batch.

## Other entry points

| command | purpose |
| --- | --- |
| `python app.py` | web demo at http://localhost:8000 (hold to talk / type), for the demo video |
| `python chat.py --show-sources` | text chat in the terminal |
| `python eval/retrieval_eval.py -v` | retrieval quality on the practice questions, **no API key needed** |
| `python eval/run_eval.py questions --mode baseline` | Step 1: LLM + system prompt only |
| `python eval/run_eval.py questions --mode rag` | Step 2: + retrieved documents |
| `python eval/run_eval.py conversations` | Step 3: 23 conversations / 96 turns, with tool-call checks |
| `python eval/make_report.py` | builds `report.pdf` from `eval/results/*.json` |
| `python -m pytest tests` | offline tests (verification gate, retry limit, ticket honesty, date maths, ...) |

## How it works

**Step 1, prompt only** (`mode=baseline`). One system prompt: role, limits, tone, "introduce yourself as the AI assistant".

**Step 2, RAG** (`src/kb.py`). `kb/*.md` is split into one chunk per `## [SECTION-ID]`, keeping doc type, status and
`effective_from`; every `products.csv` row is a chunk `products.csv:SKU`. Retrieval is BM25 plus:
* a Hinglish/Hindi to English bridge (customers say "wapas", "send it back", the documents say "return"/"reverse pickup");
* exact SKU matching, and a catalog-to-policy join (a product's `return_category` pulls in the right RET sections);
* the precedence rule from KB-00.1 as re-weighting (policy > agent guide > help > FAQ > marketing; superseded docs
  down-weighted and labelled), and KB-00.1 attached whenever an FAQ and a policy are both retrieved;
* optional dense retrieval (`EMBED_MODEL`) fused by reciprocal rank fusion.

The model must answer only from the excerpts, say "I don't have that information" and offer a human colleague
when they do not cover it, and every reply records which chunk IDs it used.

**Step 3, conversation** (`src/agent.py`). The model keeps the history and calls tools; code owns what must be exact:

| concern | where it is enforced |
| --- | --- |
| caller verification | `lookup_order` wrapper: ID first, then one proof (phone last 4 or pincode); **one retry**, then blocked. The mock tool already withholds data until proof; the wrapper also hides the pincode and never echoes secrets |
| business days / return windows | `src/dates.py` (Mon-Sat, holidays, RET-3 and RET-8 date rules); the model calls `refund_eta` / `return_window`. All refund dates in the practice set are reproduced by `tests/` |
| escalation triggers | `derived` flags on the verified order (repeat contact >= 2, refund > 3 business days overdue, damaged item > Rs 10,000) plus prompt rules for the rest of ESC-1 |
| "never claim what code did not do" | an `escalate` decision with no real `create_ticket` is sent back to the model once, and downgraded if it still has no ticket. Ticket summaries are scrubbed of verification values |
| cancellations | no cancel tool exists; the agent guides to My Orders |

**Step 4, voice** (`src/voice.py`, `run.py`, `app.py`). Gnani STT (`format="transcribe"` so numbers arrive as digits),
spelled-out digits are normalised for IDs ("eight nine seven eight ..."), replies are cleaned for the ear (dates as
"the 27th of October 2026", amounts as "49 rupees", order IDs in groups of four), and the TTS voice follows the
reply's script (Devanagari -> hi-IN, Hinglish -> Poorvi). The model is always named explicitly (`timbre-v2.5`).

## Layout

```
run.py  app.py  chat.py  run.sh        entry points
src/    config kb llm agent dates textutil voice
eval/   retrieval_eval run_eval checks make_report
tests/  offline tests (scripted fake LLM, no keys)
sample_outputs/                        example outputs (see its README)
```

## Evaluation status (honest summary)

52 practice questions, free-tier Groq. **Caveat:** when `gpt-oss-120b` hit its daily token cap, calls rotated to
`gpt-oss-20b` and `qwen3.8-27b`, and `--resume` merged runs from different days, so the Step 2 and full-agent rows mix
models and two prompt versions. The Step 1 row is `gpt-oss-120b` only.

| run | correct | errors | "I don't know" cases (5) |
| --- | --- | --- | --- |
| Step 1, prompt only | 5 / 52 | 3 (rate limit) | 0 / 5 |
| Step 2, RAG | **38 / 52** | 0 | 5 / 5 |
| Step 3 agent on single questions | 38 / 52 | 3 (rate limit, malformed tool call) | 4 / 5 |
| retrieval only, no LLM (k=6) | recall 0.97, all-gold-found 0.94 | n/a | n/a |

Not completed (free daily quota ran out): the 23-conversation evaluation and a clean end-to-end voice run (Gnani STT and
TTS themselves worked on recorded clips). Fixes made after these runs (null tool arguments tolerated, AI-assistant opening
enforced in code, empty-reply retry) are **not re-scored**. See `report.pdf` for the failure analysis. To finish:
`python eval/run_eval.py conversations --resume` and `python eval/run_eval.py questions --mode full --resume`.

## Limits and honesty notes

* No answer text is hardcoded. The lexicon and phrase bridges in `src/kb.py` only rewrite *queries*; answers come from
  retrieved documents and tool results.
* The mock identity check is a classroom simulation, not authentication.
* Hindi/Hinglish quality depends on the chosen LLM; dense multilingual retrieval (`EMBED_MODEL`) helps most there.
