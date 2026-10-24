# sample_outputs

* `retrieval_eval_text.txt`, `retrieval_eval_asr.txt`: real output of `python eval/retrieval_eval.py --k 6`
  (retrieval only, no LLM): recall of the gold section IDs on the 52 practice questions.
* Agent replies and spoken audio are **not** included yet. They need `GNANI_API_KEY` and an LLM key. After adding
  both to `.env`, record a few clips and run `python run.py my_clips --out sample_outputs/run1` to fill this folder.
