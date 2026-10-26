"""Step 2: GPT-4o-mini judge -- for every (question, passage) pair in
candidates.jsonl, decide SUPPORT (the passage directly backs the
question-answer relationship) or NOT_SUPPORT.

Resumable: results are appended to judgments.jsonl as they come in, and a
rerun skips (id, passage_idx) pairs already judged.

Requires OPENAI_API_KEY in the environment.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """Question: {question}
Expected answer: {answer}
Passage: {passage}

Does this passage directly support that "{answer}" is the correct answer to the question above? Answer with exactly one word: SUPPORT or NOT_SUPPORT."""


def judge_one(client: OpenAI, question: str, answer: str, passage: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=answer, passage=passage)}],
                max_tokens=5,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip().upper()
            if "NOT_SUPPORT" in text or "NOT SUPPORT" in text:
                return "NOT_SUPPORT"
            if "SUPPORT" in text:
                return "SUPPORT"
            return "NOT_SUPPORT"  # unparseable -> conservative default
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"  ...error ({e}), retrying in {wait}s")
            time.sleep(wait)
    return "ERROR"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(ROOT / "data/popqa_pilot/candidates.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/popqa_pilot/judgments.jsonl"))
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it in your terminal, then rerun this script.")

    client = OpenAI()

    already_done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            already_done.add((r["id"], r["passage_idx"]))
        print(f"Resuming: {len(already_done)} (question, passage) pairs already judged")

    questions = [json.loads(l) for l in open(args.candidates)]
    tasks = []
    for q in questions:
        answer = q["possible_answers"][0]
        for i, passage in enumerate(q["passages"]):
            if (q["id"], i) in already_done:
                continue
            tasks.append((q["id"], i, q["question"], answer, passage))

    print(f"{len(tasks)} (question, passage) pairs to judge, model={MODEL}, workers={args.max_workers}")
    if not tasks:
        print("Nothing to do.")
        return

    t0 = time.time()
    n_done = 0
    out_f = open(out_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(judge_one, client, question, answer, passage): (qid, pidx)
                for qid, pidx, question, answer, passage in tasks
            }
            for fut in as_completed(futures):
                qid, pidx = futures[fut]
                label = fut.result()
                out_f.write(json.dumps({"id": qid, "passage_idx": pidx, "label": label}) + "\n")
                out_f.flush()
                n_done += 1
                if n_done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta = (len(tasks) - n_done) / rate if rate > 0 else float("inf")
                    print(f"  {n_done}/{len(tasks)} done ({rate:.1f}/s, eta {eta:.0f}s)")
    finally:
        out_f.close()

    print(f"Done in {time.time()-t0:.1f}s. Wrote {args.out}")


if __name__ == "__main__":
    main()
