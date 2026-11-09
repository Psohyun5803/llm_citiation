"""Step 5: GPT-4o-mini classifies each generated answer's citation outcome
as CORRECT, MISATTRIBUTION, or FABRICATION, per the spec's definitions:

  - Correct: the generated claim is supported by the cited source (and that
    source genuinely answers the question).
  - Misattribution: the claim's content matches what the cited source says,
    but that source does not directly support the question's answer (real
    content, wrong/non-supporting source).
  - Fabrication: the claim's content does not match the cited source's
    actual content at all, or cites a source number that wasn't provided.

Reads *_generations.jsonl (output of 04_generate_llama.py or an equivalent
script for another model) and writes a parallel *_outcomes.jsonl.
"""

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """Question: {question}
Expected answer: {answer}

Candidate source the model cited ([{cited}]):
\"\"\"{cited_text}\"\"\"

Model's generated answer:
\"\"\"{generated}\"\"\"

Classify this citation into exactly one category:
- CORRECT: the generated answer's claim is supported by the cited source, and the cited source genuinely contains/supports the expected answer.
- MISATTRIBUTION: the generated answer's claim content does match what the cited source actually says, but that source does NOT contain or support the expected answer to the question (i.e. real content, but from a source that doesn't answer the question).
- FABRICATION: the generated answer's claim content does NOT match what the cited source actually says (the model made up or misreported the cited source's content).

Respond with exactly one word: CORRECT, MISATTRIBUTION, or FABRICATION."""


def classify_one(client: OpenAI, question, answer, cited, cited_text, generated, retries=4) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=question, answer=answer, cited=cited, cited_text=cited_text, generated=generated
                        ),
                    }
                ],
                max_tokens=6,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip().upper()
            for label in ("CORRECT", "MISATTRIBUTION", "FABRICATION"):
                if label in text:
                    return label
            return "FABRICATION"  # unparseable -> conservative default
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"  ...error ({e}), retrying in {wait}s")
            time.sleep(wait)
    return "ERROR"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, help="e.g. data/popqa_pilot/llama_generations.jsonl")
    ap.add_argument("--out", default=None, help="default: <generations>.replace('generations','outcomes')")
    args = ap.parse_args()
    out_path = args.out or args.generations.replace("generations", "outcomes")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it in your terminal, then rerun this script.")
    client = OpenAI()

    rows = [json.loads(l) for l in open(args.generations)]
    results = []
    counts = {"CORRECT": 0, "MISATTRIBUTION": 0, "FABRICATION": 0, "NO_CITATION": 0, "ERROR": 0}

    for i, r in enumerate(rows):
        cited = r["cited_position"]
        if cited is None:
            # genuinely no "[N]" citation marker anywhere in the output
            label = "NO_CITATION"
        elif not (1 <= cited <= len(r["sources"])):
            # cited a source number that was never provided -- this is exactly
            # the "citation을 가리키는 source의 내용이 ... 완전히 제공되지 않은
            # 내용을 citation한 경우" branch of the Fabrication definition,
            # not an absence of citation
            label = "FABRICATION"
        else:
            cited_text = r["sources"][cited - 1]
            label = classify_one(client, r["question"], r["answer"], cited, cited_text, r["generated_text"])
        counts[label] = counts.get(label, 0) + 1
        results.append({**r, "citation_outcome": label})
        print(f"[{i+1}/{len(rows)}] {r['id']} cited={cited} evidence_at={r['evidence_position']} -> {label}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nOutcome counts: {counts}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
