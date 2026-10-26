"""Step 1: sample PopQA questions and build a 100-word passage pool for
each, from ryannoonan/popqa-wikipedia-contexts.

Writes data/popqa_pilot/candidates.jsonl -- one row per question:
  {id, question, possible_answers, s_wiki_title, passages: [str, ...]}

Passages are built by merging each article's paragraph list into one text
(dropping bare section-heading entries, which carry no judgeable content),
then splitting on whitespace into ~100-word chunks (no overlap).
"""

import argparse
import csv
import gzip
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent.parent
WORDS_PER_PASSAGE = 100


def load_popqa() -> list[dict]:
    path = hf_hub_download("akariasai/PopQA", "test.tsv", repo_type="dataset")
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_contexts() -> dict[str, list[str]]:
    """title -> list of paragraph strings, for status=='ok' records only."""
    path = hf_hub_download("ryannoonan/popqa-wikipedia-contexts", "popqa_contexts.jsonl.gz", repo_type="dataset")
    by_title = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok" and rec.get("paragraphs"):
                by_title[rec["title"]] = rec["paragraphs"]
    return by_title


def is_heading_only(paragraph: str) -> bool:
    """Bare section headings ("Early life", "References", ...) carry no
    judgeable content -- merge them into the following paragraph instead of
    keeping them as their own tiny passage fragment."""
    words = paragraph.split()
    return len(words) <= 4 and not paragraph.endswith((".", "!", "?"))


def paragraphs_to_text(paragraphs: list[str]) -> str:
    merged = []
    pending_heading = None
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if is_heading_only(p):
            pending_heading = p
            continue
        if pending_heading:
            merged.append(f"{pending_heading}. {p}")
            pending_heading = None
        else:
            merged.append(p)
    return " ".join(merged)


def split_into_passages(text: str, words_per_passage: int = WORDS_PER_PASSAGE) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + words_per_passage])
        for i in range(0, len(words), words_per_passage)
        if len(words[i : i + words_per_passage]) >= 20  # drop tiny tail fragments
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=30, help="pilot default; use 1000 for the full run")
    ap.add_argument("--min-passages", type=int, default=3, help="need >=1 evidence + >=2 distractor passages")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/popqa_pilot/candidates.jsonl"))
    args = ap.parse_args()

    print("Loading PopQA...")
    popqa = load_popqa()
    print(f"  {len(popqa)} questions")

    print("Loading popqa-wikipedia-contexts...")
    contexts = load_contexts()
    print(f"  {len(contexts)} articles with status=ok")

    candidates = []
    for row in popqa:
        title = row["s_wiki_title"]
        paragraphs = contexts.get(title)
        if not paragraphs:
            continue
        text = paragraphs_to_text(paragraphs)
        passages = split_into_passages(text)
        if len(passages) < args.min_passages:
            continue
        candidates.append(
            {
                "id": row["id"],
                "question": row["question"],
                "possible_answers": json.loads(row["possible_answers"]),
                "s_wiki_title": title,
                "passages": passages,
            }
        )

    print(f"{len(candidates)} questions have an article with >= {args.min_passages} passages")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    sampled = candidates[: args.n_questions]
    print(f"Sampled {len(sampled)} questions (seed={args.seed})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for c in sampled:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")

    n_passages = [len(c["passages"]) for c in sampled]
    print(f"passages per question: min={min(n_passages)} max={max(n_passages)} "
          f"mean={sum(n_passages)/len(n_passages):.1f}")


if __name__ == "__main__":
    main()
