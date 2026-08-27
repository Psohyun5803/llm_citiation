"""Phase 1.2/1.3 (pilot): parse generated citations into fields and tag tokens.

Reads data/generated_citations/generations.jsonl (written by generate.py) and,
for each generation, rule-based-parses citation lines that follow the
requested "AuthorLastName et al. (Year). 'Paper Title'. Venue." format.

Writes:
  data/verified_citations/parsed_citations.jsonl
      one row per parsed citation, with domain field, parsed author/year/
      title/venue text, and character spans into generated_text.
  data/generated_citations/token_labels/{prompt_id}.json
      per-token field labels (author/title/venue/year) for tokens whose
      character span falls inside a parsed citation field, built from the
      saved token ids via incremental decoding.

Citations the parser can't match (e.g. purely inline prose mentions with no
structured reference line) are skipped, matching the protocol's guidance to
exclude parser failures rather than guess.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-2-9b-it"

CITATION_RE = re.compile(
    r"(?P<author>[A-Z][\w\-À-ɏ'’]+(?:,?\s+(?:and|&)\s+[A-Z][\w\-À-ɏ'’]+)?(?:\s+et\s+al\.)?)"
    r"\s*,?\s*\(\s*(?P<year>\d{4})\s*\)[,.]?\s*"
    r"['\"‘’“”](?P<title>[^'\"‘’“”]{5,200})['\"‘’“”][,.]?\s*"
    r"(?P<venue>[^\n,.]{2,150})[,.]"
)


def token_char_offsets(tokenizer, token_ids: list[int]) -> list[tuple[int, int]]:
    """Approximate per-token character spans in the decoded full text via
    incremental decoding (cumulative prefix decode)."""
    offsets = []
    prev_len = 0
    for i in range(1, len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:i], skip_special_tokens=True)
        offsets.append((prev_len, len(text)))
        prev_len = len(text)
    return offsets


def label_tokens(offsets, spans: dict[str, tuple[int, int]]) -> dict[int, str]:
    labels = {}
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for field, (fstart, fend) in spans.items():
            if start >= fstart and end <= fend:
                labels[tok_idx] = field
                break
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=str(ROOT / "data/generated_citations/generations.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/verified_citations/parsed_citations.jsonl"))
    ap.add_argument("--token-labels-dir", default=str(ROOT / "data/generated_citations/token_labels"))
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    token_labels_dir = Path(args.token_labels_dir)
    token_labels_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    rows = [json.loads(l) for l in open(args.generations)]
    parsed_rows = []
    n_matched_prompts = 0

    for row in rows:
        text = row["generated_text"]
        matches = list(CITATION_RE.finditer(text))
        if not matches:
            continue
        n_matched_prompts += 1

        act = torch.load(ROOT / row["activation_path"], weights_only=False)
        token_ids = act["token_ids"].tolist()
        offsets = token_char_offsets(tokenizer, token_ids)

        prompt_token_labels = []
        for cidx, m in enumerate(matches):
            spans = {
                "author": m.span("author"),
                "year": m.span("year"),
                "title": m.span("title"),
                "venue": m.span("venue"),
            }
            tok_labels = label_tokens(offsets, spans)
            for tok_idx, field in tok_labels.items():
                prompt_token_labels.append({"token_idx": tok_idx, "citation_idx": cidx, "field": field})

            parsed_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "domain": row["field"],
                    "citation_idx": cidx,
                    "author": m.group("author").strip(),
                    "year": m.group("year").strip(),
                    "title": m.group("title").strip(),
                    "venue": m.group("venue").strip(),
                }
            )

        with open(token_labels_dir / f"{row['prompt_id']}.json", "w") as f:
            json.dump(prompt_token_labels, f)

    with open(args.out, "w") as f:
        for r in parsed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Parsed citations from {n_matched_prompts}/{len(rows)} generations")
    print(f"Total citations parsed: {len(parsed_rows)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
