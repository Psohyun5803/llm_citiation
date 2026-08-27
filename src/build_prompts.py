"""Build the Phase 1.1 citation-generation prompt set from data/prompts/topics.json.

Usage:
    python src/build_prompts.py

Reads data/prompts/topics.json (field -> list of topics) and expands each
topic into a citation-generation prompt using the fixed template from the
experiment protocol. Writes data/prompts/prompts.jsonl and a companion
prompts.csv for quick inspection.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPICS_PATH = ROOT / "data" / "prompts" / "topics.json"
JSONL_OUT = ROOT / "data" / "prompts" / "prompts.jsonl"
CSV_OUT = ROOT / "data" / "prompts" / "prompts.csv"

TEMPLATE = (
    "Write a short academic paragraph about {topic} that cites 3-5 specific "
    "papers. For each citation, include the full author names, paper title, "
    "publication venue, and year. Format each reference as: AuthorLastName "
    "et al. (Year). 'Paper Title'. Venue."
)


def build_prompts() -> list[dict]:
    topics_by_field = json.loads(TOPICS_PATH.read_text())
    rows = []
    for field, topics in topics_by_field.items():
        for i, topic in enumerate(topics, start=1):
            rows.append(
                {
                    "prompt_id": f"{field}_{i:03d}",
                    "field": field,
                    "topic": topic,
                    "prompt": TEMPLATE.format(topic=topic),
                }
            )
    return rows


def main() -> None:
    rows = build_prompts()

    with JSONL_OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "field", "topic", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    field_counts = {}
    for row in rows:
        field_counts[row["field"]] = field_counts.get(row["field"], 0) + 1

    print(f"Wrote {len(rows)} prompts across {len(field_counts)} fields to:")
    print(f"  {JSONL_OUT.relative_to(ROOT)}")
    print(f"  {CSV_OUT.relative_to(ROOT)}")
    for field, count in field_counts.items():
        print(f"  {field}: {count}")


if __name__ == "__main__":
    main()
