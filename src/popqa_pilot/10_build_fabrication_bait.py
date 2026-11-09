"""Build fabrication-inducing instances: unlike instances.jsonl (which always
includes one genuine SUPPORT passage among the 3 sources), every instance
here has THREE distractors and NO evidence at all -- the model cannot ground
its citation in any of the given sources no matter what it cites. If it still
emits a confident "[N]" citation, the claim is much more likely to be a true
Fabrication (content invented from parametric memory, matching none of the
given passages) rather than a Misattribution (real content, wrong source),
since there is no real content anywhere in the prompt to misattribute from.

Prefers Hard distractors (same-article NOT_SUPPORT passages -- maximally
deceptive, since they're topically on-point but don't answer the question);
falls back to Easy distractors (other articles) to fill out 3 when a question
doesn't have enough Hard ones.

Writes data/popqa_pilot/fabrication_bait_instances.jsonl -- same schema as
instances.jsonl (difficulty="no_evidence", evidence_position=0 as a sentinel
meaning "no real evidence present") so it drops straight into
04_generate_llama.py and 05_classify_citation_outcome.py unmodified.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load(candidates_path, judgments_path):
    candidates = {json.loads(l)["id"]: json.loads(l) for l in open(candidates_path)}
    judgments = defaultdict(lambda: {"SUPPORT": [], "NOT_SUPPORT": []})
    for line in open(judgments_path):
        j = json.loads(line)
        if j["label"] in ("SUPPORT", "NOT_SUPPORT"):
            judgments[j["id"]][j["label"]].append(j["passage_idx"])
    return candidates, judgments


def build_three_distractors(qid, candidates, judgments, rng, n=3):
    own_not_support = list(judgments[qid]["NOT_SUPPORT"])
    rng.shuffle(own_not_support)
    chosen = [candidates[qid]["passages"][i] for i in own_not_support[:n]]

    if len(chosen) < n:
        other_qids = [q for q in candidates if q != qid]
        tries = 0
        while len(chosen) < n and tries < n * 30:
            tries += 1
            oq = rng.choice(other_qids)
            if candidates[oq]["s_wiki_title"] == candidates[qid]["s_wiki_title"]:
                continue
            chosen.append(rng.choice(candidates[oq]["passages"]))
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(ROOT / "data/popqa_pilot/candidates.jsonl"))
    ap.add_argument("--judgments", default=str(ROOT / "data/popqa_pilot/judgments.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/popqa_pilot/fabrication_bait_instances.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    candidates, judgments = load(args.candidates, args.judgments)
    rng = random.Random(args.seed)

    instances = []
    n_no_support = 0
    for qid in candidates:
        if qid not in judgments or not judgments[qid]["SUPPORT"]:
            n_no_support += 1
            continue  # keep the same universe as instances.jsonl (questions with a real answer)
        distractors = build_three_distractors(qid, candidates, judgments, rng)
        if len(distractors) < 3:
            continue
        rng.shuffle(distractors)
        instances.append(
            {
                "id": qid,
                "question": candidates[qid]["question"],
                "answer": candidates[qid]["possible_answers"][0],
                "difficulty": "no_evidence",
                "evidence_position": 0,
                "sources": distractors,
            }
        )

    print(f"{n_no_support} questions had no SUPPORT passage (skipped, same as instances.jsonl universe)")
    print(f"{len(instances)} no-evidence (fabrication-bait) instances built")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
