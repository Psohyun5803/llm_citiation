"""Step 3: from candidates.jsonl + judgments.jsonl, build the final
citation-source-selection prompt instances.

For each question with >=1 SUPPORT passage:
  - evidence = one SUPPORT passage (from its own article)
  - Easy distractors (x2) = random passages from OTHER questions' articles
    (topically unrelated)
  - Hard distractors (x2) = NOT_SUPPORT passages from the SAME article,
    falling back to NOT_SUPPORT passages (from other articles) that mention
    the gold answer string -- an answer-entity-retrieval proxy -- if the
    same-article pool has fewer than 2, and finally to same-article passages
    regardless of judgment if still short.

Each (question, difficulty) pair gets 3 position-shuffled variants: evidence
at slot 1, 2, or 3 among the 3 sources -- 2 difficulties x 3 positions = 6
instances per question. All instances of the same question are kept in the
same train/test split to avoid leakage.

Writes data/popqa_pilot/instances.jsonl
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


def build_hard_distractors(qid, candidates, judgments, rng, n=2):
    own_not_support = judgments[qid]["NOT_SUPPORT"]
    pool = [candidates[qid]["passages"][i] for i in own_not_support]
    rng.shuffle(pool)
    chosen = pool[:n]

    if len(chosen) < n:
        # fall back: NOT_SUPPORT passages elsewhere that mention the answer string
        answer = candidates[qid]["possible_answers"][0].lower()
        extra = []
        other_qids = [q for q in judgments if q != qid]
        rng.shuffle(other_qids)
        for oq in other_qids:
            for i in judgments[oq]["NOT_SUPPORT"]:
                text = candidates[oq]["passages"][i]
                if answer in text.lower():
                    extra.append(text)
            if len(chosen) + len(extra) >= n:
                break
        chosen += extra[: n - len(chosen)]

    if len(chosen) < n:
        # last resort: any other same-article passage not already used
        used = set(own_not_support) | set(judgments[qid]["SUPPORT"])
        remaining = [i for i in range(len(candidates[qid]["passages"])) if i not in used]
        rng.shuffle(remaining)
        chosen += [candidates[qid]["passages"][i] for i in remaining[: n - len(chosen)]]

    return chosen[:n]


def build_easy_distractors(qid, candidates, rng, n=2):
    other_qids = [q for q in candidates if q != qid]
    chosen = []
    tries = 0
    while len(chosen) < n and tries < n * 20:
        tries += 1
        oq = rng.choice(other_qids)
        if candidates[oq]["s_wiki_title"] == candidates[qid]["s_wiki_title"]:
            continue
        passage = rng.choice(candidates[oq]["passages"])
        chosen.append(passage)
    return chosen


def make_variants(qid, candidates, judgments, rng):
    support_idx = judgments[qid]["SUPPORT"]
    if not support_idx:
        return []
    evidence = candidates[qid]["passages"][rng.choice(support_idx)]
    answer = candidates[qid]["possible_answers"][0]
    question = candidates[qid]["question"]

    variants = []
    for difficulty, distractors in [
        ("easy", build_easy_distractors(qid, candidates, rng)),
        ("hard", build_hard_distractors(qid, candidates, judgments, rng)),
    ]:
        if len(distractors) < 2:
            continue  # not enough material for this difficulty, skip it
        d1, d2 = distractors[:2]
        slot_layouts = [
            [evidence, d1, d2],
            [d1, evidence, d2],
            [d1, d2, evidence],
        ]
        for evidence_position, sources in enumerate(slot_layouts, start=1):
            variants.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": answer,
                    "difficulty": difficulty,
                    "evidence_position": evidence_position,
                    "sources": sources,
                }
            )
    return variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(ROOT / "data/popqa_pilot/candidates.jsonl"))
    ap.add_argument("--judgments", default=str(ROOT / "data/popqa_pilot/judgments.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/popqa_pilot/instances.jsonl"))
    ap.add_argument("--split-out", default=str(ROOT / "data/popqa_pilot/splits.json"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    candidates, judgments = load(args.candidates, args.judgments)
    rng = random.Random(args.seed)

    all_instances = []
    n_no_evidence = 0
    for qid in candidates:
        if qid not in judgments or not judgments[qid]["SUPPORT"]:
            n_no_evidence += 1
            continue
        all_instances.extend(make_variants(qid, candidates, judgments, rng))

    print(f"{n_no_evidence} questions had no SUPPORT passage (dropped)")
    print(f"{len(all_instances)} total instances from {len(candidates) - n_no_evidence} questions")

    qids = sorted({inst["id"] for inst in all_instances})
    rng.shuffle(qids)
    n_test = max(1, int(len(qids) * args.test_frac))
    test_ids = set(qids[:n_test])
    train_ids = set(qids[n_test:])

    for inst in all_instances:
        inst["split"] = "test" if inst["id"] in test_ids else "train"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for inst in all_instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    json.dump({"train": sorted(train_ids), "test": sorted(test_ids)}, open(args.split_out, "w"), indent=2)

    print(f"train questions={len(train_ids)} test questions={len(test_ids)}")
    print(f"train instances={sum(1 for i in all_instances if i['split']=='train')} "
          f"test instances={sum(1 for i in all_instances if i['split']=='test')}")
    print(f"Wrote {args.out} and {args.split_out}")


if __name__ == "__main__":
    main()
