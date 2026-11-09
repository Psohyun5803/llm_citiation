"""Step 4: generate answers with Llama-3.1-8B-Instruct on PopQA instances,
using the fixed 3-source prompt template, and locate which source number
the model cited.

Prompt (exact template from the spec):
    You are given a question and a set of candidate sources.
    Question: {question}
    Candidate Sources:
    [1] {document_1}  [2] {document_2}  [3] {document_3}
    Answer the question using the provided sources.
    Answer: {your answer} [source number]

Writes data/popqa_pilot/llama_generations.jsonl -- one row per instance:
  {id, difficulty, evidence_position, sources, generated_text, cited_position}
cited_position is None if no "[N]" citation could be found in the output.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_TEMPLATE = """You are given a question and a set of candidate sources.
Question: {question}
Candidate Sources:
[1] {doc1}  [2] {doc2}  [3] {doc3}
Answer the question using the provided sources.
Answer:"""

CITATION_RE = re.compile(r"\[(\d+)\]")


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
    )
    model.eval()
    return tokenizer, model


def find_cited_position(text: str) -> int | None:
    matches = CITATION_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])  # the LAST "[N]" in the answer is the citation per the template


def instance_key(inst: dict) -> str:
    # instances.jsonl has 6 rows sharing the same "id" (2 difficulties x 3
    # evidence positions) -- this triple is what's actually unique per row
    return f"{inst['id']}::{inst['difficulty']}::{inst['evidence_position']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=str(ROOT / "data/popqa_pilot/instances.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/popqa_pilot/llama_generations.jsonl"))
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--n-pilot", type=int, default=None, help="if set, sample only this many instances; default: all")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help="split the remaining work across this many parallel processes")
    ap.add_argument("--shard-index", type=int, default=0, help="which shard this process handles (0-indexed)")
    args = ap.parse_args()

    instances = [json.loads(l) for l in open(args.instances)]
    if args.n_pilot is not None:
        rng = random.Random(args.seed)
        rng.shuffle(instances)
        instances = instances[: args.n_pilot]
    print(f"{len(instances)} instances to process")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    # merge every shard's output file so each shard sees everyone's progress
    shard_glob = out_path.with_suffix("")  # e.g. .../llama_generations
    for p in list(out_path.parent.glob(f"{shard_glob.name}*.jsonl")):
        for line in open(p):
            done_keys.add(json.loads(line)["instance_key"])
    print(f"Resuming: {len(done_keys)} instances already done across all shard files, skipping those")

    remaining = [inst for inst in instances if instance_key(inst) not in done_keys]
    todo = [inst for i, inst in enumerate(remaining) if i % args.num_shards == args.shard_index]
    print(f"{len(remaining)} instances remaining overall; this shard ({args.shard_index}/{args.num_shards}) handles {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    tokenizer, model = load_model(args.model_id, args.device)

    shard_out_path = (
        out_path if args.num_shards == 1 else out_path.parent / f"{shard_glob.name}.shard{args.shard_index}.jsonl"
    )
    out_f = open(shard_out_path, "a")
    n_cited_evidence = 0
    n_no_citation = 0
    t0 = time.time()
    try:
        for i, inst in enumerate(todo):
            prompt = PROMPT_TEMPLATE.format(
                question=inst["question"], doc1=inst["sources"][0], doc2=inst["sources"][1], doc3=inst["sources"][2]
            )
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            cited = find_cited_position(gen_text)

            row = {
                "instance_key": instance_key(inst),
                "id": inst["id"],
                "question": inst["question"],
                "answer": inst["answer"],
                "difficulty": inst["difficulty"],
                "evidence_position": inst["evidence_position"],
                "sources": inst["sources"],
                "generated_text": gen_text,
                "cited_position": cited,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            if cited == inst["evidence_position"]:
                n_cited_evidence += 1
            if cited is None:
                n_no_citation += 1

            if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(todo) - i - 1) / rate if rate > 0 else float("inf")
                print(f"[{i+1}/{len(todo)}] rate={rate:.2f}/s eta={eta/60:.1f}min "
                      f"cited={cited} evidence_at={inst['evidence_position']}")
    finally:
        out_f.close()

    print(f"\nThis run: cited evidence correctly: {n_cited_evidence}/{len(todo)}")
    print(f"This run: no citation found: {n_no_citation}/{len(todo)}")
    print(f"Wrote {shard_out_path} ({len(todo)} rows this shard)")


if __name__ == "__main__":
    main()
