"""Phase 2: for the Phase 1 critical heads, compute two attention-based
decision-state signals at the citation position:
  - A_source: total attention mass the head pays to the three source spans
    combined (vs. question/instruction text) -- "is this head even looking
    at the sources right now."
  - H_source: Shannon entropy (nats) of how that mass is split across the
    three sources, after renormalizing to sum to 1 -- "how concentrated vs.
    spread is the attention, among the sources it does look at."

Both are compared across Correct / Misattribution / Fabrication citation
outcomes. If error types are mechanistically distinct we'd expect, e.g.,
Fabrication to show low A_source (attention isn't even going to the sources)
and/or high H_source (attention is split/uncertain across sources) relative
to Correct.

Reads results/phase1_critical_heads.json for the head list and
data/popqa_pilot/llama_outcomes_full.jsonl for (generation, outcome) pairs.

Writes:
  results/phase2_decision_state.json
  results/phase2_decision_state.png
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

PROMPT_TEMPLATE = """You are given a question and a set of candidate sources.
Question: {question}
Candidate Sources:
[1] {doc1}  [2] {doc2}  [3] {doc3}
Answer the question using the provided sources.
Answer:"""

CITATION_RE = re.compile(r"\[(\d+)\]")

OUTCOME_COLORS = {"CORRECT": "#0ca30c", "MISATTRIBUTION": "#fab219", "FABRICATION": "#d03b3b"}
OUTCOMES = ["CORRECT", "MISATTRIBUTION", "FABRICATION"]


def load_model(device: str):
    model = HookedTransformer.from_pretrained_no_processing(MODEL_ID, dtype=torch.float16, device=device)
    model.eval()
    return model


def load_sample(path, seed, n_correct, n_misattribution):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["cited_position"] in (1, 2, 3) and r["citation_outcome"] in OUTCOMES]
    by_outcome = defaultdict(list)
    for r in rows:
        by_outcome[r["citation_outcome"]].append(r)
    rng = random.Random(seed)
    for g in by_outcome.values():
        rng.shuffle(g)
    sample = list(by_outcome["FABRICATION"])  # take all -- rare class
    sample += by_outcome["CORRECT"][:n_correct]
    sample += by_outcome["MISATTRIBUTION"][:n_misattribution]
    rng.shuffle(sample)
    return sample


def build_instance(tokenizer, row):
    prompt = PROMPT_TEMPLATE.format(
        question=row["question"], doc1=row["sources"][0], doc2=row["sources"][1], doc3=row["sources"][2]
    )
    gen = row["generated_text"]
    matches = list(CITATION_RE.finditer(gen))
    last = matches[-1]
    truncated_gen = gen[: last.end()]
    full_text = prompt + " " + truncated_gen
    digit_char_start = len(prompt) + 1 + last.start() + 1

    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    digit_tok_idx = None
    for i, (s, e) in enumerate(offsets):
        if s <= digit_char_start < e:
            digit_tok_idx = i
            break
    if digit_tok_idx is None or digit_tok_idx == 0:
        return None
    target_pos = digit_tok_idx - 1

    prompt_enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    prompt_offsets = prompt_enc["offset_mapping"]
    search_from = 0
    bounds = []
    for k in (1, 2, 3):
        start_char = prompt.index(f"[{k}]", search_from)
        bounds.append(start_char)
        search_from = start_char + 1
    bounds.append(prompt.index("\nAnswer the question"))

    spans = {}
    for k in (1, 2, 3):
        start_char, end_char = bounds[k - 1], bounds[k]
        tok_start = next(i for i, (s, e) in enumerate(prompt_offsets) if s <= start_char < e)
        tok_end = next((i for i, (s, e) in enumerate(prompt_offsets) if s >= end_char), len(prompt_enc["input_ids"]))
        spans[k] = (tok_start, tok_end)

    return {
        "ids": ids,
        "target_pos": target_pos,
        "spans": spans,
        "cited_position": row["cited_position"],
        "citation_outcome": row["citation_outcome"],
        "difficulty": row["difficulty"],
        "evidence_position": row["evidence_position"],
    }


def pad_batch(instances, pad_id):
    max_len = max(len(inst["ids"]) for inst in instances)
    n = len(instances)
    tokens = torch.full((n, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, inst in enumerate(instances):
        L = len(inst["ids"])
        tokens[i, :L] = torch.tensor(inst["ids"], dtype=torch.long)
        attn_mask[i, :L] = 1
    return tokens, attn_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default=str(ROOT / "results/phase1_critical_heads.json"))
    ap.add_argument("--generations", default=str(ROOT / "data/popqa_pilot/llama_outcomes_full.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/phase2_decision_state.json"))
    ap.add_argument("--plot", default=str(ROOT / "results/phase2_decision_state.png"))
    ap.add_argument("--n-correct", type=int, default=300)
    ap.add_argument("--n-misattribution", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    phase1 = json.load(open(args.phase1))
    critical_heads = [tuple(h) for h in phase1["validation"]["critical_heads"]]
    print(f"{len(critical_heads)} critical heads: {critical_heads}")

    print("Loading model...")
    model = load_model(args.device)
    tok = model.tokenizer

    rows = load_sample(args.generations, args.seed, args.n_correct, args.n_misattribution)
    print(f"sampled {len(rows)} instances: {Counter(r['citation_outcome'] for r in rows)}")

    instances = []
    for r in rows:
        inst = build_instance(tok, r)
        if inst is not None:
            instances.append(inst)
    print(f"{len(instances)} instances after sanity filtering: "
          f"{Counter(i['citation_outcome'] for i in instances)}")

    pad_id = tok.eos_token_id
    critical_layers = sorted({l for l, h in critical_heads})
    heads_by_layer = defaultdict(list)
    for l, h in critical_heads:
        heads_by_layer[l].append(h)

    per_head_records = {hd: [] for hd in critical_heads}

    for start in range(0, len(instances), args.batch_size):
        chunk = instances[start : start + args.batch_size]
        tokens, attn_mask = pad_batch(chunk, pad_id)
        tokens, attn_mask = tokens.to(model.cfg.device), attn_mask.to(model.cfg.device)

        captured = {}

        def make_hook(layer_idx):
            def hook(pattern, hook):
                captured[layer_idx] = [
                    pattern[i, :, chunk[i]["target_pos"], :].float().cpu().numpy() for i in range(len(chunk))
                ]
                return pattern

            return hook

        hooks = [(f"blocks.{l}.attn.hook_pattern", make_hook(l)) for l in critical_layers]
        with torch.no_grad():
            model.run_with_hooks(tokens, attention_mask=attn_mask, fwd_hooks=hooks, return_type=None)

        for l in critical_layers:
            for h in heads_by_layer[l]:
                for i, inst in enumerate(chunk):
                    row_attn = captured[l][i][h]  # [seq_k]
                    raw = np.array([row_attn[inst["spans"][k][0] : inst["spans"][k][1]].sum() for k in (1, 2, 3)])
                    a_source = float(raw.sum())
                    if a_source > 1e-8:
                        p = raw / a_source
                        h_source = float(-np.sum(p * np.log(p + 1e-12)))
                    else:
                        h_source = float(np.log(3))
                    per_head_records[(l, h)].append(
                        {"a_source": a_source, "h_source": h_source, "citation_outcome": inst["citation_outcome"]}
                    )
        print(f"  processed {min(start + args.batch_size, len(instances))}/{len(instances)}")

    summary = {}
    for (l, h), records in per_head_records.items():
        by_outcome = defaultdict(list)
        for r in records:
            by_outcome[r["citation_outcome"]].append(r)
        summary[f"L{l}H{h}"] = {
            outcome: {
                "n": len(v),
                "mean_a_source": float(np.mean([x["a_source"] for x in v])) if v else None,
                "std_a_source": float(np.std([x["a_source"] for x in v])) if v else None,
                "mean_h_source": float(np.mean([x["h_source"] for x in v])) if v else None,
                "std_h_source": float(np.std([x["h_source"] for x in v])) if v else None,
            }
            for outcome, v in by_outcome.items()
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"critical_heads": critical_heads, "n_instances": len(instances), "summary": summary}, f, indent=2)
    print(f"Wrote {args.out}")

    head_labels = [f"L{l}H{h}" for l, h in critical_heads]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    x = np.arange(len(head_labels))
    width = 0.25
    for metric_idx, (ax, metric, ylabel, title) in enumerate(
        zip(
            axes,
            ["mean_a_source", "mean_h_source"],
            ["A_source (total attention mass on sources)", "H_source (entropy over sources, nats)"],
            ["Attention mass on sources, by citation outcome", "Attention entropy across sources, by citation outcome"],
        )
    ):
        for oi, outcome in enumerate(OUTCOMES):
            vals = [summary[lbl].get(outcome, {}).get(metric) or 0 for lbl in head_labels]
            ax.bar(x + (oi - 1) * width, vals, width, label=outcome, color=OUTCOME_COLORS[outcome])
        ax.set_xticks(x)
        ax.set_xticklabels(head_labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
