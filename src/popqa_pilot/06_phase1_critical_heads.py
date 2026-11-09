"""Phase 1: identify citation-critical attention heads in Llama-3.1-8B-Instruct
via zero-ablation + Total Variation distance over the restricted 3-way
source-citation softmax, then validate that these heads' attention patterns
predict actual citation choices better than random non-critical heads.

Methodology:
  - Sample N instances from llama_outcomes_full.jsonl with a valid
    cited_position in {1,2,3}.
  - For each instance, reconstruct the exact context fed to the model: prompt
    + generated answer, truncated right after the last "[N]" citation. The
    token position right BEFORE the digit token is where the model is about
    to decide which source to cite.
  - At that position, restrict the logits to {token("1"), token("2"),
    token("3")} and renormalize -> a 3-way distribution P(D1,D2,D3), "which
    source am I about to cite."
  - For every one of the 32x32 (layer, head) pairs, zero-ablate that head's
    hook_z (all positions, all batch rows) and recompute this restricted
    distribution. The Total Variation distance between clean and ablated
    distributions, averaged across instances, is the head's criticality
    score -- regardless of whether ablation helps or hurts correctness (this
    is a screen for causal involvement, not for correctness).
  - Validation: using the clean (unablated) attention pattern at the same
    position, sum attention mass over each source's token span to get an
    attention-based P(Di). Check whether argmax(attention P(Di)) matches the
    actual cited_position, for the top critical heads vs a matched sample of
    random non-critical heads -- critical heads should track the actual
    citation decision much better than noise.

Writes:
  results/phase1_critical_heads.json
  results/phase1_critical_heads_heatmap.png
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

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
    model = HookedTransformer.from_pretrained_no_processing(model_id, dtype=torch.float16, device=device)
    model.eval()
    return model


def load_sample(path, n, seed):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["cited_position"] in (1, 2, 3)]
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["difficulty"], r["evidence_position"])].append(r)
    rng = random.Random(seed)
    for g in by_group.values():
        rng.shuffle(g)
    per_group = max(1, n // len(by_group))
    sample = []
    for g in by_group.values():
        sample.extend(g[:per_group])
    rng.shuffle(sample)
    return sample[:n]


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
        "difficulty": row["difficulty"],
        "evidence_position": row["evidence_position"],
        "citation_outcome": row.get("citation_outcome"),
    }


def restricted_softmax(logits_row: torch.Tensor, digit_ids: list[int]) -> np.ndarray:
    vals = logits_row[digit_ids].float()
    probs = torch.softmax(vals, dim=0)
    return probs.cpu().numpy()


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


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
    ap.add_argument("--generations", default=str(ROOT / "data/popqa_pilot/llama_outcomes_full.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/phase1_critical_heads.json"))
    ap.add_argument("--heatmap", default=str(ROOT / "results/phase1_critical_heads_heatmap.png"))
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-random-heads", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading model...")
    model = load_model(args.model_id, args.device)
    tok = model.tokenizer
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    print(f"n_layers={n_layers} n_heads={n_heads}")

    digit_ids = [tok(str(d), add_special_tokens=False)["input_ids"][-1] for d in (1, 2, 3)]
    print("digit token ids for 1/2/3:", digit_ids)

    rows = load_sample(args.generations, args.n, args.seed)
    print(f"sampled {len(rows)} instances")

    instances = []
    for r in rows:
        inst = build_instance(tok, r)
        if inst is None:
            continue
        actual_digit_id = inst["ids"][inst["target_pos"] + 1]
        expected_digit_id = digit_ids[inst["cited_position"] - 1]
        if actual_digit_id != expected_digit_id:
            continue  # sanity check failed, skip this instance
        instances.append(inst)
    print(f"{len(instances)} instances passed sanity check")

    pad_id = tok.eos_token_id
    tokens, attn_mask = pad_batch(instances, pad_id)
    tokens, attn_mask = tokens.to(model.cfg.device), attn_mask.to(model.cfg.device)
    target_pos = torch.tensor([inst["target_pos"] for inst in instances], device=model.cfg.device)
    batch_idx = torch.arange(len(instances), device=model.cfg.device)

    # --- clean pass: baseline restricted-softmax + attention-based P(Di) ---
    # extracted via per-layer hooks (not run_with_cache) so only one layer's
    # [N, n_heads, seq, seq] pattern tensor is ever materialized at once --
    # caching all 32 layers simultaneously OOMs at this batch size.
    print("Running clean forward pass (streaming attention extraction)...")
    attn_pdi = np.zeros((len(instances), n_layers, n_heads, 3))

    def make_attn_extract_hook(layer_idx):
        def hook(pattern, hook):
            for i, inst in enumerate(instances):
                row_attn = pattern[i, :, inst["target_pos"], :]  # [n_heads, seq_k]
                for k in (1, 2, 3):
                    s, e = inst["spans"][k]
                    attn_pdi[i, layer_idx, :, k - 1] = row_attn[:, s:e].sum(dim=-1).float().cpu().numpy()
            return pattern

        return hook

    attn_hooks = [(f"blocks.{l}.attn.hook_pattern", make_attn_extract_hook(l)) for l in range(n_layers)]
    with torch.no_grad():
        clean_logits = model.run_with_hooks(tokens, attention_mask=attn_mask, fwd_hooks=attn_hooks)
    clean_logits_at_pos = clean_logits[batch_idx, target_pos]  # [N, vocab]
    baseline_probs = np.stack(
        [restricted_softmax(clean_logits_at_pos[i], digit_ids) for i in range(len(instances))]
    )  # [N, 3]
    del clean_logits

    # --- zero-ablation sweep ---
    print(f"Running {n_layers}x{n_heads} zero-ablation sweep...")
    tv_grid = np.zeros((n_layers, n_heads))

    def make_hook(head):
        def hook(z, hook):
            z[:, :, head, :] = 0.0
            return z

        return hook

    for l in range(n_layers):
        for h in range(n_heads):
            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    attention_mask=attn_mask,
                    fwd_hooks=[(f"blocks.{l}.attn.hook_z", make_hook(h))],
                )
            logits_at_pos = ablated_logits[batch_idx, target_pos]
            ablated_probs = np.stack(
                [restricted_softmax(logits_at_pos[i], digit_ids) for i in range(len(instances))]
            )
            tvs = [tv_distance(baseline_probs[i], ablated_probs[i]) for i in range(len(instances))]
            tv_grid[l, h] = float(np.mean(tvs))
            del ablated_logits
        print(f"  layer {l+1}/{n_layers} done, row mean TV={tv_grid[l].mean():.4f} max={tv_grid[l].max():.4f}")

    # --- identify critical heads ---
    flat = [(l, h, tv_grid[l, h]) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(key=lambda x: -x[2])
    background_mean, background_std = tv_grid.mean(), tv_grid.std()
    top_heads = [
        {"layer": l, "head": h, "tv": tv, "z_score": (tv - background_mean) / background_std}
        for l, h, tv in flat[: args.top_k]
    ]
    print("\nTop critical heads by TV distance:")
    for e in top_heads:
        print(f"  L{e['layer']}H{e['head']}: TV={e['tv']:.4f} z={e['z_score']:.2f}")

    # --- validation: attention-P(Di) argmax vs actual citation, critical vs random heads ---
    rng = random.Random(args.seed)
    critical_set = {(e["layer"], e["head"]) for e in top_heads}
    non_critical_pool = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in critical_set]
    random_heads = rng.sample(non_critical_pool, args.n_random_heads)

    def accuracy_for(head_list):
        accs = []
        for (l, h) in head_list:
            correct = 0
            for i, inst in enumerate(instances):
                pred = int(np.argmax(attn_pdi[i, l, h])) + 1
                if pred == inst["cited_position"]:
                    correct += 1
            accs.append(correct / len(instances))
        return accs

    critical_accs = accuracy_for([(e["layer"], e["head"]) for e in top_heads])
    random_accs = accuracy_for(random_heads)
    print(f"\nValidation -- argmax(attention P(Di)) == actual citation:")
    print(f"  critical heads (n={len(critical_accs)}): mean acc = {np.mean(critical_accs):.3f}")
    print(f"  random non-critical heads (n={len(random_accs)}): mean acc = {np.mean(random_accs):.3f}")

    result = {
        "n_instances": len(instances),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "tv_grid": tv_grid.tolist(),
        "background_mean": background_mean,
        "background_std": background_std,
        "top_critical_heads": top_heads,
        "validation": {
            "critical_heads": [(e["layer"], e["head"]) for e in top_heads],
            "critical_head_accuracies": critical_accs,
            "critical_mean_acc": float(np.mean(critical_accs)),
            "random_heads": random_heads,
            "random_head_accuracies": random_accs,
            "random_mean_acc": float(np.mean(random_accs)),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")

    plt.figure(figsize=(10, 9))
    plt.imshow(tv_grid, cmap="viridis", aspect="auto")
    plt.colorbar(label="mean TV distance")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Phase 1: Zero-ablation TV-distance (citation-criticality) by head")
    plt.tight_layout()
    plt.savefig(args.heatmap, dpi=150)
    print(f"Wrote {args.heatmap}")


if __name__ == "__main__":
    main()
