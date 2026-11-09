"""Phase 3c: can a linear probe on the raw residual stream (no ground truth at
test time) separate CORRECT / MISATTRIBUTION / FABRICATION better than the
hand-crafted H_source/A_source or A/H+DLA features did?

Motivation: Phase 3's state-dependent intervention needs a discriminator that
is computable WITHOUT knowing evidence_position (the ground-truth answer) --
unlike M(x), which is a diagnostic metric only. H_source/A_source (the
proposal's original features) and the A/H+DLA combined classifier both only
reached ~chance-to-weak separation. This script asks whether probing the full
residual stream (not just attention mass over 15 hand-picked heads) at the
citation-decision position carries a cleaner signal, since it doesn't throw
away any of the model's internal representation at that point.

For every layer l, extract resid_post at the position right before the
citation digit token (same target_pos as all earlier phases), for the same
labeled sample (300 CORRECT + 300 MISATTRIBUTION + all FABRICATION). Train a
probe is done for each layer with 5-fold stratified CV, with PCA inside the
pipeline (fit per-fold, so no leakage) to handle d_model >> n_samples:

    StandardScaler -> PCA(n_components) -> LogisticRegression

Reports per-layer balanced accuracy for the 3-way task (CORRECT vs
MISATTRIBUTION vs FABRICATION) and for the narrower MISATTRIBUTION-vs-
FABRICATION binary task (the one that matters for routing which intervention
to apply), and writes a plot of both curves across layers.

Writes:
  results/phase3c_residual_probe.json
  results/phase3c_residual_probe.png
  results/phase3c_residual_probe_features.npz  (best layer's raw features, for reuse)
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
OUTCOMES = ["CORRECT", "MISATTRIBUTION", "FABRICATION"]
BAL_ACC = make_scorer(balanced_accuracy_score)


def load_model(model_id: str, device: str):
    model = HookedTransformer.from_pretrained_no_processing(model_id, dtype=torch.float16, device=device)
    model.eval()
    return model


def load_sample(path, seed, n_correct, n_misattribution):
    rows = [json.loads(l) for l in open(path)]
    rows = [
        r for r in rows
        if r["cited_position"] in (1, 2, 3) and r["citation_outcome"] in OUTCOMES and r["evidence_position"] in (1, 2, 3)
    ]
    by_outcome = defaultdict(list)
    for r in rows:
        by_outcome[r["citation_outcome"]].append(r)
    rng = random.Random(seed)
    for g in by_outcome.values():
        rng.shuffle(g)
    sample = list(by_outcome["FABRICATION"])
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
    actual_digit_id = ids[digit_tok_idx]

    return {
        "ids": ids,
        "target_pos": target_pos,
        "actual_digit_id": actual_digit_id,
        "cited_position": row["cited_position"],
        "evidence_position": row["evidence_position"],
        "citation_outcome": row["citation_outcome"],
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


def cv_balanced_acc(X, y, n_components, seed=0):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    scores = cross_val_score(pipe, X, y, cv=skf, scoring=BAL_ACC)
    return float(scores.mean()), float(scores.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=str(ROOT / "data/popqa_pilot/llama_outcomes_full.jsonl"))
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--n-correct", type=int, default=300)
    ap.add_argument("--n-misattribution", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pca-components", type=int, default=40)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/phase3c_residual_probe.json"))
    ap.add_argument("--plot", default=str(ROOT / "results/phase3c_residual_probe.png"))
    ap.add_argument("--features-out", default=str(ROOT / "results/phase3c_residual_probe_features.npz"))
    args = ap.parse_args()

    print("Loading model...")
    model = load_model(args.model_id, args.device)
    tok = model.tokenizer
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
    print(f"n_layers={n_layers} d_model={d_model}")

    digit_ids = [tok(str(d), add_special_tokens=False)["input_ids"][-1] for d in (1, 2, 3)]

    rows = load_sample(args.generations, args.seed, args.n_correct, args.n_misattribution)
    print(f"sampled {len(rows)} instances: {Counter(r['citation_outcome'] for r in rows)}")

    instances = []
    for r in rows:
        inst = build_instance(tok, r)
        if inst is None:
            continue
        expected_digit_id = digit_ids[inst["cited_position"] - 1]
        if inst["actual_digit_id"] != expected_digit_id:
            continue
        instances.append(inst)
    print(f"{len(instances)} instances after sanity filtering: {Counter(i['citation_outcome'] for i in instances)}")

    pad_id = tok.eos_token_id
    N = len(instances)
    outcomes_all = np.array([i["citation_outcome"] for i in instances])

    resid_all = np.zeros((N, n_layers, d_model), dtype=np.float32)

    for start in range(0, N, args.batch_size):
        chunk = instances[start : start + args.batch_size]
        tokens, attn_mask = pad_batch(chunk, pad_id)
        tokens, attn_mask = tokens.to(model.cfg.device), attn_mask.to(model.cfg.device)
        target_pos_t = torch.tensor([inst["target_pos"] for inst in chunk], device=model.cfg.device)
        batch_idx_t = torch.arange(len(chunk), device=model.cfg.device)

        captured = {}

        def make_resid_hook(layer_idx):
            def hook(resid, hook):
                captured[layer_idx] = resid[batch_idx_t, target_pos_t].float().cpu().numpy()
                return resid
            return hook

        hooks = [(f"blocks.{l}.hook_resid_post", make_resid_hook(l)) for l in range(n_layers)]
        with torch.no_grad():
            model.run_with_hooks(tokens, attention_mask=attn_mask, fwd_hooks=hooks, return_type=None)
        for l in range(n_layers):
            resid_all[start : start + len(chunk), l, :] = captured[l]
        print(f"  processed {min(start + args.batch_size, N)}/{N}")

    del model
    torch.cuda.empty_cache()

    mis_mask = outcomes_all == "MISATTRIBUTION"
    fab_mask = outcomes_all == "FABRICATION"
    bin_mask = mis_mask | fab_mask
    y_bin = outcomes_all[bin_mask]

    results = {
        "n_instances": int(N),
        "n_per_outcome": {o: int((outcomes_all == o).sum()) for o in OUTCOMES},
        "pca_components": args.pca_components,
        "per_layer": {},
    }

    print("\n=== per-layer residual-stream probe: 3-way (CORRECT/MISATTR/FABRIC) and MISATTRvsFABRIC ===")
    threeway_accs, binary_accs = [], []
    for l in range(n_layers):
        X_layer = resid_all[:, l, :]
        nc = min(args.pca_components, len(instances) - 5)
        acc3, std3 = cv_balanced_acc(X_layer, outcomes_all, nc, seed=args.seed)
        accb, stdb = cv_balanced_acc(X_layer[bin_mask], y_bin, min(nc, bin_mask.sum() - 5), seed=args.seed)
        threeway_accs.append(acc3)
        binary_accs.append(accb)
        results["per_layer"][l] = {
            "threeway_balanced_acc": acc3, "threeway_std": std3,
            "misattr_vs_fab_balanced_acc": accb, "misattr_vs_fab_std": stdb,
        }
        print(f"  L{l}: 3-way={acc3:.3f}(+/-{std3:.3f})  MISATTRvsFABRIC={accb:.3f}(+/-{stdb:.3f})")

    best3_layer = int(np.argmax(threeway_accs))
    bestb_layer = int(np.argmax(binary_accs))
    results["best_layer_threeway"] = {"layer": best3_layer, "balanced_acc": threeway_accs[best3_layer]}
    results["best_layer_misattr_vs_fab"] = {"layer": bestb_layer, "balanced_acc": binary_accs[bestb_layer]}
    print(f"\nBest layer for 3-way: L{best3_layer} ({threeway_accs[best3_layer]:.3f})")
    print(f"Best layer for MISATTRvsFABRIC: L{bestb_layer} ({binary_accs[bestb_layer]:.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")

    np.savez(
        args.features_out,
        resid_best3=resid_all[:, best3_layer, :],
        resid_bestb=resid_all[:, bestb_layer, :],
        outcomes=outcomes_all,
        best3_layer=best3_layer,
        bestb_layer=bestb_layer,
    )
    print(f"Wrote {args.features_out}")

    plt.figure(figsize=(12, 5))
    x = np.arange(n_layers)
    plt.plot(x, threeway_accs, marker="o", label="3-way (CORRECT/MISATTR/FABRIC)", color="#2a6fdb")
    plt.plot(x, binary_accs, marker="o", label="MISATTRIBUTION vs FABRICATION", color="#d03b3b")
    plt.axhline(1 / 3, color="#2a6fdb", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.axhline(0.5, color="#d03b3b", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.xlabel("Layer")
    plt.ylabel("5-fold CV balanced accuracy")
    plt.title("Phase 3c: residual-stream linear probe accuracy by layer (Llama-3.1-8B)")
    plt.legend()
    plt.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()


