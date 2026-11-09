"""Phase 2b: using A_source and H_source as a 2D state space (per critical
head), test whether Correct / Misattribution / Fabrication instances occupy
distinguishable regions -- in particular whether the two ERROR types
(Misattribution, Fabrication) are separable from each other, not just from
Correct.

This does not presuppose which axis drives any separation found; it reports
whatever the data shows. If the three (or just the two error) conditions
separate significantly, that supports designing outcome-specific interventions
in Phase 3. If they don't, the two error types may share a common internal
mechanism.

Methodology, per head:
  - Standardize (A_source, H_source) across the full pooled sample (z-score).
  - For every pair of outcomes: Euclidean centroid distance in the
    standardized 2D space, tested via a label-permutation test (10000
    permutations) -- nonparametric, robust to the class-size imbalance
    (Fabrication is rare).
  - Hotelling's two-sample T^2 test (parametric cross-check) for the same
    pairs.
  - A 5-fold stratified cross-validated LDA classifier: (a) 3-way
    Correct/Misattribution/Fabrication, (b) binary Misattribution-vs-
    Fabrication only -- balanced accuracy vs. chance.

Reads results/phase1_critical_heads.json (for context) and
data/popqa_pilot/llama_outcomes_full.jsonl.

Writes:
  results/phase2b_separation_<head>.json
  results/phase2b_separation_<head>.png
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


def hotelling_t2(x: np.ndarray, y: np.ndarray):
    """Two-sample Hotelling's T^2 test, converted to an F statistic / p-value."""
    n1, n2 = len(x), len(y)
    p = x.shape[1]
    mean_diff = x.mean(axis=0) - y.mean(axis=0)
    cov_pooled = ((n1 - 1) * np.cov(x, rowvar=False) + (n2 - 1) * np.cov(y, rowvar=False)) / (n1 + n2 - 2)
    cov_inv = np.linalg.pinv(cov_pooled)
    t2 = (n1 * n2 / (n1 + n2)) * mean_diff @ cov_inv @ mean_diff
    f_stat = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    df1, df2 = p, n1 + n2 - p - 1
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    return float(t2), float(f_stat), float(p_value)


def permutation_centroid_test(x: np.ndarray, y: np.ndarray, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    observed = np.linalg.norm(x.mean(axis=0) - y.mean(axis=0))
    pooled = np.vstack([x, y])
    n1 = len(x)
    n_total = len(pooled)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(n_total)
        px, py = pooled[idx[:n1]], pooled[idx[n1:]]
        d = np.linalg.norm(px.mean(axis=0) - py.mean(axis=0))
        if d >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return float(observed), float(p_value)


def analyze_head(layer, head, records, out_prefix):
    outcomes = [r["citation_outcome"] for r in records]
    feats = np.array([[r["a_source"], r["h_source"]] for r in records])
    mean, std = feats.mean(axis=0), feats.std(axis=0)
    z = (feats - mean) / std

    by_outcome_z = {o: z[[i for i, oc in enumerate(outcomes) if oc == o]] for o in OUTCOMES}
    by_outcome_raw = {o: feats[[i for i, oc in enumerate(outcomes) if oc == o]] for o in OUTCOMES}

    pairwise = {}
    for a, b in combinations(OUTCOMES, 2):
        if len(by_outcome_z[a]) < 2 or len(by_outcome_z[b]) < 2:
            continue
        t2, f_stat, p_hotelling = hotelling_t2(by_outcome_z[a], by_outcome_z[b])
        obs_dist, p_perm = permutation_centroid_test(by_outcome_z[a], by_outcome_z[b])
        pairwise[f"{a}_vs_{b}"] = {
            "n_a": len(by_outcome_z[a]),
            "n_b": len(by_outcome_z[b]),
            "centroid_distance_z": obs_dist,
            "permutation_p": p_perm,
            "hotelling_t2": t2,
            "hotelling_f": f_stat,
            "hotelling_p": p_hotelling,
        }

    # 3-way and binary (error-only) cross-validated LDA
    y_all = np.array(outcomes)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc_3way = cross_val_score(LinearDiscriminantAnalysis(), z, y_all, cv=skf, scoring="balanced_accuracy")

    error_mask = np.isin(y_all, ["MISATTRIBUTION", "FABRICATION"])
    z_err, y_err = z[error_mask], y_all[error_mask]
    skf_err = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc_binary_err = cross_val_score(LinearDiscriminantAnalysis(), z_err, y_err, cv=skf_err, scoring="balanced_accuracy")

    result = {
        "head": f"L{layer}H{head}",
        "n_total": len(records),
        "n_per_outcome": dict(Counter(outcomes)),
        "pairwise": pairwise,
        "lda_3way_balanced_acc_mean": float(acc_3way.mean()),
        "lda_3way_balanced_acc_std": float(acc_3way.std()),
        "lda_3way_chance": 1 / 3,
        "lda_misattr_vs_fabrication_balanced_acc_mean": float(acc_binary_err.mean()),
        "lda_misattr_vs_fabrication_balanced_acc_std": float(acc_binary_err.std()),
        "lda_binary_chance": 0.5,
    }

    plt.figure(figsize=(8, 7))
    for o in OUTCOMES:
        pts = by_outcome_raw[o]
        plt.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.55, label=f"{o} (n={len(pts)})", color=OUTCOME_COLORS[o])
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        plt.scatter([cx], [cy], s=220, marker="X", color=OUTCOME_COLORS[o], edgecolors="black", linewidths=1.2, zorder=5)
    plt.xlabel("A_source (total attention mass on sources)")
    plt.ylabel("H_source (entropy over sources, nats)")
    plt.title(f"L{layer}H{head}: decision-state space by citation outcome\n(X = group centroid)")
    plt.legend()
    plt.grid(color="#e1e0d9", linewidth=0.8, zorder=0)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    plt.close()

    with open(f"{out_prefix}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=str(ROOT / "data/popqa_pilot/llama_outcomes_full.jsonl"))
    ap.add_argument("--heads", default="13,18;8,11", help="semicolon-separated layer,head pairs")
    ap.add_argument("--n-correct", type=int, default=300)
    ap.add_argument("--n-misattribution", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    target_heads = [tuple(int(x) for x in pair.split(",")) for pair in args.heads.split(";")]
    print(f"target heads: {target_heads}")

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
    critical_layers = sorted({l for l, h in target_heads})
    heads_by_layer = defaultdict(list)
    for l, h in target_heads:
        heads_by_layer[l].append(h)

    per_head_records = {hd: [] for hd in target_heads}

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
                    row_attn = captured[l][i][h]
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

    for (l, h), records in per_head_records.items():
        out_prefix = str(Path(args.out_dir) / f"phase2b_separation_L{l}H{h}")
        result = analyze_head(l, h, records, out_prefix)
        print(f"\n=== L{l}H{h} ===")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
