"""Phase 3b: Correct-vs-Wrong Citation Logit Margin, and each critical head's
directional causal contribution to it.

Unlike Phase 1's TV distance (undirected: ANY change in the restricted 3-way
softmax counts, whether it helps or hurts), this margin is signed and
task-aligned: it asks specifically whether the model's raw logits favor the
one source that actually supports the answer (evidence_position) over the
two that don't.

For instance x, let E(x) = {evidence_position} (the single genuine SUPPORT
source -- always exactly one, by construction of this dataset) and W(x) =
the other two source indices. At the position right before the citation
digit is emitted, with z_i the raw (unrestricted) logit of digit token i:

    M(x) = log(sum_{i in E} exp(z_i)) - log(sum_{j in W} exp(z_j))
         = z_evidence - logsumexp(z_wrong1, z_wrong2)      [|E(x)| = 1 here]

M > 0: logit mass favors the correct source. M < 0: favors a wrong source.

For each Phase-1 critical head h, zero-ablate it (same hook_z intervention as
Phase 1) and recompute M_{-h}(x). The head's task-specific causal importance
over a distribution T is the average drop in the correct-vs-wrong margin
caused by removing it:

    I_h^select(T) = E_{x~T}[ M(x) - M_{-h}(x) ]

I_h > 0: the head causally PROMOTES selecting the correct source (removing it
hurts M). I_h < 0: the head causally SUPPRESSES the correct source (removing
it helps M) -- e.g. a head that actively pulls toward a plausible-looking
wrong source.

Reported overall and split by citation_outcome (CORRECT / MISATTRIBUTION /
FABRICATION), plus a permutation test on whether a head's per-instance causal
effect d_h(x) = M(x) - M_{-h}(x) differs between the Misattribution and
Fabrication subsets -- i.e. does this head's causal pull toward the correct
source differ depending on which error the model ends up making.

Writes:
  results/phase3_logit_margin.json
  results/phase3_logit_margin.png
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
from scipy import stats
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


def logsumexp(vals: np.ndarray) -> np.ndarray:
    m = vals.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(vals - m).sum(axis=-1, keepdims=True))).squeeze(-1)


def compute_margin(z: np.ndarray, evidence_idx: np.ndarray) -> np.ndarray:
    """z: [N, 3] raw digit logits. evidence_idx: [N] in {0,1,2}. Returns M(x): [N]."""
    n = z.shape[0]
    rows = np.arange(n)
    z_evidence = z[rows, evidence_idx]
    wrong_mask = np.ones_like(z, dtype=bool)
    wrong_mask[rows, evidence_idx] = False
    z_wrong = z[wrong_mask].reshape(n, 2)
    return z_evidence - logsumexp(z_wrong)


def permutation_test_diff_means(x: np.ndarray, y: np.ndarray, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    observed = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    n1 = len(x)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        d = pooled[idx[:n1]].mean() - pooled[idx[n1:]].mean()
        if abs(d) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return float(observed), float(p_value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=str(ROOT / "data/popqa_pilot/llama_outcomes_full.jsonl"))
    ap.add_argument("--critical-heads", default=str(ROOT / "results/phase1_critical_heads.json"))
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--n-correct", type=int, default=300)
    ap.add_argument("--n-misattribution", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-sweep", action="store_true", help="ablate all 32x32 heads instead of just Phase 1's 15 critical heads")
    ap.add_argument("--top-k", type=int, default=15, help="in --full-sweep mode, how many heads to run the robust permutation test on")
    ap.add_argument("--out", default=str(ROOT / "results/phase3_logit_margin.json"))
    ap.add_argument("--plot", default=str(ROOT / "results/phase3_logit_margin.png"))
    args = ap.parse_args()

    phase1 = json.load(open(args.critical_heads))
    critical_heads = [(e["layer"], e["head"]) for e in phase1["top_critical_heads"]]
    n_layers, n_heads = phase1["n_layers"], phase1["n_heads"]
    if args.full_sweep:
        target_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        print(f"FULL SWEEP: ablating all {len(target_heads)} heads")
    else:
        target_heads = critical_heads
        print(f"critical heads from Phase 1: {critical_heads}")

    print("Loading model...")
    model = load_model(args.model_id, args.device)
    tok = model.tokenizer

    digit_ids = [tok(str(d), add_special_tokens=False)["input_ids"][-1] for d in (1, 2, 3)]
    print("digit token ids for 1/2/3:", digit_ids)

    rows = load_sample(args.generations, args.seed, args.n_correct, args.n_misattribution)
    print(f"sampled {len(rows)} instances: {Counter(r['citation_outcome'] for r in rows)}")

    instances = []
    for r in rows:
        inst = build_instance(tok, r)
        if inst is None:
            continue
        expected_digit_id = digit_ids[inst["cited_position"] - 1]
        if inst["actual_digit_id"] != expected_digit_id:
            continue  # sanity check, as in earlier phases
        instances.append(inst)
    print(f"{len(instances)} instances after sanity filtering: "
          f"{Counter(i['citation_outcome'] for i in instances)}")

    pad_id = tok.eos_token_id
    N = len(instances)
    outcomes_all = np.array([i["citation_outcome"] for i in instances])
    evidence_idx_all = np.array([i["evidence_position"] - 1 for i in instances])

    W_U_digits = model.W_U[:, digit_ids].float()  # [d_model, 3] -- project straight onto the 3 digit
    # columns instead of materializing full-vocab logits ([batch, seq, 128256] OOMs well before this
    # does, and we only ever need these 3 values).

    z_clean_all = np.zeros((N, 3))
    z_ablated_all = {hd: np.zeros((N, 3)) for hd in target_heads}

    def make_ablate_hook(head):
        def hook(z, hook):
            z[:, :, head, :] = 0.0
            return z
        return hook

    for start in range(0, N, args.batch_size):
        chunk = instances[start : start + args.batch_size]
        tokens, attn_mask = pad_batch(chunk, pad_id)
        tokens, attn_mask = tokens.to(model.cfg.device), attn_mask.to(model.cfg.device)
        target_pos_t = torch.tensor([inst["target_pos"] for inst in chunk], device=model.cfg.device)
        batch_idx_t = torch.arange(len(chunk), device=model.cfg.device)

        captured = {}

        def normed_hook(resid, hook):
            captured["normed"] = resid[batch_idx_t, target_pos_t].float()  # [b, d_model]
            return resid

        with torch.no_grad():
            model.run_with_hooks(
                tokens, attention_mask=attn_mask, fwd_hooks=[("ln_final.hook_normalized", normed_hook)], return_type=None
            )
            z_clean = (captured["normed"] @ W_U_digits).cpu().numpy()  # [b, 3]
        z_clean_all[start : start + len(chunk)] = z_clean

        for (l, h) in target_heads:
            with torch.no_grad():
                model.run_with_hooks(
                    tokens, attention_mask=attn_mask,
                    fwd_hooks=[(f"blocks.{l}.attn.hook_z", make_ablate_hook(h)), ("ln_final.hook_normalized", normed_hook)],
                    return_type=None,
                )
                z_ab = (captured["normed"] @ W_U_digits).cpu().numpy()
            z_ablated_all[(l, h)][start : start + len(chunk)] = z_ab
        print(f"  processed {min(start + args.batch_size, N)}/{N}")

    M_clean = compute_margin(z_clean_all, evidence_idx_all)
    mis_mask = outcomes_all == "MISATTRIBUTION"
    fab_mask = outcomes_all == "FABRICATION"

    results = {
        "n_instances": int(N),
        "n_per_outcome": {o: int((outcomes_all == o).sum()) for o in OUTCOMES},
        "M_clean_overall_mean": float(M_clean.mean()),
        "M_clean_by_outcome": {o: float(M_clean[outcomes_all == o].mean()) for o in OUTCOMES},
        "heads": {},
    }

    print("\n=== M(x) by outcome (clean, no ablation) ===")
    for o in OUTCOMES:
        print(f"  {o}: mean M = {M_clean[outcomes_all == o].mean():.4f} (n={int((outcomes_all==o).sum())})")

    # fast per-head screen (every head, Welch t-test -- closed form, no resampling)
    print(f"\n=== screening {len(target_heads)} heads (fast Welch t-test, no resampling) ===")
    d_by_head = {}
    screen = []
    for (l, h) in target_heads:
        M_ablated = compute_margin(z_ablated_all[(l, h)], evidence_idx_all)
        d = M_clean - M_ablated  # d(x) = M(x) - M_{-h}(x); d>0 => head promotes correct source
        d_by_head[(l, h)] = d
        t_stat, p_fast = stats.ttest_ind(d[mis_mask], d[fab_mask], equal_var=False)
        screen.append({"head": (l, h), "I_select": float(d.mean()), "t_stat": float(t_stat), "p_fast": float(p_fast)})

    if args.full_sweep:
        # two independent screens: (a) overall causal importance |I_select|, mirroring Phase 1's TV
        # ranking but signed/directional; (b) heads whose causal effect itself differs most between
        # Misattribution and Fabrication. Robust (permutation) test only on the union -- running
        # 10000 resamples x 1024 heads would be wasteful when a closed-form test already screens them.
        by_importance = sorted(screen, key=lambda e: -abs(e["I_select"]))[: args.top_k]
        by_mis_fab_split = sorted(screen, key=lambda e: e["p_fast"])[: args.top_k]
        chosen = {e["head"] for e in by_importance} | {e["head"] for e in by_mis_fab_split}
        heads_to_report = sorted(chosen)
        results["full_sweep_grid"] = {
            f"L{l}H{h}": {"I_select": e["I_select"], "t_stat": e["t_stat"], "p_fast_welch": e["p_fast"]}
            for (l, h), e in zip([e["head"] for e in screen], screen)
        }
        results["full_sweep_top_by_importance"] = [f"L{l}H{h}" for (l, h) in [e["head"] for e in by_importance]]
        results["full_sweep_top_by_misattr_fab_split"] = [f"L{l}H{h}" for (l, h) in [e["head"] for e in by_mis_fab_split]]
        print(f"top {args.top_k} by |I_select|: {results['full_sweep_top_by_importance']}")
        print(f"top {args.top_k} by misattr-vs-fab Welch p: {results['full_sweep_top_by_misattr_fab_split']}")
        print(f"running robust permutation test on the union ({len(heads_to_report)} heads)...")
    else:
        heads_to_report = target_heads

    print("\n=== I_h^select(T): directional causal importance (robust permutation test) ===")
    for (l, h) in heads_to_report:
        d = d_by_head[(l, h)]
        per_outcome = {}
        for o in OUTCOMES:
            mask = outcomes_all == o
            per_outcome[o] = {"mean": float(d[mask].mean()), "std": float(d[mask].std()), "n": int(mask.sum())}

        obs_diff, p_diff = permutation_test_diff_means(d[mis_mask], d[fab_mask])

        head_result = {
            "head": f"L{l}H{h}",
            "I_select_overall": float(d.mean()),
            "I_select_std_overall": float(d.std()),
            "I_select_by_outcome": per_outcome,
            "misattr_vs_fabrication_diff_in_causal_effect": {
                "observed_diff": obs_diff,
                "permutation_p": p_diff,
            },
        }
        results["heads"][f"L{l}H{h}"] = head_result
        print(
            f"  L{l}H{h}: I_select={d.mean():+.4f} | "
            f"CORRECT={per_outcome['CORRECT']['mean']:+.4f} "
            f"MISATTR={per_outcome['MISATTRIBUTION']['mean']:+.4f} "
            f"FABRIC={per_outcome['FABRICATION']['mean']:+.4f} | "
            f"MISATTRvsFABRIC diff={obs_diff:+.4f} p={p_diff:.4f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")

    # bar chart: I_select per head, broken down by outcome
    heads_sorted = heads_to_report
    labels = [f"L{l}H{h}" for l, h in heads_sorted]
    x = np.arange(len(labels))
    width = 0.25
    colors = {"CORRECT": "#0ca30c", "MISATTRIBUTION": "#fab219", "FABRICATION": "#d03b3b"}
    plt.figure(figsize=(14, 6))
    for i, o in enumerate(OUTCOMES):
        vals = [results["heads"][lbl]["I_select_by_outcome"][o]["mean"] for lbl in labels]
        plt.bar(x + (i - 1) * width, vals, width, label=o, color=colors[o])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(r"$I_h^{select}$ (mean drop in correct-vs-wrong margin when head is ablated)")
    plt.title("Phase 3b: directional causal importance per critical head, by citation outcome")
    plt.legend()
    plt.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
