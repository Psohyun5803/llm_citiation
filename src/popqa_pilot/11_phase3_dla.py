"""Phase 3: Direct Logit Attribution (DLA) per critical head.

Phase 1 found 15 heads whose zero-ablation strongly perturbs the restricted
citation-digit softmax (high TV distance) -- i.e. causally important. Phase
1's own validation and Phase 2b/2c then showed that for most of these heads,
*where the head attends* (A_source / H_source, summed over each source's
token span) does NOT predict the actual citation, and does NOT separate
Misattribution from Fabrication. So causal importance is established, but
not explained by attention location.

DLA tests the other half of the attention head: not the QK pattern (where it
looks) but the OV circuit (what it writes to the residual stream once it has
looked). For each critical head, at the same "about to emit the citation
digit" position used throughout this project:

  1. Take the head's hook_z output (post-softmax, pre-W_O), shape [d_head].
  2. Project through that head's own W_O slice -> head_out, the head's direct
     additive contribution to the residual stream, shape [d_model].
  3. Undo Llama's final RMSNorm *scale* (the cached per-position scale factor
     -- RMSNorm has no mean-centering and no bias, so this is exact, unlike
     LayerNorm models) and apply ln_final's elementwise gamma.
  4. Project onto the three unembedding columns for digit tokens "1"/"2"/"3"
     -> a 3-vector of this head's direct contribution to each citation logit,
     with every other component of the network (all other heads, all MLPs,
     all other layers) held out. This is linear and exact given the clean
     run's scale factor; it is the standard DLA decomposition.

Two analyses, mirroring the earlier phases but with DLA instead of attention
mass:
  (a) DLA-argmax accuracy: does argmax(DLA_1, DLA_2, DLA_3) match the actual
      citation, for critical heads vs. random non-critical heads? (mirrors
      Phase 1's attention-based validation)
  (b) Separation: does (DLA_cited, DLA_margin) -- the head's direct push
      toward the digit the model actually emitted, and the margin over the
      runner-up -- separate CORRECT / MISATTRIBUTION / FABRICATION, in
      particular the two error types from each other? (mirrors Phase 2b)

Writes:
  results/phase3_dla.json
  results/phase3_dla_<head>.png  (one scatter per analyzed head, DLA space)
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
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
OUTCOMES = ["CORRECT", "MISATTRIBUTION", "FABRICATION"]
OUTCOME_COLORS = {"CORRECT": "#0ca30c", "MISATTRIBUTION": "#fab219", "FABRICATION": "#d03b3b"}


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

    actual_digit_id = ids[digit_tok_idx]
    return {
        "ids": ids,
        "target_pos": target_pos,
        "actual_digit_id": actual_digit_id,
        "cited_position": row["cited_position"],
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


def analyze_separation(outcomes, feats):
    """feats: [N, 2] = (DLA_cited, DLA_margin). Same stats battery as Phase 2b."""
    outcomes = np.array(outcomes)
    mean, std = feats.mean(axis=0), feats.std(axis=0)
    z = (feats - mean) / std

    by_outcome_z = {o: z[outcomes == o] for o in OUTCOMES}
    by_outcome_raw = {o: feats[outcomes == o] for o in OUTCOMES}

    pairwise = {}
    for a, b in combinations(OUTCOMES, 2):
        if len(by_outcome_z[a]) < 2 or len(by_outcome_z[b]) < 2:
            continue
        t2, f_stat, p_hotelling = hotelling_t2(by_outcome_z[a], by_outcome_z[b])
        obs_dist, p_perm = permutation_centroid_test(by_outcome_z[a], by_outcome_z[b])
        pairwise[f"{a}_vs_{b}"] = {
            "n_a": int(len(by_outcome_z[a])),
            "n_b": int(len(by_outcome_z[b])),
            "centroid_distance_z": obs_dist,
            "permutation_p": p_perm,
            "hotelling_t2": t2,
            "hotelling_f": f_stat,
            "hotelling_p": p_hotelling,
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc_3way = cross_val_score(LinearDiscriminantAnalysis(), z, outcomes, cv=skf, scoring="balanced_accuracy")

    error_mask = np.isin(outcomes, ["MISATTRIBUTION", "FABRICATION"])
    z_err, y_err = z[error_mask], outcomes[error_mask]
    skf_err = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc_binary_err = cross_val_score(LinearDiscriminantAnalysis(), z_err, y_err, cv=skf_err, scoring="balanced_accuracy")

    return {
        "pairwise": pairwise,
        "lda_3way_balanced_acc_mean": float(acc_3way.mean()),
        "lda_3way_balanced_acc_std": float(acc_3way.std()),
        "lda_3way_chance": 1 / 3,
        "lda_misattr_vs_fabrication_balanced_acc_mean": float(acc_binary_err.mean()),
        "lda_misattr_vs_fabrication_balanced_acc_std": float(acc_binary_err.std()),
        "lda_binary_chance": 0.5,
    }, by_outcome_raw


def plot_head(layer, head, by_outcome_raw, out_path):
    plt.figure(figsize=(8, 7))
    for o in OUTCOMES:
        pts = by_outcome_raw[o]
        if len(pts) == 0:
            continue
        plt.scatter(pts[:, 0], pts[:, 1], s=28, alpha=0.55, label=f"{o} (n={len(pts)})", color=OUTCOME_COLORS[o])
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        plt.scatter([cx], [cy], s=220, marker="X", color=OUTCOME_COLORS[o], edgecolors="black", linewidths=1.2, zorder=5)
    plt.axhline(0, color="#999", linewidth=0.8)
    plt.xlabel("DLA_cited (direct logit contribution to the emitted digit)")
    plt.ylabel("DLA_margin (DLA_cited - max DLA_other)")
    plt.title(f"L{layer}H{head}: Direct Logit Attribution space by citation outcome\n(X = group centroid)")
    plt.legend()
    plt.grid(color="#e1e0d9", linewidth=0.8, zorder=0)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=str(ROOT / "data/popqa_pilot/llama_outcomes_full.jsonl"))
    ap.add_argument("--critical-heads", default=str(ROOT / "results/phase1_critical_heads.json"))
    ap.add_argument("--n-correct", type=int, default=300)
    ap.add_argument("--n-misattribution", type=int, default=300)
    ap.add_argument("--n-random-heads", type=int, default=15)
    ap.add_argument("--plot-top-k", type=int, default=4, help="how many top critical heads to render scatter plots for")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/phase3_dla.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    phase1 = json.load(open(args.critical_heads))
    critical_heads = [(e["layer"], e["head"]) for e in phase1["top_critical_heads"]]
    n_layers, n_heads = phase1["n_layers"], phase1["n_heads"]
    print(f"critical heads from Phase 1: {critical_heads}")

    rng = random.Random(args.seed)
    non_critical_pool = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in set(critical_heads)]
    random_heads = rng.sample(non_critical_pool, args.n_random_heads)
    all_heads = critical_heads + random_heads

    print("Loading model...")
    model = load_model(args.device)
    tok = model.tokenizer
    d_model = model.cfg.d_model

    digit_ids = [tok(str(d), add_special_tokens=False)["input_ids"][-1] for d in (1, 2, 3)]
    digit_id_to_k = {did: k for k, did in enumerate(digit_ids)}  # k: 0,1,2 <-> digit 1,2,3
    print("digit token ids for 1/2/3:", digit_ids)

    W_U_digits = model.W_U[:, digit_ids].float()  # [d_model, 3]
    ln_final_gamma = model.ln_final.w.float()  # [d_model]

    rows = load_sample(args.generations, args.seed, args.n_correct, args.n_misattribution)
    print(f"sampled {len(rows)} instances: {Counter(r['citation_outcome'] for r in rows)}")

    instances = []
    for r in rows:
        inst = build_instance(tok, r)
        if inst is None:
            continue
        expected_digit_id = digit_ids[inst["cited_position"] - 1]
        if inst["actual_digit_id"] != expected_digit_id:
            continue  # sanity check, as in Phase 1/2b
        instances.append(inst)
    print(f"{len(instances)} instances after sanity filtering: "
          f"{Counter(i['citation_outcome'] for i in instances)}")

    pad_id = tok.eos_token_id
    layers_needed = sorted({l for l, h in all_heads})

    # per-head, per-instance DLA 3-vector (direct contribution to digit-1/2/3 logits)
    dla = {hd: [] for hd in all_heads}
    outcomes_all = []
    cited_positions_all = []

    for start in range(0, len(instances), args.batch_size):
        chunk = instances[start : start + args.batch_size]
        tokens, attn_mask = pad_batch(chunk, pad_id)
        tokens, attn_mask = tokens.to(model.cfg.device), attn_mask.to(model.cfg.device)
        target_pos_t = torch.tensor([inst["target_pos"] for inst in chunk], device=model.cfg.device)
        batch_idx_t = torch.arange(len(chunk), device=model.cfg.device)

        captured_z = {}
        captured_scale = {}

        def make_z_hook(layer_idx):
            def hook(z, hook):
                # z: [batch, seq, n_heads, d_head] -- keep only the target position
                captured_z[layer_idx] = z[batch_idx_t, target_pos_t].float().cpu()  # [batch, n_heads, d_head]
                return z

            return hook

        def scale_hook(scale, hook):
            # ln_final.hook_scale: [batch, seq, 1]
            captured_scale["final"] = scale[batch_idx_t, target_pos_t].float().cpu()  # [batch, 1]
            return scale

        hooks = [(f"blocks.{l}.attn.hook_z", make_z_hook(l)) for l in layers_needed]
        hooks.append(("ln_final.hook_scale", scale_hook))

        with torch.no_grad():
            model.run_with_hooks(tokens, attention_mask=attn_mask, fwd_hooks=hooks, return_type=None)

        scale = captured_scale["final"]  # [batch, 1]
        with torch.no_grad():
            for (l, h) in all_heads:
                z_lh = captured_z[l][:, h, :]  # [batch, d_head]
                W_O_lh = model.W_O[l, h].float().cpu()  # [d_head, d_model]
                head_out = z_lh @ W_O_lh  # [batch, d_model] -- this head's additive contribution to resid_post[l]
                # RMSNorm (Llama, no_processing): no mean-centering, no bias.
                normed = head_out / scale * ln_final_gamma.cpu()  # [batch, d_model]
                contrib = normed @ W_U_digits.cpu()  # [batch, 3] -- direct contribution to digit-1/2/3 logits
                dla[(l, h)].append(contrib.detach().numpy())

        outcomes_all.extend(inst["citation_outcome"] for inst in chunk)
        cited_positions_all.extend(inst["cited_position"] for inst in chunk)
        print(f"  processed {min(start + args.batch_size, len(instances))}/{len(instances)}")

    for hd in dla:
        dla[hd] = np.concatenate(dla[hd], axis=0)  # [N, 3]

    outcomes_all = np.array(outcomes_all)
    cited_idx = np.array([p - 1 for p in cited_positions_all])  # 0,1,2
    N = len(outcomes_all)

    def dla_accuracy(head_list):
        accs = []
        for hd in head_list:
            pred = np.argmax(dla[hd], axis=1)
            accs.append(float((pred == cited_idx).mean()))
        return accs

    critical_dla_accs = dla_accuracy(critical_heads)
    random_dla_accs = dla_accuracy(random_heads)
    print("\n=== DLA-argmax accuracy (does the head's OV output alone predict the citation?) ===")
    for hd, acc in zip(critical_heads, critical_dla_accs):
        print(f"  L{hd[0]}H{hd[1]} (critical): DLA-acc={acc:.3f}")
    print(f"  critical heads mean: {np.mean(critical_dla_accs):.3f}")
    print(f"  random non-critical heads mean: {np.mean(random_dla_accs):.3f}")

    print("\n=== Separation analysis (DLA_cited, DLA_margin) by citation outcome, per critical head ===")
    separation_results = {}
    heads_sorted_by_tv = critical_heads  # already sorted by Phase 1 TV descending
    for rank, (l, h) in enumerate(heads_sorted_by_tv):
        contrib = dla[(l, h)]  # [N, 3]
        dla_cited = contrib[np.arange(N), cited_idx]
        other_mask = np.ones_like(contrib, dtype=bool)
        other_mask[np.arange(N), cited_idx] = False
        dla_other_max = contrib[other_mask].reshape(N, 2).max(axis=1)
        dla_margin = dla_cited - dla_other_max
        feats = np.stack([dla_cited, dla_margin], axis=1)

        stats_result, by_outcome_raw = analyze_separation(outcomes_all, feats)
        stats_result["head"] = f"L{l}H{h}"
        stats_result["n_total"] = int(N)
        stats_result["n_per_outcome"] = {o: int((outcomes_all == o).sum()) for o in OUTCOMES}
        stats_result["dla_accuracy"] = critical_dla_accs[rank]
        stats_result["group_means_raw"] = {
            o: {"dla_cited_mean": float(by_outcome_raw[o][:, 0].mean()), "dla_cited_std": float(by_outcome_raw[o][:, 0].std()),
                "dla_margin_mean": float(by_outcome_raw[o][:, 1].mean()), "dla_margin_std": float(by_outcome_raw[o][:, 1].std()),
                "frac_margin_negative": float((by_outcome_raw[o][:, 1] < 0).mean())}
            for o in OUTCOMES
        }
        separation_results[f"L{l}H{h}"] = stats_result
        print(f"\n--- L{l}H{h} ---")
        print(json.dumps(stats_result, indent=2))

        if rank < args.plot_top_k:
            plot_head(l, h, by_outcome_raw, str(Path(args.out_dir) / f"phase3_dla_L{l}H{h}.png"))

    result = {
        "n_instances": int(N),
        "critical_heads": critical_heads,
        "random_heads": random_heads,
        "dla_accuracy": {
            "critical_heads": {f"L{l}H{h}": acc for (l, h), acc in zip(critical_heads, critical_dla_accs)},
            "critical_mean_acc": float(np.mean(critical_dla_accs)),
            "random_heads": {f"L{l}H{h}": acc for (l, h), acc in zip(random_heads, random_dla_accs)},
            "random_mean_acc": float(np.mean(random_dla_accs)),
        },
        "separation": separation_results,
    }
    # per-instance DLA feature export (dla_cited, dla_margin per critical head) for
    # combining with Phase 2c's (A_source, H_source) features in a joint classifier.
    dla_feat_cols = []
    dla_feat_names = []
    for (l, h) in critical_heads:
        contrib = dla[(l, h)]
        dla_cited = contrib[np.arange(N), cited_idx]
        other_mask = np.ones_like(contrib, dtype=bool)
        other_mask[np.arange(N), cited_idx] = False
        dla_other_max = contrib[other_mask].reshape(N, 2).max(axis=1)
        dla_margin = dla_cited - dla_other_max
        dla_feat_cols.extend([dla_cited, dla_margin])
        dla_feat_names.extend([f"L{l}H{h}_DLA_cited", f"L{l}H{h}_DLA_margin"])
    X_dla = np.stack(dla_feat_cols, axis=1)  # [N, 30]
    np.savez(
        str(Path(args.out).with_suffix("")) + "_features.npz",
        X=X_dla, y=outcomes_all, feature_names=np.array(dla_feat_names),
    )
    print(f"Wrote {str(Path(args.out).with_suffix('')) + '_features.npz'}")

    # combined (A_source, H_source, DLA_cited, DLA_margin) 3-way classification,
    # if Phase 2c's matching A/H feature export is available for the same sample.
    ah_path = Path(args.out).parent / "phase2c_multihead_separation_features.npz"
    if ah_path.exists():
        ah = np.load(ah_path, allow_pickle=True)
        X_ah, y_ah, names_ah = ah["X"], ah["y"], ah["feature_names"]
        if len(y_ah) == N and list(y_ah) == list(outcomes_all):
            X_combined = np.concatenate([X_ah, X_dla], axis=1)
            names_combined = list(names_ah) + dla_feat_names

            def cv_bal_acc(clf, X, y):
                scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
                return float(scores.mean()), float(scores.std())

            scaler = StandardScaler()
            comparisons = {
                "A_H_only_30d": X_ah,
                "DLA_only_30d": X_dla,
                "A_H_plus_DLA_60d": X_combined,
            }
            combined_results = {}
            for name, X in comparisons.items():
                Xz = scaler.fit_transform(X)
                lda_acc, lda_std = cv_bal_acc(LinearDiscriminantAnalysis(), Xz, outcomes_all)
                log_acc, log_std = cv_bal_acc(
                    LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1), Xz, outcomes_all
                )
                rf_acc, rf_std = cv_bal_acc(
                    RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=0), X, outcomes_all
                )
                combined_results[name] = {
                    "n_features": X.shape[1],
                    "lda_3way_balanced_acc": lda_acc, "lda_3way_std": lda_std,
                    "logreg_3way_balanced_acc": log_acc, "logreg_3way_std": log_std,
                    "random_forest_3way_balanced_acc": rf_acc, "random_forest_3way_std": rf_std,
                    "chance": 1 / 3,
                }
                print(f"  {name} ({X.shape[1]}d): LDA={lda_acc:.3f} logreg={log_acc:.3f} RF={rf_acc:.3f} (chance=0.333)")

            # also restricted to the Misattribution-vs-Fabrication subset, same comparison
            err_mask = np.isin(outcomes_all, ["MISATTRIBUTION", "FABRICATION"])
            combined_results_err = {}
            for name, X in comparisons.items():
                Xz = scaler.fit_transform(X[err_mask])
                y_err = outcomes_all[err_mask]
                lda_acc, lda_std = cv_bal_acc(LinearDiscriminantAnalysis(), Xz, y_err)
                try:
                    qda_acc, qda_std = cv_bal_acc(QuadraticDiscriminantAnalysis(reg_param=0.5), Xz, y_err)
                except (ValueError, np.linalg.LinAlgError):
                    qda_acc, qda_std = None, None  # n_features > smallest per-fold class size; QDA can't fit regardless of reg_param, and cross_val_score re-raises as ValueError when every fold fails
                rf_acc, rf_std = cv_bal_acc(
                    RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=0), X[err_mask], y_err
                )
                combined_results_err[name] = {
                    "n_features": X.shape[1],
                    "lda_misattr_vs_fabrication_acc": lda_acc,
                    "qda_misattr_vs_fabrication_acc": qda_acc,
                    "rf_misattr_vs_fabrication_acc": rf_acc,
                    "chance": 0.5,
                }
                qda_str = f"{qda_acc:.3f}" if qda_acc is not None else "N/A (rank-deficient)"
                print(f"  [misattr-vs-fab] {name} ({X.shape[1]}d): LDA={lda_acc:.3f} QDA={qda_str} RF={rf_acc:.3f} (chance=0.5)")

            result["combined_feature_comparison"] = {
                "feature_names": names_combined,
                "three_way": combined_results,
                "misattr_vs_fabrication": combined_results_err,
            }
        else:
            print(f"Skipping combined analysis: phase2c features don't match this sample (len {len(y_ah)} vs {N}, or label order differs)")
    else:
        print(f"Skipping combined analysis: {ah_path} not found")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
