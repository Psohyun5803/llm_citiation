"""Phase 2c: can ANY formulation of A_source/H_source separate Misattribution
from Fabrication? Phase 2b's LDA already searched all linear combinations of
(A_source, H_source) for a single head and found exact chance -- so no new
linear formula on those two numbers will help. This script checks the two
remaining possibilities:

  1. Variance/shape difference: Levene's test on A_source and H_source
     separately (do the two error types differ in spread even if their means
     coincide?), and a QDA classifier (which allows per-class covariance,
     unlike LDA) -- if QDA beats LDA, there's quadratic/variance structure a
     linear formula can't capture.
  2. Joint information across all 15 Phase-1 critical heads: concatenate
     (A_source, H_source) for all 15 heads into one 30-dim vector per
     instance, plus pairwise product/ratio engineered features, and test
     logistic regression / QDA / random forest with cross-validation.

Reuses the same 657-instance sample (300 Correct, 300 Misattribution, all 57
Fabrication, seed=0) as Phase 2/2b for direct comparability, but restricts
the actual classification tests to the Misattribution-vs-Fabrication subset.

Writes results/phase2c_multihead_separation.json
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

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

    return {"ids": ids, "target_pos": target_pos, "spans": spans, "citation_outcome": row["citation_outcome"]}


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
    ap.add_argument("--out", default=str(ROOT / "results/phase2c_multihead_separation.json"))
    ap.add_argument("--n-correct", type=int, default=300)
    ap.add_argument("--n-misattribution", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    phase1 = json.load(open(args.phase1))
    target_heads = [tuple(h) for h in phase1["validation"]["critical_heads"]]
    print(f"{len(target_heads)} heads: {target_heads}")

    print("Loading model...")
    model = load_model(args.device)
    tok = model.tokenizer

    rows = load_sample(args.generations, args.seed, args.n_correct, args.n_misattribution)
    instances = [inst for r in rows if (inst := build_instance(tok, r)) is not None]
    print(f"{len(instances)} instances: {Counter(i['citation_outcome'] for i in instances)}")

    pad_id = tok.eos_token_id
    critical_layers = sorted({l for l, h in target_heads})
    heads_by_layer = defaultdict(list)
    for l, h in target_heads:
        heads_by_layer[l].append(h)

    # feature matrix: [n_instances, n_heads * 2] (A_source, H_source per head)
    head_order = target_heads
    feat_rows = []
    outcomes = []

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

        for i, inst in enumerate(chunk):
            feats = []
            for (l, h) in head_order:
                row_attn = captured[l][i][h]
                raw = np.array([row_attn[inst["spans"][k][0] : inst["spans"][k][1]].sum() for k in (1, 2, 3)])
                a_source = float(raw.sum())
                if a_source > 1e-8:
                    p = raw / a_source
                    h_source = float(-np.sum(p * np.log(p + 1e-12)))
                else:
                    h_source = float(np.log(3))
                feats.extend([a_source, h_source])
            feat_rows.append(feats)
            outcomes.append(inst["citation_outcome"])
        print(f"  processed {min(start + args.batch_size, len(instances))}/{len(instances)}")

    X_all = np.array(feat_rows)  # [N, 30]
    y_all = np.array(outcomes)
    feature_names = [f"L{l}H{h}_{ax}" for (l, h) in head_order for ax in ("A", "H")]
    np.savez(
        str(Path(args.out).with_suffix("")) + "_features.npz",
        X=X_all, y=y_all, feature_names=np.array(feature_names),
    )
    print(f"Wrote {str(Path(args.out).with_suffix('')) + '_features.npz'}")

    # restrict to the Misattribution-vs-Fabrication question
    err_mask = np.isin(y_all, ["MISATTRIBUTION", "FABRICATION"])
    X_err, y_err = X_all[err_mask], y_all[err_mask]
    print(f"\nMisattribution vs Fabrication subset: {Counter(y_err)}")

    scaler = StandardScaler()
    X_err_z = scaler.fit_transform(X_err)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = {"n_misattribution": int((y_err == "MISATTRIBUTION").sum()), "n_fabrication": int((y_err == "FABRICATION").sum())}

    def cv_acc(clf, X, y):
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
        return float(scores.mean()), float(scores.std())

    # (1) per-head QDA vs LDA, and Levene's test for variance equality
    per_head = {}
    for idx, (l, h) in enumerate(head_order):
        a_vals = X_err[:, idx * 2]
        h_vals = X_err[:, idx * 2 + 1]
        mis_mask = y_err == "MISATTRIBUTION"
        lev_a = stats.levene(a_vals[mis_mask], a_vals[~mis_mask])
        lev_h = stats.levene(h_vals[mis_mask], h_vals[~mis_mask])
        xi = X_err_z[:, idx * 2 : idx * 2 + 2]
        lda_acc, lda_std = cv_acc(LinearDiscriminantAnalysis(), xi, y_err)
        qda_acc, qda_std = cv_acc(QuadraticDiscriminantAnalysis(reg_param=0.3), xi, y_err)
        per_head[f"L{l}H{h}"] = {
            "levene_A_p": float(lev_a.pvalue),
            "levene_H_p": float(lev_h.pvalue),
            "lda_acc": lda_acc,
            "qda_acc": qda_acc,
        }
    results["per_head"] = per_head
    best_lda = max(per_head.items(), key=lambda kv: kv[1]["lda_acc"])
    best_qda = max(per_head.items(), key=lambda kv: kv[1]["qda_acc"])
    print(f"best single-head LDA: {best_lda}")
    print(f"best single-head QDA: {best_qda}")

    # (2) all 15 heads combined (30-dim): logistic regression, QDA (regularized), random forest
    log_acc, log_std = cv_acc(LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1), X_err_z, y_err)
    qda_all_acc, qda_all_std = cv_acc(QuadraticDiscriminantAnalysis(reg_param=0.5), X_err_z, y_err)
    rf_acc, rf_std = cv_acc(
        RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=0), X_err, y_err
    )
    results["all_heads_combined"] = {
        "n_features": X_all.shape[1],
        "logistic_regression_acc": log_acc,
        "logistic_regression_std": log_std,
        "qda_regularized_acc": qda_all_acc,
        "qda_regularized_std": qda_all_std,
        "random_forest_acc": rf_acc,
        "random_forest_std": rf_std,
        "chance": 0.5,
    }
    print(f"\nAll 15 heads combined (30-dim): logreg={log_acc:.3f} qda={qda_all_acc:.3f} rf={rf_acc:.3f} (chance=0.5)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
