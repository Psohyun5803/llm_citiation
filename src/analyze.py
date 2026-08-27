"""Phase 2.2/2.3 (pilot): field prediction from SAE features + a first look
at hallucination-associated features.

Reads results/sae_features_pilot.pt (from extract_features.py).

This is a pilot-scale run (tens of citations, not the full 200-300 prompt
protocol), so the numbers here are a sanity check that the pipeline and
analysis machinery work end to end -- not a statistically powered result.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["author", "title", "venue", "year"]


def grouped_split(meta: list[dict], train_frac: float = 0.8):
    prompt_ids = sorted({m["prompt_id"] for m in meta})
    n_train = max(1, int(len(prompt_ids) * train_frac))
    train_ids = set(prompt_ids[:n_train])
    train_idx = [i for i, m in enumerate(meta) if m["prompt_id"] in train_ids]
    test_idx = [i for i, m in enumerate(meta) if m["prompt_id"] not in train_ids]
    return train_idx, test_idx


def field_prediction(features: np.ndarray, meta: list[dict], out_dir: Path) -> None:
    y = np.array([m["citation_field"] for m in meta])
    prompt_ids = sorted({m["prompt_id"] for m in meta})

    print(f"\n=== Field prediction (4-class: {FIELDS}) ===")
    print(f"n={len(y)} labeled tokens from {len(prompt_ids)} prompts; class counts: {Counter(y.tolist())}")

    train_idx, test_idx = grouped_split(meta)
    held_out = len(test_idx) > 0 and len(set(y[test_idx])) > 0

    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(features[train_idx], y[train_idx])

    if held_out:
        y_pred = clf.predict(features[test_idx])
        y_true = y[test_idx]
        split_desc = f"held-out test set ({len(test_idx)} tokens from prompts not seen in training)"
    else:
        # pilot sample too small for a meaningful prompt-level held-out split
        y_pred = clf.predict(features[train_idx])
        y_true = y[train_idx]
        split_desc = f"TRAIN SET ONLY (pilot sample too small for a held-out split) -- {len(train_idx)} tokens"

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    majority = Counter(y_true.tolist()).most_common(1)[0][1] / len(y_true)

    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)

    print(f"Evaluated on: {split_desc}")
    print(f"accuracy={acc:.3f}  macro_f1={f1:.3f}  majority_baseline={majority:.3f}  random_baseline={1/len(FIELDS):.3f}")
    print(f"labels: {labels_present}")
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    result = {
        "n_tokens": len(y),
        "n_prompts": len(prompt_ids),
        "class_counts": dict(Counter(y.tolist())),
        "split": split_desc,
        "accuracy": acc,
        "macro_f1": f1,
        "majority_baseline": majority,
        "random_baseline": 1 / len(FIELDS),
        "confusion_matrix": {"labels": labels_present, "matrix": cm.tolist()},
    }
    with open(out_dir / "field_prediction_pilot.json", "w") as f:
        json.dump(result, f, indent=2)


def selectivity_index(features: np.ndarray, meta: list[dict], out_dir: Path, top_k: int = 10) -> None:
    print("\n=== Field selectivity index ===")
    y = np.array([m["citation_field"] for m in meta])
    overall_mean = features.mean(axis=0) + 1e-8

    report = {}
    for field in FIELDS:
        mask = y == field
        if mask.sum() == 0:
            continue
        field_mean = features[mask].mean(axis=0)
        selectivity = field_mean / overall_mean
        top = np.argsort(-selectivity)[:top_k]
        report[field] = [{"feature_idx": int(i), "selectivity": float(selectivity[i])} for i in top if features[mask][:, i].mean() > 0]
        print(f"{field} (n={mask.sum()}): top feature idxs {[r['feature_idx'] for r in report[field]][:5]}")

    with open(out_dir / "selectivity_pilot.json", "w") as f:
        json.dump(report, f, indent=2)


def hallucination_signal(features: np.ndarray, meta: list[dict], out_dir: Path) -> None:
    print("\n=== Correct vs. incorrect field activation (pilot) ===")
    report = {}
    for field in FIELDS:
        idx = [i for i, m in enumerate(meta) if m["citation_field"] == field]
        if not idx:
            continue
        correct = [i for i in idx if meta[i]["field_correct"]]
        incorrect = [i for i in idx if not meta[i]["field_correct"]]
        print(f"{field}: correct={len(correct)} incorrect={len(incorrect)}")
        report[field] = {"n_correct": len(correct), "n_incorrect": len(incorrect)}
        if len(correct) >= 3 and len(incorrect) >= 3:
            from scipy.stats import mannwhitneyu

            pvals = []
            for feat_idx in range(features.shape[1]):
                c = features[correct, feat_idx]
                inc = features[incorrect, feat_idx]
                if c.sum() == 0 and inc.sum() == 0:
                    continue
                try:
                    _, p = mannwhitneyu(c, inc, alternative="two-sided")
                    pvals.append((feat_idx, float(p)))
                except ValueError:
                    continue
            pvals.sort(key=lambda x: x[1])
            report[field]["top_diff_features"] = pvals[:10]
            print(f"  top differentiating feature idx/p-value: {pvals[:5]}")
        else:
            print("  too few examples in one class for a significance test (pilot-scale limitation)")

    with open(out_dir / "hallucination_signal_pilot.json", "w") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "results/sae_features_pilot.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.features, weights_only=False)
    features = data["features"].numpy()
    meta = data["meta"]

    if features.shape[0] == 0:
        print("No labeled feature vectors found -- nothing to analyze.")
        return

    field_prediction(features, meta, out_dir)
    selectivity_index(features, meta, out_dir)
    hallucination_signal(features, meta, out_dir)


if __name__ == "__main__":
    main()
