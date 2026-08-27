"""Phase 2.1 (pilot): run saved residual-stream activations through the
Gemma Scope SAE and build a token-level feature table.

Reads:
  data/generated_citations/token_labels/{prompt_id}.json  (from parse.py)
  data/generated_citations/activations/{prompt_id}.pt      (from generate.py)
  data/verified_citations/verified_citations.jsonl         (from verify.py)

Writes:
  results/sae_features_pilot.pt
      {
        "features": FloatTensor [N, d_sae],   # sparse SAE activations
        "meta": list[dict]                     # one row per feature vector
      }
  meta rows carry: prompt_id, domain, citation_idx, citation_field
  (author/title/venue/year), token_idx, citation_label (real/partial/
  fabricated), and field_correct (bool) for that specific citation field.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from sae_lens import SAE

ROOT = Path(__file__).resolve().parent.parent
LAYER = 20
SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
SAE_ID = f"layer_{LAYER}/width_16k/canonical"


def load_sae(device: str):
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    return sae.to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-labels-dir", default=str(ROOT / "data/generated_citations/token_labels"))
    ap.add_argument("--activations-dir", default=str(ROOT / "data/generated_citations/activations"))
    ap.add_argument("--verified", default=str(ROOT / "data/verified_citations/verified_citations.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/sae_features_pilot.pt"))
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    verified = [json.loads(l) for l in open(args.verified)]
    verified_by_key = {(v["prompt_id"], v["citation_idx"]): v for v in verified}

    sae = load_sae(args.device)

    all_feats = []
    all_meta = []

    token_labels_dir = Path(args.token_labels_dir)
    for label_path in sorted(token_labels_dir.glob("*.json")):
        prompt_id = label_path.stem
        token_labels = json.load(open(label_path))
        if not token_labels:
            continue

        act = torch.load(Path(args.activations_dir) / f"{prompt_id}.pt", weights_only=False)
        resid = act["resid_post"].to(args.device)  # [seq_len, d_model]

        with torch.no_grad():
            feats = sae.encode(resid)  # [seq_len, d_sae]

        for tl in token_labels:
            key = (prompt_id, tl["citation_idx"])
            v = verified_by_key.get(key)
            if v is None or v["citation_label"] == "unverified_api_error":
                continue
            field_correct = v["field_correct"][tl["field"]]
            if field_correct is None:  # ground truth unknown for this field (e.g. arXiv-only venue record)
                continue
            all_feats.append(feats[tl["token_idx"]].cpu())
            all_meta.append(
                {
                    "prompt_id": prompt_id,
                    "citation_idx": tl["citation_idx"],
                    "token_idx": tl["token_idx"],
                    "citation_field": tl["field"],
                    "domain": v["domain"],
                    "citation_label": v["citation_label"],
                    "field_correct": field_correct,
                }
            )

    feats_tensor = torch.stack(all_feats) if all_feats else torch.empty(0, sae.cfg.d_sae)
    torch.save({"features": feats_tensor, "meta": all_meta, "layer": LAYER, "sae_id": SAE_ID}, args.out)

    print(f"Extracted {feats_tensor.shape[0]} labeled token feature vectors (d_sae={feats_tensor.shape[-1]})")
    from collections import Counter

    print("citation_field counts:", Counter(m["citation_field"] for m in all_meta))
    print("citation_label counts:", Counter(m["citation_label"] for m in all_meta))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
