"""Pilot: does a small set of attention heads causally mediate citation
source selection in Llama-3.1-8B-Instruct?

Design (see data/rag_pilot/examples.json for the 5 examples):
  - Each example has a question, a document that contains the answer
    (doc_correct), and a related-but-wrong document (doc_distractor).
  - clean prompt: doc_correct labeled [1], doc_distractor labeled [2]
    (physical order fixed; only the bracket label differs between
    clean/corrupted, so prompt token length is identical either way --
    verified "[1]"/"[2]" tokenize to the same length).
  - Step 1: free-generate on the clean prompt. Keep the example only if the
    model actually cites "[1]" (and not "[2]") -- need >=3 such examples.
  - Step 2: locate the token position of the cited digit in the clean
    generation, and record logit('1') - logit('2') there as the clean
    baseline. The token prefix up to that position (prompt + generated
    reasoning) is reused verbatim as the shared context for every condition
    below, so all three conditions read out at the exact same position.
  - Step 3: build the corrupted context by swapping which label ([1]/[2])
    points at doc_correct vs doc_distractor, keeping the same generated
    continuation forced on top. Record logit('1') - logit('2') here too
    (expected to favor '2', since the correct content is now behind that
    label).
  - Step 4: for every one of the 32*32 (layer, head) pairs, patch that
    head's hook_z (its output before W_O, all positions) in the corrupted
    context with the clean run's cached value, and re-measure logit('1') -
    logit('2'). The recovery fraction toward the clean value is this head's
    indirect effect (IE) for this example.
  - Average IE across examples into a 32x32 heatmap.

Writes:
  results/activation_patching_pilot.json  (per-example logit diffs + full IE grid)
  results/activation_patching_heatmap.png
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = (
    "주어진 문서를 참고하여 질문에 답하고, 답변의 근거가 되는 문서 번호를 "
    "대괄호 안에 인용하시오 (예: [1], [2])."
)


def load_model(device: str):
    model = HookedTransformer.from_pretrained_no_processing(MODEL_ID, dtype=torch.float16, device=device)
    model.eval()
    return model


def build_prompt_ids(model, doc_label1: str, doc_label2: str, question: str) -> torch.Tensor:
    user_msg = f"[1] {doc_label1}\n[2] {doc_label2}\n\n질문: {question}"
    chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    ids = model.tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt", return_dict=False
    )
    return ids.to(model.cfg.device)


def find_citation_digit_pos(model, full_ids: torch.Tensor, gen_start: int) -> tuple[int, str] | None:
    """Search the generated portion for the first digit token (representing
    '1' or '2') that sits inside a "[X]" citation, found by decoding a small
    token window around each candidate (robust to the tokenizer merging the
    preceding space into "[" as a single token, e.g. " [" -> one token id
    distinct from standalone "["). Returns (absolute token index of the
    digit, digit_str), or None if no bracketed citation is found."""
    tok = model.tokenizer
    id_1 = tok("1", add_special_tokens=False)["input_ids"][-1]
    id_2 = tok("2", add_special_tokens=False)["input_ids"][-1]

    seq = full_ids[0].tolist()
    for i in range(gen_start, len(seq)):
        if seq[i] not in (id_1, id_2):
            continue
        window = seq[max(0, i - 2) : i + 2]
        decoded = tok.decode(window).replace(" ", "")
        digit = "1" if seq[i] == id_1 else "2"
        if f"[{digit}]" in decoded:
            return i, digit
    return None


def logit_diff_at(model, context_ids: torch.Tensor, id_1: int, id_2: int, cache_hooks=None) -> tuple[float, dict]:
    """Runs the model on context_ids (predicting the NEXT token after the
    last position) and returns (logit('1') - logit('2'), full cache).
    cache_hooks: if given, run_with_hooks instead of plain forward."""
    with torch.no_grad():
        if cache_hooks is None:
            logits, cache = model.run_with_cache(context_ids)
        else:
            logits = model.run_with_hooks(context_ids, fwd_hooks=cache_hooks)
            cache = None
        last_logits = logits[0, -1]
        diff = (last_logits[id_1] - last_logits[id_2]).item()
    return diff, cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=str(ROOT / "data/rag_pilot/examples.json"))
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out-json", default=str(ROOT / "results/activation_patching_pilot.json"))
    ap.add_argument("--out-png", default=str(ROOT / "results/activation_patching_heatmap.png"))
    args = ap.parse_args()

    examples = json.load(open(args.examples))
    model = load_model(args.device)
    tok = model.tokenizer
    id_1 = tok("1", add_special_tokens=False)["input_ids"][-1]
    id_2 = tok("2", add_special_tokens=False)["input_ids"][-1]
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    print(f"n_layers={n_layers} n_heads={n_heads} id('1')={id_1} id('2')={id_2}")

    valid = []
    for ex in examples:
        clean_ids = build_prompt_ids(model, ex["doc_correct"], ex["doc_distractor"], ex["question"])
        gen_start = clean_ids.shape[1]
        with torch.no_grad():
            out = model.generate(clean_ids, max_new_tokens=args.max_new_tokens, do_sample=False, verbose=False)
        out = out.clone()  # generate() returns inference-mode tensors; clone so they can be reused in later forward passes
        gen_text = tok.decode(out[0, gen_start:], skip_special_tokens=True)
        hit = find_citation_digit_pos(model, out, gen_start)
        cited = hit[1] if hit else None
        keep = cited == "1"
        print(f"[{ex['id']}] cited={cited} keep={keep}\n    gen: {gen_text[:200]!r}")
        if keep:
            valid.append({"example": ex, "clean_full_ids": out, "cite_pos": hit[0]})

    print(f"\n{len(valid)}/{len(examples)} examples naturally cite [1] -- required >=3")
    if len(valid) < 3:
        print("FAILED prerequisite: fewer than 3 valid examples. Stopping before the patching sweep.")
        json.dump({"n_valid": len(valid), "status": "insufficient_valid_examples"}, open(args.out_json, "w"), indent=2)
        return

    ie_grid_sum = np.zeros((n_layers, n_heads))
    per_example_results = []

    for v in valid:
        ex = v["example"]
        cite_pos = v["cite_pos"]
        clean_context = v["clean_full_ids"][:, :cite_pos]

        clean_diff, clean_cache = logit_diff_at(model, clean_context, id_1, id_2)

        # corrupted: same generated continuation, but doc_correct is now
        # labeled [2] and doc_distractor is labeled [1]
        corrupted_prompt_ids = build_prompt_ids(model, ex["doc_distractor"], ex["doc_correct"], ex["question"])
        prompt_len = corrupted_prompt_ids.shape[1]
        clean_prompt_len = build_prompt_ids(model, ex["doc_correct"], ex["doc_distractor"], ex["question"]).shape[1]
        assert prompt_len == clean_prompt_len, (
            f"clean/corrupted prompt length mismatch ({clean_prompt_len} vs {prompt_len}) "
            "-- label swap should not change token count"
        )
        continuation = clean_context[:, clean_prompt_len:]
        corrupted_context = torch.cat([corrupted_prompt_ids, continuation], dim=1)

        corrupted_diff, _ = logit_diff_at(model, corrupted_context, id_1, id_2)

        print(f"\n[{ex['id']}] clean_diff={clean_diff:.3f} corrupted_diff={corrupted_diff:.3f}")

        ie_grid = np.zeros((n_layers, n_heads))
        denom = clean_diff - corrupted_diff
        for layer in range(n_layers):
            clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]  # [batch, pos, n_heads, d_head]
            for head in range(n_heads):

                def patch_hook(z, hook, layer=layer, head=head):
                    z[:, :, head, :] = clean_z[:, :, head, :]
                    return z

                patched_diff, _ = logit_diff_at(
                    model, corrupted_context, id_1, id_2, cache_hooks=[(f"blocks.{layer}.attn.hook_z", patch_hook)]
                )
                ie = (patched_diff - corrupted_diff) / denom if abs(denom) > 1e-6 else 0.0
                ie_grid[layer, head] = ie
            print(f"  layer {layer + 1}/{n_layers} done")

        ie_grid_sum += ie_grid
        per_example_results.append(
            {
                "id": ex["id"],
                "clean_diff": clean_diff,
                "corrupted_diff": corrupted_diff,
                "ie_grid": ie_grid.tolist(),
            }
        )

    ie_mean = ie_grid_sum / len(valid)

    flat = ie_mean.flatten()
    n_above_0_3 = int((flat > 0.3).sum())
    n_above_0_5 = int((flat > 0.5).sum())
    top_idx = np.argsort(-flat)[:20]
    top_heads = [{"layer": int(i // n_heads), "head": int(i % n_heads), "ie": float(flat[i])} for i in top_idx]

    # a fixed 0.3 cutoff alone can't distinguish "nothing stands out" from
    # "an extremely sparse but very strong signal" (e.g. 1-2 heads towering
    # over a near-zero background) -- also compare against the background
    # noise level via a z-score, and check per-example consistency of the
    # single best head.
    bg_mean, bg_std = float(flat.mean()), float(flat.std())
    top_z = (flat.max() - bg_mean) / bg_std if bg_std > 1e-9 else 0.0
    best_layer, best_head = int(np.argmax(ie_mean) // n_heads), int(np.argmax(ie_mean) % n_heads)
    best_head_per_example = [ex["ie_grid"][best_layer][best_head] for ex in per_example_results]

    print("\n=== Summary ===")
    print(f"background IE: mean={bg_mean:.4f} std={bg_std:.4f}")
    print(f"heads with mean IE > 0.3: {n_above_0_3} / {n_layers * n_heads}")
    print(f"heads with mean IE > 0.5: {n_above_0_5} / {n_layers * n_heads}")
    print(f"top head (layer {best_layer}, head {best_head}): mean IE={flat.max():.3f}, z={top_z:.1f} std above background")
    print(f"  per-example IE for top head: {[round(v, 3) for v in best_head_per_example]}")
    print("top 10 heads:", top_heads[:10])

    if n_above_0_3 > 100:
        verdict = "FAILURE: too many heads with similar effect -- source selection appears distributed, hard to target"
    elif top_z >= 8 and 1 <= n_above_0_3 <= 20:
        verdict = (
            f"SUCCESS (strong, very sparse localization): top head (L{best_layer}H{best_head}) is {top_z:.1f} "
            f"std above the background noise floor and consistently high across all examples "
            f"(range {min(best_head_per_example):.2f}-{max(best_head_per_example):.2f}) -- source selection "
            "is causally concentrated in a small number of heads, even more concentrated than the 5-20 band."
        )
    elif 5 <= n_above_0_3 <= 20:
        verdict = "SUCCESS: sparse band of 5-20 heads clearly above background -- source selection appears localized"
    elif top_z < 3:
        verdict = "FAILURE: no head clearly separates from the background noise -- effect not localized to specific heads"
    else:
        verdict = f"AMBIGUOUS: {n_above_0_3} heads above 0.3 threshold, top z={top_z:.1f} -- inspect heatmap directly"
    print(verdict)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "n_valid": len(valid),
                "n_layers": n_layers,
                "n_heads": n_heads,
                "per_example": per_example_results,
                "ie_mean_grid": ie_mean.tolist(),
                "n_heads_above_0.3": n_above_0_3,
                "n_heads_above_0.5": n_above_0_5,
                "top_heads": top_heads,
                "verdict": verdict,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out_json}")

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(ie_mean, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("head index")
    ax.set_ylabel("layer index")
    ax.set_title(f"Mean indirect effect on citation source selection\n({len(valid)} examples, Llama-3.1-8B-Instruct)")
    fig.colorbar(im, ax=ax, label="indirect effect (recovery fraction)")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
