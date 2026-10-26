"""Stress-test the L13H18 finding from activation_patching_pilot.py.

A single head recovering 77-97% of the clean-vs-corrupted logit gap across
12 diverse examples is a strikingly clean result -- clean enough to be
suspicious. Patching only shows SUFFICIENCY (putting clean's value back in
fixes a corrupted run). This script checks two independent things:

  1. NECESSITY: if we ablate (zero, and separately mean-ablate) L13H18 in
     the plain CLEAN run -- no corruption at all -- does logit('1')-logit('2')
     collapse? If the head only mattered for patching corrupted runs but the
     clean run is fine without it, the "this head IS citation selection"
     story is much weaker.
  2. WHAT IT ATTENDS TO: at the exact token position where the model is
     about to emit the citation digit, what does L13H18's attention pattern
     actually look like? If it attends to the bracket-label token or the
     correct document's content, that's mechanistically coherent with a
     source-selection/label-copying role. If it attends to something
     unrelated (e.g. always the first token), the patching effect may be
     driven by something else entirely.

Writes results/head_l13h18_validation.json
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LAYER, HEAD = 13, 18
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


def find_citation_digit_pos(model, full_ids: torch.Tensor, gen_start: int):
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


def main() -> None:
    device = "cuda:3"
    examples = json.load(open(ROOT / "data/rag_pilot/examples.json"))
    model = load_model(device)
    tok = model.tokenizer
    id_1 = tok("1", add_special_tokens=False)["input_ids"][-1]
    id_2 = tok("2", add_special_tokens=False)["input_ids"][-1]

    results = []
    for ex in examples:
        clean_ids = build_prompt_ids(model, ex["doc_correct"], ex["doc_distractor"], ex["question"])
        gen_start = clean_ids.shape[1]
        with torch.no_grad():
            out = model.generate(clean_ids, max_new_tokens=120, do_sample=False, verbose=False)
        out = out.clone()
        hit = find_citation_digit_pos(model, out, gen_start)
        if hit is None or hit[1] != "1":
            print(f"[{ex['id']}] skipped (did not cite [1])")
            continue
        cite_pos = hit[0]
        clean_context = out[:, :cite_pos]

        with torch.no_grad():
            logits, cache = model.run_with_cache(clean_context)
            clean_diff = (logits[0, -1, id_1] - logits[0, -1, id_2]).item()

            head_z = cache[f"blocks.{LAYER}.attn.hook_z"][:, :, HEAD, :]  # [1, pos, d_head]
            mean_z = head_z.mean(dim=1, keepdim=True)  # mean over sequence positions

            def zero_ablate(z, hook):
                z[:, :, HEAD, :] = 0.0
                return z

            def mean_ablate(z, hook):
                z[:, :, HEAD, :] = mean_z
                return z

            zero_logits = model.run_with_hooks(clean_context, fwd_hooks=[(f"blocks.{LAYER}.attn.hook_z", zero_ablate)])
            zero_diff = (zero_logits[0, -1, id_1] - zero_logits[0, -1, id_2]).item()

            mean_logits = model.run_with_hooks(clean_context, fwd_hooks=[(f"blocks.{LAYER}.attn.hook_z", mean_ablate)])
            mean_diff = (mean_logits[0, -1, id_1] - mean_logits[0, -1, id_2]).item()

            # attention pattern of L13H18 at the decision position (last token of clean_context)
            pattern = cache[f"blocks.{LAYER}.attn.hook_pattern"][0, HEAD, -1, :]  # [key_pos]
            top_vals, top_idx = torch.topk(pattern, k=min(8, pattern.shape[0]))
            token_strs = [tok.decode([clean_context[0, i].item()]) for i in top_idx.tolist()]

        print(
            f"[{ex['id']}] clean_diff={clean_diff:.3f} zero_ablated={zero_diff:.3f} mean_ablated={mean_diff:.3f}"
        )
        print(f"  top attended tokens (L{LAYER}H{HEAD} @ decision pos): "
              + ", ".join(f"{t!r}({v:.2f})" for t, v in zip(token_strs, top_vals.tolist())))

        results.append(
            {
                "id": ex["id"],
                "clean_diff": clean_diff,
                "zero_ablated_diff": zero_diff,
                "mean_ablated_diff": mean_diff,
                "top_attended_tokens": [{"token": t, "weight": v} for t, v in zip(token_strs, top_vals.tolist())],
            }
        )

    import statistics

    clean_vals = [r["clean_diff"] for r in results]
    zero_vals = [r["zero_ablated_diff"] for r in results]
    mean_vals = [r["mean_ablated_diff"] for r in results]
    print("\n=== Necessity summary ===")
    print(f"clean_diff:        mean={statistics.mean(clean_vals):.3f}")
    print(f"zero-ablated diff: mean={statistics.mean(zero_vals):.3f}")
    print(f"mean-ablated diff: mean={statistics.mean(mean_vals):.3f}")
    n_flipped_zero = sum(1 for c, z in zip(clean_vals, zero_vals) if c > 0 and z < 0)
    n_flipped_mean = sum(1 for c, m in zip(clean_vals, mean_vals) if c > 0 and m < 0)
    print(f"examples where zero-ablation FLIPS sign (was citing [1], now favors [2]): {n_flipped_zero}/{len(results)}")
    print(f"examples where mean-ablation FLIPS sign: {n_flipped_mean}/{len(results)}")

    Path(ROOT / "results/head_l13h18_validation.json").write_text(
        json.dumps(
            {
                "layer": LAYER,
                "head": HEAD,
                "results": results,
                "mean_clean_diff": statistics.mean(clean_vals),
                "mean_zero_ablated_diff": statistics.mean(zero_vals),
                "mean_mean_ablated_diff": statistics.mean(mean_vals),
                "n_flipped_zero_ablation": n_flipped_zero,
                "n_flipped_mean_ablation": n_flipped_mean,
            },
            indent=2,
        )
    )
    print("\nWrote results/head_l13h18_validation.json")


if __name__ == "__main__":
    main()
