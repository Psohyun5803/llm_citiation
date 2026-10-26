"""Follow-up to validate_head_l13h18.py: the earlier check only looked at
L13H18's attention pattern at the FINAL (citation-digit) position, which
turned out to be dominated by punctuation/bracket/BOS tokens -- not
obviously "semantic". Since hook_z patching was applied across ALL
positions, the real causal action may be happening earlier: specifically at
the "[1]"/"[2]" label-digit positions in the prompt, where each document is
first tagged with its number.

This script, for each valid example:
  1. Locates the digit-token positions of "[1]" and "[2]" IN THE PROMPT
     (not the generated citation).
  2. Prints L13H18's attention pattern at each of those two positions --
     what is the head looking at when it first "sees" each label?
  3. Re-runs the activation-patching sufficiency test but restricted to
     patching hook_z ONLY at those two label positions (instead of the
     whole sequence), and compares the recovered indirect effect to the
     full-sequence-patch IE already on record. If restricted patching
     reproduces most of the effect, the label positions are where the
     causally relevant computation happens.

Writes results/head_label_position_validation.json
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


def find_digit_positions(model, ids: torch.Tensor, start: int, end: int, want_digits=("1", "2")) -> dict[str, int]:
    """Finds the first "[X]" occurrence for each digit in want_digits within
    ids[0, start:end]."""
    tok = model.tokenizer
    id_map = {d: tok(d, add_special_tokens=False)["input_ids"][-1] for d in want_digits}
    seq = ids[0].tolist()
    found = {}
    for i in range(start, end):
        for d, tid in id_map.items():
            if d in found:
                continue
            if seq[i] != tid:
                continue
            window = seq[max(0, i - 2) : i + 2]
            decoded = tok.decode(window).replace(" ", "")
            if f"[{d}]" in decoded:
                found[d] = i
        if len(found) == len(want_digits):
            break
    return found


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
    device = "cuda:1"
    examples = json.load(open(ROOT / "data/rag_pilot/examples.json"))
    model = load_model(device)
    tok = model.tokenizer
    id_1 = tok("1", add_special_tokens=False)["input_ids"][-1]
    id_2 = tok("2", add_special_tokens=False)["input_ids"][-1]

    results = []
    for ex in examples:
        clean_prompt_ids = build_prompt_ids(model, ex["doc_correct"], ex["doc_distractor"], ex["question"])
        gen_start = clean_prompt_ids.shape[1]
        with torch.no_grad():
            out = model.generate(clean_prompt_ids, max_new_tokens=120, do_sample=False, verbose=False)
        out = out.clone()
        hit = find_citation_digit_pos(model, out, gen_start)
        if hit is None or hit[1] != "1":
            print(f"[{ex['id']}] skipped (did not cite [1])")
            continue
        cite_pos = hit[0]
        clean_context = out[:, :cite_pos]

        label_pos = find_digit_positions(model, clean_context, 0, gen_start)
        if len(label_pos) != 2:
            print(f"[{ex['id']}] could not locate both label positions, skipping")
            continue

        corrupted_prompt_ids = build_prompt_ids(model, ex["doc_distractor"], ex["doc_correct"], ex["question"])
        continuation = clean_context[:, gen_start:]
        corrupted_context = torch.cat([corrupted_prompt_ids, continuation], dim=1)
        corrupted_label_pos = find_digit_positions(model, corrupted_context, 0, gen_start)

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_context)
            clean_diff = (clean_logits[0, -1, id_1] - clean_logits[0, -1, id_2]).item()
            corrupted_logits = model(corrupted_context)
            corrupted_diff = (corrupted_logits[0, -1, id_1] - corrupted_logits[0, -1, id_2]).item()

            # attention pattern at each label position
            pattern = clean_cache[f"blocks.{LAYER}.attn.hook_pattern"][0, HEAD]  # [query_pos, key_pos]
            label_attn = {}
            for d, pos in label_pos.items():
                vals, idx = torch.topk(pattern[pos], k=min(6, pos + 1))
                toks = [tok.decode([clean_context[0, i].item()]) for i in idx.tolist()]
                label_attn[d] = [{"token": t, "weight": v} for t, v in zip(toks, vals.tolist())]

            clean_z = clean_cache[f"blocks.{LAYER}.attn.hook_z"][:, :, HEAD, :]

            # restricted patch: only at the two label positions (clean -> corrupted)
            restrict_positions = list(corrupted_label_pos.values())

            def patch_restricted(z, hook):
                for p_clean, p_corrupt in zip(label_pos.values(), corrupted_label_pos.values()):
                    z[:, p_corrupt, HEAD, :] = clean_z[:, p_clean, :]
                return z

            def patch_full(z, hook):
                z[:, :, HEAD, :] = clean_z[:, :, :]
                return z

            restricted_logits = model.run_with_hooks(
                corrupted_context, fwd_hooks=[(f"blocks.{LAYER}.attn.hook_z", patch_restricted)]
            )
            restricted_diff = (restricted_logits[0, -1, id_1] - restricted_logits[0, -1, id_2]).item()

            full_logits = model.run_with_hooks(
                corrupted_context, fwd_hooks=[(f"blocks.{LAYER}.attn.hook_z", patch_full)]
            )
            full_diff = (full_logits[0, -1, id_1] - full_logits[0, -1, id_2]).item()

        denom = clean_diff - corrupted_diff
        ie_restricted = (restricted_diff - corrupted_diff) / denom if abs(denom) > 1e-6 else 0.0
        ie_full = (full_diff - corrupted_diff) / denom if abs(denom) > 1e-6 else 0.0

        print(f"\n[{ex['id']}] clean={clean_diff:.2f} corrupted={corrupted_diff:.2f} "
              f"IE_restricted(2 label positions only)={ie_restricted:.3f} IE_full(all positions)={ie_full:.3f}")
        for d in ("1", "2"):
            print(f"  attention at label [{d}] position: {label_attn[d]}")

        results.append(
            {
                "id": ex["id"],
                "clean_diff": clean_diff,
                "corrupted_diff": corrupted_diff,
                "ie_restricted_label_positions": ie_restricted,
                "ie_full_sequence": ie_full,
                "label_attention": label_attn,
            }
        )

    import statistics

    ie_r = [r["ie_restricted_label_positions"] for r in results]
    ie_f = [r["ie_full_sequence"] for r in results]
    print("\n=== Summary ===")
    print(f"mean IE restricted to label positions: {statistics.mean(ie_r):.3f}")
    print(f"mean IE full sequence:                 {statistics.mean(ie_f):.3f}")
    print(f"ratio (restricted/full):                {statistics.mean(ie_r)/statistics.mean(ie_f):.3f}")

    Path(ROOT / "results/head_label_position_validation.json").write_text(json.dumps(results, indent=2))
    print("\nWrote results/head_label_position_validation.json")


if __name__ == "__main__":
    main()
