"""Phase 1.1 (pilot): generate citations with Gemma 2 9B IT and save
per-token residual-stream activations at one Gemma Scope layer.

Usage:
    python src/generate.py --per-field 2 --out-dir data/generated_citations

Loads a subset of data/prompts/prompts.jsonl (--per-field prompts per
domain), generates with greedy decoding, and for each generation saves:
  - the generated text to generations.jsonl
  - the residual stream at LAYER (post-block, matching Gemma Scope's
    `blocks.{LAYER}.hook_resid_post`) for every generated token, to
    activations/{prompt_id}.pt

LAYER defaults to 20 because Gemma Scope only ships IT-tuned SAEs
(gemma-scope-9b-it-res) for layers 9, 20, and 31, and 20 sits in the
mid-network range the protocol recommends.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # triton JIT needs python3-dev headers, unavailable here

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-2-9b-it"
LAYER = 20


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:3", attn_implementation="eager"
    )
    model.eval()
    return tokenizer, model


def select_prompts(prompts_path: Path, per_field: int, domains: list[str] | None) -> list[dict]:
    rows = [json.loads(l) for l in open(prompts_path)]
    if domains:
        rows = [r for r in rows if r["field"] in domains]
    selected = []
    seen: dict[str, int] = {}
    for row in rows:
        c = seen.get(row["field"], 0)
        if c < per_field:
            selected.append(row)
            seen[row["field"]] = c + 1
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(ROOT / "data/prompts/prompts.jsonl"))
    ap.add_argument("--per-field", type=int, default=2, help="prompts sampled per domain, in file order")
    ap.add_argument("--domains", nargs="*", default=None, help="restrict to these domains; default = all")
    ap.add_argument("--out-dir", default=str(ROOT / "data/generated_citations"))
    ap.add_argument("--max-new-tokens", type=int, default=300)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    selected = select_prompts(Path(args.prompts), args.per_field, args.domains)
    print(f"Selected {len(selected)} prompts across {len(set(r['field'] for r in selected))} domains")

    tokenizer, model = load_model()

    captured = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["resid"] = hs.detach()

    handle = model.model.layers[LAYER].register_forward_hook(hook)

    results = []
    for i, row in enumerate(selected):
        chat = [{"role": "user", "content": row["prompt"]}]
        input_ids = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=False
        ).to(model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen_ids = out[0][input_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        with torch.no_grad():
            model(out, use_cache=False)
        resid = captured["resid"][0]
        gen_resid = resid[input_len:].to(torch.float32).cpu()

        act_path = act_dir / f"{row['prompt_id']}.pt"
        torch.save({"layer": LAYER, "token_ids": gen_ids.cpu(), "resid_post": gen_resid}, act_path)

        results.append(
            {
                "prompt_id": row["prompt_id"],
                "field": row["field"],
                "topic": row["topic"],
                "prompt": row["prompt"],
                "generated_text": gen_text,
                "activation_path": str(act_path.relative_to(ROOT)),
                "n_generated_tokens": len(gen_ids),
            }
        )
        print(f"[{i + 1}/{len(selected)}] {row['prompt_id']}: {len(gen_ids)} tokens")

    handle.remove()

    out_path = out_dir / "generations.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} generations to {out_path}")


if __name__ == "__main__":
    main()
