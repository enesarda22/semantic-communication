"""
Score the frozen pool with an open-source generative LLM (Llama, Qwen, Mistral, ...).

Default backend: HuggingFace transformers, optionally with bitsandbytes 4-/8-bit.
For larger models on the supercomputer, swap to vLLM by replacing `infer_one()`.

Outputs one JSON line per (pair, prompt) cell to --out. Supports multi-prompt
sweeps (--prompt-keys), temperature override (--temperature), and seed
override (--seed) for the OSS-side stability axes parallel to the GPT sweep.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from transformers import BitsAndBytesConfig
        BNB_AVAILABLE = True
    except ImportError:
        BNB_AVAILABLE = False
except ImportError:
    print("Install: pip install transformers accelerate torch (and bitsandbytes for quant)",
          file=sys.stderr)
    sys.exit(1)

from semantic_communication.utils.general import set_seed, get_device


SCORE_RE = re.compile(r'"\s*score\s*"\s*:\s*(-?\d+(?:\.\d+)?)')


def parse_score(raw: str):
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "score" in obj:
            score = float(obj["score"])
            if -1.0 <= score <= 1.0:
                return score
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = SCORE_RE.search(raw)
    if m:
        try:
            score = float(m.group(1))
            if -1.0 <= score <= 1.0:
                return score
        except ValueError:
            pass
    return None


def build_messages(prompt_text: str, source: str, reconstruction: str):
    user_msg = f"{prompt_text}\n\nSentence A: {source}\nSentence B: {reconstruction}"
    return [{"role": "user", "content": user_msg}]


def infer_one(model, tokenizer, messages, max_new_tokens: int,
              temperature: float = 0.0) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
    with torch.no_grad():
        out = model.generate(inputs, **gen_kwargs)
    new_tokens = out[0, inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--prompts", default="prompts.json")
    p.add_argument("--prompt-keys", default="canonical",
                   help="'canonical', 'all' (canonical+5 variants), or "
                        "comma-separated keys e.g. 'canonical,variant_1,variant_3'.")
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--pilot", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                   help="Quantization for memory savings (requires bitsandbytes)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0.0 triggers greedy decoding).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for stochastic decoding (only used when T>0).")
    args = p.parse_args()

    set_seed(args.seed)
    if args.temperature > 0.0:
        torch.manual_seed(args.seed)
        try:
            torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

    prompts = json.loads(Path(args.prompts).read_text())
    repair = prompts["repair"]
    # Resolve prompt-keys argument.
    if args.prompt_keys == "canonical":
        selected_prompts = [("canonical", prompts["canonical"])]
    elif args.prompt_keys == "all":
        selected_prompts = [("canonical", prompts["canonical"])] + [
            (f"variant_{i+1}", v) for i, v in enumerate(prompts["variants"])
        ]
    else:
        keys = [k.strip() for k in args.prompt_keys.split(",")]
        selected_prompts = []
        for k in keys:
            if k == "canonical":
                selected_prompts.append((k, prompts["canonical"]))
            elif k.startswith("variant_"):
                idx = int(k.split("_")[1]) - 1
                selected_prompts.append((k, prompts["variants"][idx]))
            else:
                print(f"FATAL: unknown prompt key '{k}'.", file=sys.stderr)
                sys.exit(2)
    for pid, p_text in selected_prompts:
        if "[TBD" in p_text:
            print(f"FATAL: prompt '{pid}' in {args.prompts} still has [TBD].",
                  file=sys.stderr)
            sys.exit(2)
    print(f"Using {len(selected_prompts)} prompt(s); T={args.temperature}; seed={args.seed}",
          file=sys.stderr)

    quantization_config = None
    if args.quant != "none":
        if not BNB_AVAILABLE:
            print("FATAL: --quant requires bitsandbytes; pip install bitsandbytes", file=sys.stderr)
            sys.exit(3)
        if args.quant == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif args.quant == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} (quant={args.quant}, dtype={args.dtype})", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quantization_config,
    )
    model.eval()
    print(f"Model loaded on device map {model.device}.", file=sys.stderr)

    pool = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    if args.pilot > 0:
        pool = pool[: args.pilot]

    out_path = Path(args.out)
    # Resume from existing output: skip (pair_id, prompt_id) combos already scored. Append mode.
    done_cells = set()
    if out_path.is_file() and out_path.stat().st_size > 0:
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done_cells.add((rec["pair_id"], rec.get("prompt_id", "canonical")))
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"Resume: found {len(done_cells)} already-scored cells in {out_path}.",
              file=sys.stderr)
    total_cells = len(pool) * len(selected_prompts)
    remaining = total_cells - len(done_cells)
    print(f"Scoring {remaining} of {total_cells} (pair, prompt) cells with {args.model}.",
          file=sys.stderr)

    n_done = 0
    n_invalid = 0
    with out_path.open("a") as f:
        for pair in pool:
            for prompt_id, prompt_text in selected_prompts:
                if (pair["pair_id"], prompt_id) in done_cells:
                    continue
                messages = build_messages(prompt_text, pair["source"], pair["reconstruction"])
                raws: list = []
                score = None
                success = False
                for attempt in range(args.max_retries + 1):
                    raw = infer_one(model, tokenizer, messages, args.max_new_tokens,
                                    temperature=args.temperature)
                    raws.append(raw)
                    score = parse_score(raw)
                    if score is not None:
                        success = True
                        break
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": repair})
                record = {
                    "pair_id": pair["pair_id"],
                    "method": pair["method"],
                    "d_sd": pair["d_sd"],
                    "model_id": args.model,
                    "prompt_id": prompt_id,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "score": score,
                    "raw_responses": raws,
                    "parse_attempts": len(raws),
                    "parse_success": success,
                }
                f.write(json.dumps(record) + "\n")
                n_done += 1
                if not success:
                    n_invalid += 1
    print(f"Wrote {n_done} cells to {out_path}; {n_invalid} invalid after retries.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
