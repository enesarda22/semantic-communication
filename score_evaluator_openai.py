"""
Score the frozen pool with GPT-family evaluators (round-2 stability + comparison).

For each (pair, prompt, model) cell:
  - One API call at temperature 0 with strict JSON response format.
  - On parse failure or out-of-range output, retry with the repair prompt
    up to --max-retries times. Mark cell invalid if still bad.
  - Log raw response, parsed score (or null), parse_attempts, parse_success.

Outputs one JSON line per (pair, prompt, model) cell to --out.

Set OPENAI_API_KEY in environment, or pass --api-key.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed; pip install openai", file=sys.stderr)
    sys.exit(1)

from semantic_communication.utils.general import set_seed


SCORE_RE = re.compile(r'"\s*score\s*"\s*:\s*(-?\d+(?:\.\d+)?)')


def parse_score(raw: str):
    """Strict JSON parse; fall back to regex extraction. Returns float in [-1,1] or None."""
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


def build_user_msg(prompt_text: str, source: str, reconstruction: str) -> str:
    return f"{prompt_text}\n\nSentence A: {source}\nSentence B: {reconstruction}"


def score_one(client, model, prompt_text, repair_prompt, source, reconstruction,
              seed, max_retries):
    user_msg = build_user_msg(prompt_text, source, reconstruction)
    history = [{"role": "user", "content": user_msg}]
    raws: list = []
    for attempt in range(max_retries + 1):
        kwargs = dict(
            model=model,
            messages=history,
            temperature=0,
            top_p=1.0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content
        raws.append(raw)
        score = parse_score(raw)
        if score is not None:
            return score, raws, True
        history.append({"role": "assistant", "content": raw})
        history.append({"role": "user", "content": repair_prompt})
    return None, raws, False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True, help="frozen_pool.jsonl path")
    p.add_argument("--prompts", default="prompts.json")
    p.add_argument(
        "--prompt-keys",
        default="canonical",
        help="'canonical', 'all' (canonical+5 variants), or "
             "comma-separated keys e.g. 'canonical,variant_1,variant_3'",
    )
    p.add_argument(
        "--models",
        required=True,
        help="Comma-separated dated GPT model IDs, e.g. "
             "gpt-4o-mini-2024-07-18,gpt-4.1-nano-...,gpt-5-nano-...",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--pilot", type=int, default=0,
                   help="If >0, only score the first N pairs (pilot mode).")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-seed", action="store_true",
                   help="Do not pass the OpenAI seed parameter (some models reject it).")
    p.add_argument("--rate-limit-sleep", type=float, default=0.0)
    p.add_argument("--api-key", type=str, default=None)
    args = p.parse_args()

    set_seed()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set; pass --api-key or set env var.", file=sys.stderr)
        sys.exit(1)

    prompts = json.loads(Path(args.prompts).read_text())
    if args.prompt_keys == "canonical":
        selected = [("canonical", prompts["canonical"])]
    elif args.prompt_keys == "all":
        selected = [("canonical", prompts["canonical"])] + [
            (f"variant_{i+1}", v) for i, v in enumerate(prompts["variants"])
        ]
    else:
        keys = [k.strip() for k in args.prompt_keys.split(",")]
        selected = []
        for k in keys:
            if k == "canonical":
                selected.append((k, prompts["canonical"]))
            elif k.startswith("variant_"):
                idx = int(k.split("_")[1]) - 1
                selected.append((k, prompts["variants"][idx]))
            else:
                raise ValueError(f"Unknown prompt key: {k}")

    # Sanity check: detect TBD prompts and abort to avoid wasted API calls.
    for tag, text in selected:
        if "[TBD" in text:
            print(
                f"FATAL: prompt {tag} still contains [TBD ...] markers in {args.prompts}. "
                f"Fill it in (e.g., from coauthor) before running.",
                file=sys.stderr,
            )
            sys.exit(2)

    models = [m.strip() for m in args.models.split(",")]
    seed = None if args.no_seed else args.seed

    client = OpenAI()

    pool = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    if args.pilot > 0:
        pool = pool[: args.pilot]
    print(f"Scoring {len(pool)} pairs x {len(selected)} prompts x {len(models)} models "
          f"= {len(pool) * len(selected) * len(models)} cells.")

    out_path = Path(args.out)
    n_done = 0
    n_invalid = 0
    with out_path.open("w") as f:
        for pair in pool:
            for prompt_id, prompt_text in selected:
                for model in models:
                    try:
                        score, raws, success = score_one(
                            client, model, prompt_text, prompts["repair"],
                            pair["source"], pair["reconstruction"],
                            seed, args.max_retries,
                        )
                    except Exception as e:
                        score, raws, success = None, [f"ERROR: {e}"], False
                    record = {
                        "pair_id": pair["pair_id"],
                        "method": pair["method"],
                        "d_sd": pair["d_sd"],
                        "model_id": model,
                        "prompt_id": prompt_id,
                        "score": score,
                        "raw_responses": raws,
                        "parse_attempts": len(raws),
                        "parse_success": success,
                        "seed_used": seed,
                    }
                    f.write(json.dumps(record) + "\n")
                    n_done += 1
                    if not success:
                        n_invalid += 1
                    if args.rate_limit_sleep > 0:
                        time.sleep(args.rate_limit_sleep)
    print(f"Wrote {n_done} cells to {out_path}; {n_invalid} invalid after retries.")


if __name__ == "__main__":
    main()
