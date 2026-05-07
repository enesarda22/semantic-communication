"""
Determinism sanity check: pick a 50-pair stratified subset, score with one GPT
model 3x at temperature 0, report max per-pair std. Used in the response letter
to address R2's reproducibility concern about API non-determinism.

Outputs a single JSON summary to --out.
"""

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed; pip install openai", file=sys.stderr)
    sys.exit(1)

# Reuse helpers from score_evaluator_openai.py (in same directory).
sys.path.insert(0, str(Path(__file__).parent))
from score_evaluator_openai import parse_score, build_user_msg  # noqa: E402

from semantic_communication.utils.general import set_seed  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--prompts", default="prompts.json")
    p.add_argument("--model", required=True, help="dated GPT model id")
    p.add_argument("--out", default="repeat_check.json")
    p.add_argument("--subset", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api-key", type=str, default=None)
    args = p.parse_args()

    set_seed()
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL: OPENAI_API_KEY not set; pass --api-key or set env var.", file=sys.stderr)
        sys.exit(1)

    pool = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]

    # Stratified subset across (method, d_sd) cells.
    by_cell = {}
    for pair in pool:
        by_cell.setdefault((pair["method"], pair["d_sd"]), []).append(pair)
    cells = list(by_cell.keys())
    rng = random.Random(args.seed)
    per_cell = max(1, args.subset // len(cells))
    subset = []
    for cell in cells:
        subset.extend(rng.sample(by_cell[cell], min(per_cell, len(by_cell[cell]))))
    subset = subset[: args.subset]
    print(f"Stratified subset: {len(subset)} pairs across {len(cells)} cells.")

    prompts = json.loads(Path(args.prompts).read_text())
    canonical = prompts["canonical"]
    if "[TBD" in canonical:
        print(f"FATAL: canonical prompt in {args.prompts} still has [TBD].",
              file=sys.stderr)
        sys.exit(2)

    client = OpenAI()

    per_pair_stds = []
    raw_log = []
    for pair in subset:
        scores = []
        for r in range(args.repeats):
            msg = build_user_msg(canonical, pair["source"], pair["reconstruction"])
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": msg}],
                temperature=0,
                top_p=1.0,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            sc = parse_score(raw)
            if sc is None:
                print(f"WARN parse-fail pair={pair['pair_id']} repeat={r}", file=sys.stderr)
                continue
            scores.append(sc)
        if len(scores) >= 2:
            per_pair_stds.append(statistics.stdev(scores))
        raw_log.append({"pair_id": pair["pair_id"], "scores": scores})

    summary = {
        "model_id": args.model,
        "subset_size": len(subset),
        "repeats": args.repeats,
        "max_per_pair_std": max(per_pair_stds) if per_pair_stds else None,
        "mean_per_pair_std": (
            statistics.mean(per_pair_stds) if per_pair_stds else None
        ),
        "n_pairs_with_full_scores": len(per_pair_stds),
        "details": raw_log,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}.")
    print(f"  max per-pair std = {summary['max_per_pair_std']}")
    print(f"  mean per-pair std = {summary['mean_per_pair_std']}")


if __name__ == "__main__":
    main()
