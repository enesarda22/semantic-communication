"""
Score the frozen pool with BERTScore F1 (default: roberta-large, rescale_with_baseline=True).

No prompt or model variation; one score per pair. Outputs one JSON line per pair.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from bert_score import BERTScorer
except ImportError:
    print("Install: pip install bert-score", file=sys.stderr)
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pilot", type=int, default=0)
    p.add_argument("--lang", default="en")
    p.add_argument("--model-type", default="roberta-large")
    p.add_argument("--no-rescale", action="store_true",
                   help="Disable rescale_with_baseline (default: enabled)")
    args = p.parse_args()

    pool = [json.loads(line) for line in Path(args.pool).read_text().splitlines() if line.strip()]
    if args.pilot > 0:
        pool = pool[: args.pilot]

    sources = [p_["source"] for p_ in pool]
    recons = [p_["reconstruction"] for p_ in pool]

    scorer = BERTScorer(
        model_type=args.model_type,
        lang=args.lang,
        rescale_with_baseline=not args.no_rescale,
    )
    P, R, F1 = scorer.score(recons, sources)

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for pair, p_, r_, f1_ in zip(pool, P.tolist(), R.tolist(), F1.tolist()):
            record = {
                "pair_id": pair["pair_id"],
                "method": pair["method"],
                "d_sd": pair["d_sd"],
                "model_id": f"BERTScore({args.model_type})",
                "prompt_id": "n/a",
                "score": f1_,
                "score_precision": p_,
                "score_recall": r_,
                "parse_success": True,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(pool)} BERTScore F1 entries to {out_path}.")


if __name__ == "__main__":
    main()
