"""
Aggregate raw evaluator outputs into the statistics needed for v13.tex
(expanded Table III) and the new evaluator-comparison subsubsection.

Inputs (any subset; missing files are skipped with a warning):
  --gpt        gpt_stability_raw.jsonl  (3 GPT models x N prompts x 600 pairs)
  --oss        comma-separated list of OSS-LLM raw jsonl files (1 per model)
  --bertscore  bertscore_raw.jsonl

Outputs (under --out dir):
  table3_provenance.json    - N, parse success, hyperparameters footnote
  table3_stability.csv      - rows: dimension; cols: mean, std, min, max, bootstrap CI
  comparison_table.csv      - rows: evaluator; cols: type, parse success,
                              rho-vs-LOO-consensus, top-method-agreement
  consensus_summary.json    - median/min pairwise Spearman among generative LLMs
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import spearmanr
except ImportError:
    print("Install: pip install numpy scipy", file=sys.stderr)
    sys.exit(1)


def load_jsonl(path: Path):
    if not path.is_file():
        print(f"WARN: {path} does not exist; skipping.", file=sys.stderr)
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cell_means(records):
    """Return {(method, d_sd): mean_score} from successful records only."""
    by_cell = defaultdict(list)
    for r in records:
        if r.get("parse_success") and r.get("score") is not None:
            by_cell[(r["method"], r["d_sd"])].append(r["score"])
    return {k: statistics.mean(v) for k, v in by_cell.items() if v}


def rank_vector(score_map, cell_order):
    scores = np.array([score_map[c] for c in cell_order])
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    for i, idx in enumerate(order):
        ranks[idx] = i + 1
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = sum(range(i + 1, j + 2)) / (j - i + 1)
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def best_method_per_distance(records, distances):
    cm = cell_means(records)
    best = {}
    for d in distances:
        cands = [(m, s) for (m, dd), s in cm.items() if dd == d]
        if cands:
            best[d] = max(cands, key=lambda x: x[1])[0]
    return best


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpt", required=True)
    p.add_argument("--oss", default="", help="Comma-separated OSS-LLM jsonl files.")
    p.add_argument("--bertscore", default="")
    p.add_argument("--out", default="summary")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gpt = load_jsonl(Path(args.gpt))
    if not gpt:
        print(f"FATAL: GPT input is required and empty: {args.gpt}", file=sys.stderr)
        sys.exit(1)
    oss_files = [s.strip() for s in args.oss.split(",") if s.strip()]
    oss_records = {Path(f).stem: load_jsonl(Path(f)) for f in oss_files}
    bs = load_jsonl(Path(args.bertscore)) if args.bertscore else []

    # ---------- Provenance ----------
    n_pairs = len({r["pair_id"] for r in gpt})
    parse_rate_gpt = sum(1 for r in gpt if r["parse_success"]) / max(1, len(gpt))
    provenance = {
        "n_pairs": n_pairs,
        "n_models": len({r["model_id"] for r in gpt}),
        "n_prompts": len({r["prompt_id"] for r in gpt}),
        "calls_per_cell": 1,
        "temperature": 0,
        "parse_success_rate": parse_rate_gpt,
    }
    (out / "table3_provenance.json").write_text(json.dumps(provenance, indent=2))

    # ---------- Stability (Table III) ----------
    valid = [r for r in gpt if r["parse_success"] and r["score"] is not None]
    by_pair = defaultdict(list)
    by_model = defaultdict(list)
    by_prompt = defaultdict(list)
    for r in valid:
        by_pair[r["pair_id"]].append(r["score"])
        by_model[r["model_id"]].append(r["score"])
        by_prompt[r["prompt_id"]].append(r["score"])

    rows = []
    all_scores = [r["score"] for r in valid]
    overall_mean = statistics.mean(all_scores)
    overall_std = statistics.pstdev(all_scores)
    lo, hi = bootstrap_ci(all_scores)
    rows.append(("Overall (all cells)", overall_mean, overall_std, min(all_scores),
                 max(all_scores), f"[{lo:.3f}, {hi:.3f}]"))

    pair_means = [statistics.mean(v) for v in by_pair.values()]
    rows.append(("Pairs (avg over models & prompts)", statistics.mean(pair_means),
                 statistics.pstdev(pair_means), min(pair_means), max(pair_means), "-"))

    model_means = [statistics.mean(v) for v in by_model.values()]
    rows.append(("Models (avg over pairs & prompts)", statistics.mean(model_means),
                 statistics.pstdev(model_means), min(model_means), max(model_means), "-"))

    prompt_means = [statistics.mean(v) for v in by_prompt.values()]
    rows.append(("Prompts (avg over pairs & models)", statistics.mean(prompt_means),
                 statistics.pstdev(prompt_means), min(prompt_means), max(prompt_means), "-"))

    with (out / "table3_stability.csv").open("w") as f:
        f.write("Dimension,Mean,Std,Min,Max,95%-bootstrap-CI\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # ---------- Comparison table (LOO-consensus over generative LLMs) ----------
    distances = sorted({r["d_sd"] for r in gpt})
    cells = sorted({(r["method"], r["d_sd"]) for r in gpt})
    cell_order = list(cells)

    evaluators = {
        "gpt-family": {
            "type": "Generative (GPT)",
            "records": valid,
            "in_consensus": True,
        }
    }
    for tag, recs in oss_records.items():
        evaluators[tag] = {
            "type": "Generative (OSS)",
            "records": [r for r in recs if r.get("parse_success") and r.get("score") is not None],
            "in_consensus": True,
        }
    if bs:
        evaluators["bertscore"] = {
            "type": "Embedding (non-generative)",
            "records": bs,
            "in_consensus": False,
        }

    eval_cell_means = {tag: cell_means(info["records"]) for tag, info in evaluators.items()}
    eval_rank_vectors = {}
    for tag, cm in eval_cell_means.items():
        if all(c in cm for c in cell_order):
            eval_rank_vectors[tag] = rank_vector(cm, cell_order)

    consensus_evaluators = [
        tag for tag, info in evaluators.items()
        if info["in_consensus"] and tag in eval_rank_vectors
    ]

    rows = []
    rho_consensus_per_eval = {}
    for tag, info in evaluators.items():
        recs = info["records"]
        parse_succ = sum(1 for r in recs if r.get('parse_success')) / max(1, len(recs))
        if tag not in eval_rank_vectors:
            rows.append((tag, info["type"], f"{parse_succ:.3f}", "n/a", "n/a"))
            continue
        if tag in consensus_evaluators:
            others = [t for t in consensus_evaluators if t != tag]
        else:
            others = consensus_evaluators
        if not others:
            rho = float("nan")
        else:
            consensus_rank = np.mean([eval_rank_vectors[t] for t in others], axis=0)
            rho, _ = spearmanr(eval_rank_vectors[tag], consensus_rank)
        rho_consensus_per_eval[tag] = rho

        eval_best = best_method_per_distance(recs, distances)
        consensus_cell_score = defaultdict(list)
        for t in others:
            for c, s in eval_cell_means[t].items():
                consensus_cell_score[c].append(s)
        consensus_avg = {c: float(np.mean(v)) for c, v in consensus_cell_score.items()}
        consensus_best = {}
        for d in distances:
            cands = [(m, s) for (m, dd), s in consensus_avg.items() if dd == d]
            if cands:
                consensus_best[d] = max(cands, key=lambda x: x[1])[0]
        agree = sum(1 for d in distances if eval_best.get(d) == consensus_best.get(d))
        rows.append((tag, info["type"], f"{parse_succ:.3f}", f"{rho:.3f}",
                     f"{agree}/{len(distances)}"))

    with (out / "comparison_table.csv").open("w") as f:
        f.write("Evaluator,Type,Parse success,rho vs LOO/LLM consensus,Top-method agreement\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # ---------- Pairwise Spearman among generative evaluators ----------
    pairwise = []
    for i, t1 in enumerate(consensus_evaluators):
        for t2 in consensus_evaluators[i + 1:]:
            rho, _ = spearmanr(eval_rank_vectors[t1], eval_rank_vectors[t2])
            pairwise.append({"a": t1, "b": t2, "rho": float(rho)})
    summary = {
        "n_generative_evaluators": len(consensus_evaluators),
        "median_pairwise_rho": (
            float(statistics.median([p["rho"] for p in pairwise])) if pairwise else None
        ),
        "min_pairwise_rho": float(min(p["rho"] for p in pairwise)) if pairwise else None,
        "max_pairwise_rho": float(max(p["rho"] for p in pairwise)) if pairwise else None,
        "pairwise": pairwise,
        "rho_vs_loo_consensus": rho_consensus_per_eval,
    }
    (out / "consensus_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Aggregation complete. Outputs in {out}/")


if __name__ == "__main__":
    main()
