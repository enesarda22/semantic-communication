"""
Build the unified stability+agreement figure (and a small companion table) that
replaces v12's Table III. Reads jsonl outputs from the four OSS evaluators (and,
when available, the GPT-family evaluators).

Inputs (any subset is fine; missing files are skipped with a warning):
  --llama   llama8b_1500.jsonl
  --qwen    qwen7b_1500.jsonl
  --mistral mistral7b_1500.jsonl
  --mixtral mixtral8x7b_1500.jsonl
  --gpt     gpt_stability_raw.jsonl   (optional; multi-prompt/multi-model GPT)

Outputs (under --out dir, default ./figs):
  stability_figure.pdf          one-panel bar chart, evaluator-mean +/- 95% CI
  stability_table3.csv          companion table for v13.tex
  cross_evaluator_spearman.csv  pairwise Spearman rho matrix
  best_method_per_distance.csv  unanimous-or-not check per distance

Usage:
  python make_stability_figure.py \\
      --llama llama8b_1500.jsonl \\
      --qwen qwen7b_1500.jsonl \\
      --mistral mistral7b_1500.jsonl \\
      --mixtral mixtral8x7b_1500.jsonl \\
      --out figs/
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

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Matplotlib not available; will skip figure rendering.", file=sys.stderr)


METHODS = ["SLF", "SPF", "SSF", "AE-JSCC", "LLM-Conv"]
DISTANCES = [1000, 2000, 3000]


def load_jsonl(path: Path):
    if not path or not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def valid_records(records):
    return [r for r in records if r.get("parse_success") and r.get("score") is not None]


def cell_means(records):
    by_cell = defaultdict(list)
    for r in records:
        by_cell[(r["method"], r["d_sd"])].append(r["score"])
    return {k: statistics.mean(v) for k, v in by_cell.items()}


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    boots = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama", help="Llama 8B jsonl")
    p.add_argument("--qwen", help="Qwen 7B jsonl")
    p.add_argument("--mistral", help="Mistral 7B jsonl")
    p.add_argument("--mixtral", help="Mixtral 8x7B jsonl")
    p.add_argument("--gpt", help="GPT-family jsonl (one row per (pair,model,prompt))")
    p.add_argument("--out", default="figs", help="Output directory")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all evaluators
    raw = {
        "Llama 3.1 8B": load_jsonl(Path(args.llama)) if args.llama else [],
        "Qwen 2.5 7B": load_jsonl(Path(args.qwen)) if args.qwen else [],
        "Mistral 7B": load_jsonl(Path(args.mistral)) if args.mistral else [],
        "Mixtral 8x7B": load_jsonl(Path(args.mixtral)) if args.mixtral else [],
    }
    if args.gpt:
        gpt_recs = load_jsonl(Path(args.gpt))
        for model_id in sorted({r["model_id"] for r in gpt_recs}):
            label = model_id  # e.g. gpt-4o-mini-2024-07-18
            raw[label] = [r for r in gpt_recs if r["model_id"] == model_id]

    # Filter to valid records and drop empty evaluators
    evals = {name: valid_records(recs) for name, recs in raw.items() if recs}
    evals = {name: recs for name, recs in evals.items() if recs}
    if not evals:
        print("FATAL: no evaluator data provided.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(evals)} evaluators: {list(evals.keys())}")
    for name, recs in evals.items():
        print(f"  {name}: {len(recs)} valid records")

    # ----------- Companion table: std decomposition along axes -----------
    rows = []
    for name, recs in evals.items():
        all_scores = [r["score"] for r in recs]
        cm = cell_means(recs)
        method_means = {
            m: statistics.mean([cm[(m, d)] for d in DISTANCES if (m, d) in cm])
            for m in METHODS if any((m, d) in cm for d in DISTANCES)
        }
        cond_means = {
            d: statistics.mean([cm[(m, d)] for m in METHODS if (m, d) in cm])
            for d in DISTANCES if any((m, d) in cm for m in METHODS)
        }
        lo, hi = bootstrap_ci(all_scores)
        rows.append({
            "evaluator": name,
            "n": len(recs),
            "mean": statistics.mean(all_scores),
            "std_pairs": statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0,
            "std_cells": (statistics.stdev(cm.values()) if len(cm) > 1 else 0.0),
            "std_methods": (statistics.stdev(method_means.values()) if len(method_means) > 1 else 0.0),
            "std_conditions": (statistics.stdev(cond_means.values()) if len(cond_means) > 1 else 0.0),
            "ci_lo": lo,
            "ci_hi": hi,
        })

    csv_path = out_dir / "stability_table3.csv"
    with csv_path.open("w") as f:
        f.write("Evaluator,N,Mean,Std(pairs),Std(cells),Std(methods),Std(conditions),CI_lo,CI_hi\n")
        for r in rows:
            f.write(f"{r['evaluator']},{r['n']},{r['mean']:.4f},{r['std_pairs']:.4f},"
                    f"{r['std_cells']:.4f},{r['std_methods']:.4f},{r['std_conditions']:.4f},"
                    f"{r['ci_lo']:.4f},{r['ci_hi']:.4f}\n")
    print(f"Wrote {csv_path}")

    # ----------- Cross-evaluator Spearman rho on cell means -----------
    eval_cm = {name: cell_means(recs) for name, recs in evals.items()}
    common = sorted(set.intersection(*[set(cm.keys()) for cm in eval_cm.values()]))
    eval_names = list(evals.keys())
    spearman_csv = out_dir / "cross_evaluator_spearman.csv"
    with spearman_csv.open("w") as f:
        f.write("," + ",".join(eval_names) + "\n")
        for n1 in eval_names:
            v1 = [eval_cm[n1][c] for c in common]
            f.write(n1)
            for n2 in eval_names:
                v2 = [eval_cm[n2][c] for c in common]
                rho, _ = spearmanr(v1, v2)
                f.write(f",{rho:.4f}")
            f.write("\n")
    print(f"Wrote {spearman_csv}")

    # ----------- Best method per distance per evaluator -----------
    best_csv = out_dir / "best_method_per_distance.csv"
    with best_csv.open("w") as f:
        f.write("Evaluator," + ",".join([f"d={d}m" for d in DISTANCES]) + ",Unanimous\n")
        unanimous_counts = []
        for d in DISTANCES:
            choices = []
            for name, cm in eval_cm.items():
                cands = [(m, cm.get((m, d), float("-inf"))) for m in METHODS]
                best = max(cands, key=lambda x: x[1])[0]
                choices.append(best)
            unanimous_counts.append(len(set(choices)) == 1)
        for name, cm in eval_cm.items():
            row = [name]
            for d in DISTANCES:
                cands = [(m, cm.get((m, d), float("-inf"))) for m in METHODS]
                best = max(cands, key=lambda x: x[1])[0]
                row.append(best)
            row.append("yes" if all(unanimous_counts) else "no")
            f.write(",".join(row) + "\n")
        f.write(f"Unanimous,{','.join('yes' if u else 'no' for u in unanimous_counts)},\n")
    print(f"Wrote {best_csv}")

    # ----------- Figure: per-evaluator mean grouped by distance condition -----------
    # X-axis: evaluators
    # For each evaluator: 3 grouped bars (one per distance), each averaging over 5 methods.
    # Y-axis: mean GPT-similarity score
    # Color: distance (light to dark)
    # Error bars: 95% bootstrap CI on the per-(evaluator, distance) cell.
    if HAS_MPL:
        names = [r["evaluator"] for r in rows]
        n_eval = len(names)
        n_dist = len(DISTANCES)
        bar_w = 0.25

        # Per (evaluator, distance) mean and bootstrap CI
        dist_data = {}
        for name, recs in evals.items():
            for d in DISTANCES:
                vals = [r["score"] for r in recs if r["d_sd"] == d]
                if not vals:
                    continue
                lo, hi = bootstrap_ci(vals)
                dist_data[(name, d)] = {
                    "mean": statistics.mean(vals),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }

        # Pleasant colormap: light->dark blues for 3 distances
        cmap = ["#a5b4fc", "#6366f1", "#312e81"]
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        for i, d in enumerate(DISTANCES):
            xs = [j + (i - 1) * bar_w for j in range(n_eval)]
            means_d = [dist_data[(name, d)]["mean"] for name in names]
            err_lo = [dist_data[(name, d)]["mean"] - dist_data[(name, d)]["ci_lo"] for name in names]
            err_hi = [dist_data[(name, d)]["ci_hi"] - dist_data[(name, d)]["mean"] for name in names]
            ax.bar(xs, means_d, bar_w, yerr=[err_lo, err_hi],
                   capsize=2, color=cmap[i], edgecolor="black", linewidth=0.4,
                   label=f"$d_{{SD}}={d}$ m")

        ax.set_xticks(range(n_eval))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Mean score", fontsize=9)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=3, columnspacing=0.8)
        fig.tight_layout()
        fig_path = out_dir / "stability_figure.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Wrote {fig_path}")
        plt.close(fig)

    # Print summary
    print()
    print("="*60)
    print("HEADLINE NUMBERS")
    print("="*60)
    print(f"Evaluators: {len(evals)}")
    if len(eval_names) > 1:
        pairwise_rhos = []
        for i, n1 in enumerate(eval_names):
            for n2 in eval_names[i+1:]:
                v1 = [eval_cm[n1][c] for c in common]
                v2 = [eval_cm[n2][c] for c in common]
                rho, _ = spearmanr(v1, v2)
                pairwise_rhos.append(rho)
        print(f"Pairwise Spearman rho range: [{min(pairwise_rhos):.3f}, {max(pairwise_rhos):.3f}]")
        print(f"  median: {statistics.median(pairwise_rhos):.3f}")
    print(f"Unanimous best-method count: {sum(unanimous_counts)}/{len(DISTANCES)} distances")


if __name__ == "__main__":
    main()
