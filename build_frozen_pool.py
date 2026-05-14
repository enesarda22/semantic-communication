"""
Build a stratified frozen pool for round-2 evaluator-comparison work.

Reads precomputed (source, reconstruction) pairs from the per-method xlsx files in
the artifacts/ directory, takes a stratified random sample of --n-per-cell
sentences per (method, d_sd) cell at the midpoint relay (gamma=0.5), and writes
one JSON line per pair to --out.

Total pool size = n_per_cell * 5 methods * 3 distances. With n_per_cell=40 the
pool has 600 pairs; with n_per_cell=100 the pool has 1500 pairs (matches the
round-1 Table III sample size while adding stratification across methods and
operating conditions).

The same set of source-sentence ids is used at every (method, distance) cell, so
each pool sentence has reconstructions from all five methods at all three
distances. Cross-method pairing is done by source-sentence text (not row index),
because the conventional baseline file uses a different sentence ordering.

Usage:
  python build_frozen_pool.py \\
      --artifacts-dir /Users/enesarda/projects/jstsp2025/artifacts \\
      --n-per-cell 100 \\
      --out frozen_pool_1500.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from semantic_communication.utils.general import set_seed

METHODS_TO_FILES = {
    "SLF": "forward_proposed_output.xlsx",
    "SPF": "predict_proposed_output.xlsx",
    "SSF": "sentence_proposed_output.xlsx",
    "AE-JSCC": "baseline_output_AWGN.xlsx",
    "LLM-Conv": "llmsc_baseline_output_AWGN.xlsx",
    # NOTE: 'Conv' (conventional_output_AWGN.xlsx) is intentionally excluded by
    # default because it uses a different test-sentence pool than the other
    # methods -- the source-sentence intersection across all six methods drops
    # to only 28 sentences when Conv is included. Excluding it gives an
    # intersection of ~921 sentences across the remaining five methods, which
    # is plenty for the round-2 stratified pool. Conv's behavior is already
    # documented qualitatively in Table IV of the manuscript.
}
DISTANCES_M = [1000, 2000, 3000]  # d_sd
GAMMA = 0.5  # midpoint relay, d_sr = d_rd = 0.5 * d_sd
N_PER_CELL = 40
SEED = 42


def load_method_cell(xlsx_path: Path, d_sd: int, gamma: float) -> pd.DataFrame:
    """Load (source, reconstruction) pairs for a single (method, d_sd, gamma) cell.

    Handles the SLF file which has an extra unnamed index column at position 0.
    """
    df = pd.read_excel(xlsx_path)
    # Drop unnamed index column if present
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    cell = df[(df["d_sd"] == d_sd) & (df["Gamma"] == gamma)].copy()
    return cell[["Sentence 1", "Sentence 2"]].rename(
        columns={"Sentence 1": "source", "Sentence 2": "reconstruction"}
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing the per-method *_output*.xlsx files.",
    )
    p.add_argument("--out", default="frozen_pool.jsonl")
    p.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    set_seed()  # repo-wide deterministic seed
    # Pandas sampling below uses args.seed independently.

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        print(f"FATAL: --artifacts-dir does not exist: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    # Step 1: load every (method, d_sd) cell into a dict.
    by_method_dist: Dict[tuple, pd.DataFrame] = {}
    for method, fn in METHODS_TO_FILES.items():
        path = artifacts_dir / fn
        if not path.is_file():
            print(f"FATAL: missing artifact for {method}: {path}", file=sys.stderr)
            sys.exit(2)
        for d_sd in DISTANCES_M:
            cell = load_method_cell(path, d_sd, GAMMA)
            if cell.empty:
                print(
                    f"FATAL: {method} has no rows at d_sd={d_sd}, gamma={GAMMA} "
                    f"in {path}",
                    file=sys.stderr,
                )
                sys.exit(3)
            by_method_dist[(method, d_sd)] = cell

    # Step 2: pick the source-sentence set, frozen and reused everywhere.
    # We require each chosen source to be present in EVERY (method, d_sd) cell so
    # the resulting pool is perfectly balanced (no skipped pairs downstream).
    # The intersection is necessary because (a) Conv uses a different sentence
    # ordering, and (b) a sentence sampled at d_sd=1000 may not appear at
    # d_sd=2000 or 3000 for every method.
    cell_source_sets = {
        key: set(df["source"].tolist()) for key, df in by_method_dist.items()
    }
    intersect_sources = set.intersection(*cell_source_sets.values())
    print(
        f"Source intersection across all {len(by_method_dist)} cells: "
        f"{len(intersect_sources)} sentences."
    )
    if len(intersect_sources) < args.n_per_cell:
        print(
            f"FATAL: intersection has only {len(intersect_sources)} sources; "
            f"need {args.n_per_cell}.",
            file=sys.stderr,
        )
        sys.exit(4)

    # Stratified-uniform sample: same N source sentences across every cell.
    sampled = sorted(intersect_sources)  # deterministic order before sampling
    sampled = pd.Series(sampled).sample(
        n=args.n_per_cell, random_state=args.seed
    ).tolist()
    print(f"Sampled {len(sampled)} source sentences (seed={args.seed}).")

    # Step 3: emit one JSON line per pair.
    out_path = Path(args.out)
    n_written = 0
    n_skipped = 0
    with out_path.open("w") as f:
        for d_sd in DISTANCES_M:
            for method, _ in METHODS_TO_FILES.items():
                cell = by_method_dist[(method, d_sd)]
                # Build {source: reconstruction} from this cell.
                # If a source appears multiple times, take the first.
                src_to_recon: Dict[str, str] = {}
                for src, recon in zip(cell["source"], cell["reconstruction"]):
                    if src not in src_to_recon:
                        src_to_recon[src] = recon
                for sid_idx, src in enumerate(sampled):
                    if src not in src_to_recon:
                        n_skipped += 1
                        continue
                    record = {
                        "pair_id": f"{method}_{d_sd}_{sid_idx:03d}",
                        "sentence_id": f"s{sid_idx:03d}",
                        "method": method,
                        "d_sd": d_sd,
                        "d_sr": d_sd // 2,
                        "gamma": GAMMA,
                        "source": str(src),
                        "reconstruction": str(src_to_recon[src]),
                    }
                    f.write(json.dumps(record) + "\n")
                    n_written += 1

    print(f"Wrote {n_written} pairs to {out_path} (skipped {n_skipped}).")
    expected = args.n_per_cell * len(METHODS_TO_FILES) * len(DISTANCES_M)
    print(f"Expected {expected}; got {n_written}.")


if __name__ == "__main__":
    main()
