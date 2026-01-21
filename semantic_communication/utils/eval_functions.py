import os
import pandas as pd
from torch.nn import functional as F
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plotter(x_axis_values, y_axis_values, x_label, y_label, title):
    plt.figure()
    plt.plot(x_axis_values, y_axis_values)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{title}.png", dpi=400)


def sbert_semantic_similarity_score(target_sentence, received_sentence, sbert_model):
    target_emb = sbert_model.encode(target_sentence, convert_to_tensor=True).unsqueeze(0)
    received_emb = sbert_model.encode(received_sentence, convert_to_tensor=True).unsqueeze(0)
    scores = F.cosine_similarity(target_emb, received_emb)
    return scores[0].item()


def semantic_similarity_score(target_sentences, received_sentences, client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are skilled in evaluating how similar the two sentences are. Provide a number between -1 "
                           "and 1 denoting the semantic similarity score for given sentences A and B with precision "
                           "0.01. 1 means they are perfectly similar and -1 means they are opposite while 0 means their "
                           "meanings are uncorrelated. Just provide a score without any words or symbols.",
            },
            {
                "role": "user",
                "content": f"A=({target_sentences})  B=({received_sentences})",
            },
        ],
    )

    if completion.choices[0].finish_reason == "stop":
        pattern = re.compile(r"(?<![\d.-])-?(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")
        res = pattern.findall(completion.choices[0].message.content)
        if len(res) == 1:
            return float(res[0])
        else:
            print(res)
            return float("nan")
    else:
        return float("nan")


def eval_semantic_from_xlsx_to_csv(
    xlsx_path: str,
    prompts: list,
    models: list,
    output_csv_path: str,
    client,
    sheet=0,
    header=None,
    drop_empty=True,
    temp0_models=("gpt-4.1-nano", "gpt-4o-mini"),   # models that will receive temperature=0
    temperature=0.0,
    verbose=True,
):

    # --- Load pairs ---
    df = pd.read_excel(xlsx_path, sheet_name=sheet, header=header)
    if drop_empty:
        df = df.dropna(subset=[0, 1])

    targets = df.iloc[:, 0].astype(str).tolist()
    receiveds = df.iloc[:, 1].astype(str).tolist()
    pairs = list(zip(targets, receiveds))
    n_pairs = len(pairs)

    if n_pairs == 0:
        raise ValueError("No (target, received) pairs found in the first two columns of the XLSX.")

    # --- Prepare output parsing ---
    score_re = re.compile(
        r"^\s*(?:"
        r"-1(?:\.0+)?"
        r"|"
        r"-?0(?:\.\d+)?"
        r"|"
        r"1(?:\.0+)?"
        r")\s*$"
    )

    # No try/except in the hot loop; only send temperature to whitelisted models
    request_kwargs_by_model = {}
    for m in models:
        if m in temp0_models:
            request_kwargs_by_model[m] = {"temperature": float(temperature)}
        else:
            request_kwargs_by_model[m] = {}

    def score_once(model_name: str, system_prompt: str, target_text: str, received_text: str) -> float:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"A=({target_text})  B=({received_text})"},
            ],
            **request_kwargs_by_model.get(model_name, {}),
        )

        if completion.choices[0].finish_reason != "stop":
            return float("nan")

        raw = completion.choices[0].message.content
        m = score_re.match(raw)
        if not m:
            return float("nan")

        val = float(m.group(0))
        return val if (-1.0 <= val <= 1.0) else float("nan")

    def summarize(arr) -> dict:
        arr = np.asarray(arr, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return {
                "n": int(arr.size), "n_valid": 0, "invalid_rate": 1.0,
                "mean": np.nan, "std": np.nan, "var": np.nan, "min": np.nan, "max": np.nan
            }
        std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
        return {
            "n": int(arr.size),
            "n_valid": int(valid.size),
            "invalid_rate": float(1.0 - valid.size / arr.size),
            "mean": float(np.mean(valid)),
            "std": std,
            "var": float(std ** 2),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
        }

    # --- Compute all scores ---
    n_models = len(models)
    n_prompts = len(prompts)

    # scores[i, m, p]
    scores = np.full((n_pairs, n_models, n_prompts), np.nan, dtype=float)

    total_calls = n_pairs * n_models * n_prompts
    call_i = 0

    with tqdm(total=total_calls, desc="Scoring", unit="call") as pbar:
        for i, (t, r) in enumerate(pairs):
            for mi, model_name in enumerate(models):
                for pi, prompt in enumerate(prompts):
                    scores[i, mi, pi] = score_once(model_name, prompt, t, r)
                    pbar.update(1)

    # --- Build wide CSV rows ---
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    columns = ["target_text", "received_text"]
    for model_name in models:
        for pi in range(n_prompts):
            columns.append(f"{model_name}__prompt{pi}")

    out_rows = []
    for i, (t, r) in enumerate(pairs):
        row = [t, r]
        for mi in range(n_models):
            for pi in range(n_prompts):
                row.append(scores[i, mi, pi])
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows, columns=columns)
    out_df.to_csv(output_csv_path, index=False)

    # --- Dimension stats (exactly as requested) ---

    # 1) Pair dimension: average over models+prompts per pair, then stats across pairs
    pair_avg = np.nanmean(scores, axis=(1, 2))  # shape (n_pairs,)
    pair_stats = summarize(pair_avg)

    # 2) Model dimension: average over pairs+prompts per model, then stats across models
    model_avg = np.nanmean(scores, axis=(0, 2))  # shape (n_models,)
    model_stats = summarize(model_avg)

    # 3) Prompt dimension: average over pairs+models per prompt, then stats across prompts
    prompt_avg = np.nanmean(scores, axis=(0, 1))  # shape (n_prompts,)
    prompt_stats = summarize(prompt_avg)

    # Also useful: overall invalid rate across all cells
    overall_stats = summarize(scores.reshape(-1))

    if verbose:
        print("\n=== Run summary ===")
        print(f"Pairs: {n_pairs}")
        print(f"Models: {n_models} -> {models}")
        print(f"Prompts: {n_prompts}")
        print(f"Total calls: {total_calls}")
        print(f"Saved CSV: {output_csv_path}")

        print("\n=== Overall (all pair×model×prompt cells) ===")
        print(overall_stats)

        print("\n=== Pairs dimension ===")
        print("Computed: pair_avg[i] = mean over models+prompts for each pair i")
        print(pair_stats)

        print("\n=== Models dimension ===")
        print("Computed: model_avg[m] = mean over pairs+prompts for each model m")
        print(model_stats)

        print("\n=== Prompts dimension ===")
        print("Computed: prompt_avg[p] = mean over pairs+models for each prompt p")
        print(prompt_stats)

    return {
        "output_df": out_df,
        "scores_array": scores,          # (pairs, models, prompts)
        "pair_avg": pair_avg,
        "model_avg": model_avg,
        "prompt_avg": prompt_avg,
        "overall_stats": overall_stats,
        "pair_stats": pair_stats,
        "model_stats": model_stats,
        "prompt_stats": prompt_stats,
        "output_csv_path": output_csv_path,
    }
