import os
import argparse
from openai import OpenAI
from semantic_communication.utils.eval_functions import eval_semantic_from_xlsx_to_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--outcsv", default="Results/semantic_prompt_sweep/all_pairs_scores.csv")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Provide --api-key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    prompt_1 = "You are skilled in evaluating how similar the two sentences are. Provide a number between -1 and 1 denoting the semantic similarity score for given sentences A and B with precision 0.01. 1 means they are perfectly similar and -1 means they are opposite while 0 means their meanings are uncorrelated. Just provide a score without any words or symbols."
    prompt_2 = "Rate the semantic similarity of sentences A and B. Respond with exactly one number in [-1, 1] to two decimal places. Use 1.00 for the same meaning, 0.00 for unrelated meaning, and -1.00 for opposite/contradictory meaning. Output only the number."
    prompt_3 = "Compare the meanings of sentences A and B and output a single similarity score between -1 and 1 (precision 0.01). 1 means their meanings match, 0 means they are unrelated, and -1 means they express opposite or contradictory meanings. Return only the numeric score."
    prompt_4 = "Evaluate how similar in meaning sentence A is to sentence B. Provide one number between -1 and 1 with precision 0.01: 1 indicates identical meaning, 0 indicates no meaningful relation, and -1 indicates contradiction/opposition. Output only the number."
    prompt_5 = "Give a semantic similarity score for A and B as a single value in [-1, 1] rounded to two decimals. 1 means same meaning, 0 means unrelated meanings, and -1 means opposite/contradictory meanings. Output only the number."
    prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5]

    models = ["gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini"]

    eval_semantic_from_xlsx_to_csv(
        xlsx_path=args.xlsx,
        prompts=prompts,
        models=models,
        output_csv_path=args.outcsv,
        client=client,
        sheet=args.sheet,
        header=None,              # important if your file has no header row
        temp0_models=("gpt-4.1-nano", "gpt-4o-mini"),  # temperature=0 only where safe
        temperature=0.0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
