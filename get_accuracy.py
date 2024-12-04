from openai import OpenAI
import numpy as np
import glob
from typing import List
from tqdm import tqdm
import argparse
import json

CLIENT = OpenAI()


def get_accuracy(result_files: List[str], methods: List[str]):
    samples = []
    accuracies = {method: [] for method in methods}
    tokens = {method: [] for method in methods}

    for result_file in tqdm(result_files, desc="Loading files"):
        with open(result_file, "r") as f:
            result = json.load(f)
        samples.append(result)

    for item in tqdm(samples, desc="Processing samples"):
        try:
            for method in methods:
                if method not in item:
                    continue

                response = item[method]["response"]
                if method == "condensed":
                    tokens[method].append(item["condensed_length"])
                elif method == "causal":
                    tokens[method].append(item["context_length"])

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"The question is {item['question']}. Acceptable answers are {item['answers']}. The correct response don't need to exactly match the answer, but it should be a reasonable answer.\n Return 'yes' if the response is correct, otherwise return 'no'.\nResponse: {response}",
                    },
                ]

                response = CLIENT.chat.completions.create(
                    model="gpt-4", messages=messages, temperature=0
                )

                accuracies[method].append(
                    "yes" in response.choices[0].message.content.lower()
                )

        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            continue

    results = {}
    for method in methods:
        if accuracies[method]:  # Only include if we have data
            results[f"{method}_accuracy"] = np.mean(accuracies[method])
            if tokens[method]:  # Only include token stats if available
                results[f"{method}_tokens_mean"] = np.mean(tokens[method])
                results[f"{method}_tokens_std"] = np.std(tokens[method])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate accuracy for different methods"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["condensed"],
        help="Methods to evaluate (condensed, causal, press)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/simonjegou/ruler/4096",
        help="Directory containing result files",
    )

    args = parser.parse_args()
    result_files = glob.glob(f"{args.results_dir}/*.json")

    results = get_accuracy(result_files, args.methods)

    # Print results
    for key, value in results.items():
        if "accuracy" in key:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
