from openai import OpenAI
import numpy as np
import glob
from typing import List
from tqdm import tqdm
CLIENT = OpenAI()

def get_accuracy(result_files: List[str]):
    samples = []
    condense_accuracy = []
    causal_accuracy = []
    condense_tokens = []
    causal_tokens = []
    for result_file in tqdm(result_files, desc="Loading files"):
        with open(result_file, "r") as f:
            result = json.load(f)
        samples.append(result)
    for item in tqdm(samples, desc="Processing samples"):
        try:
            condense_response = item['condensed']['response']
            causal_response = item['causal']['response']
            answers = item['answers']

            condense_tokens.append(item['n_seen_tokens'])
            causal_tokens.append(item['context_length'])

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"The question is {item['question']}. The answer is {answers[0]}.\n Return 'yes' if the response is correct, otherwise return 'no'.\nResponse: {condense_response}"}
            ]
            response = CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )
            print(response.choices[0].message.content)

            condense_accuracy.append('yes' in response.choices[0].message.content.lower())

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"The question is {item['question']}. The answer is {answers[0]}.\n Return 'yes' if the response is correct, otherwise return 'no'.\nResponse: {causal_response}"}
            ]

            response = CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0
            )

            causal_accuracy.append('yes' in response.choices[0].message.content.lower())
        except Exception as e:
            print(f"Error processing sample {item}: {str(e)}")
            continue

    return np.mean(condense_accuracy), np.mean(causal_accuracy), np.mean(condense_tokens), np.mean(causal_tokens), np.std(condense_tokens), np.std(causal_tokens)

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    result_files = glob.glob("results/simonjegou/ruler/4096/*.json")
    condense_acc, causal_acc, condense_tokens, causal_tokens, condense_std, causal_std = get_accuracy(result_files)
    print(f"Condense Accuracy: {condense_acc:.2f}")
    print(f"Causal Accuracy: {causal_acc:.2f}")
    print(f"Condense Tokens: {condense_tokens:.2f} ± {condense_std:.2f}")
    print(f"Causal Tokens: {causal_tokens:.2f} ± {causal_std:.2f}")

    # Plot accuracy results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(['Condense', 'Causal'], [condense_acc, causal_acc])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    
    # Plot token counts with error bars
    plt.subplot(1, 2, 2)
    plt.bar(['Condense', 'Causal'], [condense_tokens, causal_tokens], 
            yerr=[condense_std, causal_std], capsize=5)
    plt.title('Token Usage')
    plt.ylabel('Number of Tokens')
    
    plt.tight_layout()
    plt.savefig('results/comparison_plots.png')
    plt.close()

