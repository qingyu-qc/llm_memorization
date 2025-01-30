import json
import os
from tqdm import tqdm
import evaluate

# Initialize ROUGE evaluator
rouge = evaluate.load("rouge")


def tokenize_text(text):
    return text.split()


def compute_rouge_for_entry(groundtruth, response):
    results = rouge.compute(predictions=[response], references=[groundtruth])
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"]
    }


def evaluate_rouge(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["meditron_response"]

        # Compute ROUGE scores for each entry
        rouge_score = compute_rouge_for_entry(groundtruth, response)
        entry["rouge"] = rouge_score

        # Accumulate ROUGE scores
        rouge_scores["rouge1"].append(rouge_score["rouge1"])
        rouge_scores["rouge2"].append(rouge_score["rouge2"])
        rouge_scores["rougeL"].append(rouge_score["rougeL"])

    # Compute average ROUGE scores
    avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
    avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
    avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

    print(f"Average ROUGE-1: {avg_rouge1}")
    print(f"Average ROUGE-2: {avg_rouge2}")
    print(f"Average ROUGE-L: {avg_rougeL}")

    # Check and create output file directory if it does not exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)

    return rouge_scores


if __name__ == "__main__":
    input_file = 'results/consecutive.json'
    output_file = 'results/rouge_result.json'

    evaluate_rouge(input_file, output_file)
