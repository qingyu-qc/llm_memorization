import json
import os
from tqdm import tqdm
import evaluate

# Initialize ROUGE evaluator
rouge = evaluate.load("rouge")

def truncate_sentences(text, first_k_sentences=None):
    """
    Truncate the text to the first k sentences.
    """
    if first_k_sentences:
        # Split the text by sentences and take the first k sentences
        sentences = text.split('.')[:first_k_sentences]
        truncated_text = '.'.join(sentences).strip() + '.'  # Re-add the period at the end
    else:
        truncated_text = text
    return truncated_text

def compute_rouge_for_entry(groundtruth, response):
    results = rouge.compute(predictions=[response], references=[groundtruth])
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"]
    }

def evaluate_rouge(input_file, output_file, first_k_sentences=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["pmc_response"]

        # Apply first k sentences truncation
        groundtruth = truncate_sentences(groundtruth, first_k_sentences)
        response = truncate_sentences(response, first_k_sentences)

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
    input_file = 'results/4k_100_results/4k_100_consecutive.json'
    output_file = 'results/4k_100_results/4k_100_rouge.json'
    first_k_sentences = 3  

    evaluate_rouge(input_file, output_file, first_k_sentences)
