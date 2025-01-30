import json
import os
from tqdm import tqdm
import evaluate

# Initialize BLEU evaluator
bleu = evaluate.load("bleu")

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

def compute_bleu_for_entry(groundtruth, response):
    if not response.strip():
        return 0.0  # If the response is empty, return BLEU score 0
    references = [groundtruth]  # references should be a list of strings
    predictions = response  # predictions should be a string
    results = bleu.compute(predictions=[predictions], references=[references])
    return results["bleu"]

def evaluate_bleu(input_file, output_file, first_k_sentences=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bleu_scores = []

    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["pmc_response"]

        # Apply first k sentences truncation
        groundtruth = truncate_sentences(groundtruth, first_k_sentences)
        response = truncate_sentences(response, first_k_sentences)

        # Compute BLEU score for each entry
        bleu_score = compute_bleu_for_entry(groundtruth, response)
        entry["bleu"] = bleu_score

        # Accumulate BLEU scores
        bleu_scores.append(bleu_score)

    # Compute average BLEU score
    if bleu_scores:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
    else:
        avg_bleu = 0.0

    print(f"Average BLEU: {avg_bleu}")

    # Check and create output file directory if it does not exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)

    return {"bleu": avg_bleu}


if __name__ == "__main__":
    input_file = 'results/4k_100_results/4k_100_rouge.json'
    output_file = 'results/4k_100_results/4k_100_bleu.json'
    first_k_sentences = 3  # Set the number of first k sentences to evaluate

    evaluate_bleu(input_file, output_file, first_k_sentences)
