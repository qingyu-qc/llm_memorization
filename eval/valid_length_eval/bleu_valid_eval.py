import json
import os
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer

# Initialize BLEU evaluator
bleu = evaluate.load("bleu")
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")  # local path

def tokenize_text(text):
    # Use the tokenizer to split text into tokens
    tokens = tokenizer.tokenize(text)
    return tokens

def truncate_tokens(text, valid_token_length=None):
    tokens = tokenize_text(text)
    if valid_token_length:
        tokens = tokens[:valid_token_length]
    return tokenizer.convert_tokens_to_string(tokens)

def compute_bleu_for_entry(groundtruth, response):
    if not response.strip():
        return 0.0  # If the response is empty, return BLEU score 0
    references = [groundtruth]  # references should be a list of strings
    predictions = response  # predictions should be a string
    results = bleu.compute(predictions=[predictions], references=[references])
    return results["bleu"]

def evaluate_bleu(input_file, output_file, valid_token_length=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bleu_scores = []

    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["meditron_response"]

        # Apply valid token length if provided
        groundtruth = truncate_tokens(groundtruth, valid_token_length)
        response = truncate_tokens(response, valid_token_length)

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
    valid_token_length = 100  

    evaluate_bleu(input_file, output_file, valid_token_length)
