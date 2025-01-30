from transformers import LlamaTokenizer
from evaluate import load
import json
import os
import argparse
from tqdm import tqdm
import torch
import transformers
from perplexity import Perplexity
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the perplexity evaluator
#perplexity = load("perplexity", module_type="metric")

def evaluate_perplexity(input_file, output_file, model_name):
    # Load input data from JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        #tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name) 
        tokenizer = AutoTokenizer.from_pretrained("yahma/llama-7b-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("pad_token was not set; using eos_token as pad_token.")
    except Exception as e:
        print(f"Error loading LlamaTokenizer for model {model_name}: {e}")
        return

    # Verify tokenizer was loaded correctly
    if not tokenizer or isinstance(tokenizer, bool):
        print("Tokenizer failed to load or is not a valid tokenizer. Please check the model name and try again.")
        return

    # Initialize the custom perplexity evaluator
    perplexity_evaluator = Perplexity()

    # Tokenize each entry's pmc_response and groundtruth, truncate to first 100 tokens
    pmc_responses = []
    groundtruths = []
    skipped_indices = []

    for idx, entry in enumerate(data):
        if "llama2_response" not in entry or "groundtruth" not in entry:
            print(f"Missing 'llama2_response' or 'groundtruth' in entry at index {idx}.")
            skipped_indices.append(idx)
            continue

        response = entry["llama2_response"].strip()
        groundtruth = entry["groundtruth"].strip()

        if not response or not groundtruth:
            print(f"Skipping empty response or groundtruth at index {idx}.")
            skipped_indices.append(idx)
            continue

        pmc_tokens = tokenizer.tokenize(response)[:100]
        groundtruth_tokens = tokenizer.tokenize(groundtruth)[:100]

        if len(pmc_tokens) == 0 or len(groundtruth_tokens) == 0:
            print(f"Skipping entry with empty tokens at index {idx}.")
            skipped_indices.append(idx)
            continue

        pmc_responses.append(tokenizer.convert_tokens_to_string(pmc_tokens))
        groundtruths.append(tokenizer.convert_tokens_to_string(groundtruth_tokens))

    pmc_results = perplexity_evaluator._compute(
        predictions=pmc_responses,
        model_id=model_name,
        batch_size=16,
        add_start_token=True
    )

    groundtruth_results = perplexity_evaluator._compute(
        predictions=groundtruths,
        model_id=model_name,
        batch_size=16,
        add_start_token=True
    )

    pmc_index = 0
    groundtruth_index = 0
    for i, entry in enumerate(data):
        if i in skipped_indices:
            entry["llama2_response_perplexity"] = None
            entry["groundtruth_perplexity"] = None
        else:
            entry["llama2_response_perplexity"] = pmc_results["perplexities"][pmc_index]
            entry["groundtruth_perplexity"] = groundtruth_results["perplexities"][groundtruth_index]
            pmc_index += 1
            groundtruth_index += 1

    # Create output directory if it does not exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"Output file '{output_file}' was successfully saved.")
        print(f"File size: {file_size} bytes.")
    else:
        print(f"Failed to save the output file '{output_file}'. Please check the code and directory permissions.")

    # Print average perplexity values
    print(f"Average llama2_response Perplexity: {pmc_results['mean_perplexity']}")
    print(f"Average Groundtruth Perplexity: {groundtruth_results['mean_perplexity']}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Perplexity for PMC Responses and Groundtruths")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to be used for perplexity calculation")

    args = parser.parse_args()
    evaluate_perplexity(args.input_file, args.output_file, args.model_name)
