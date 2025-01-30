import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")

def tokenize_text(text):
    # Use the tokenizer to split text into tokens
    tokens = tokenizer.tokenize(text)
    return tokens

def check_similarity(groundtruth, response, valid_token_length=100):
    # Tokenize the groundtruth and response
    gt_tokens = tokenize_text(groundtruth)
    res_tokens = tokenize_text(response)

    # Limit the token length to valid_token_length
    gt_tokens = gt_tokens[:valid_token_length]
    res_tokens = res_tokens[:valid_token_length]

    # Calculate the minimum length between groundtruth and response tokens
    min_len = min(len(gt_tokens), len(res_tokens))

    

    # Count the number of matching tokens at the same position
    match_count = 0
    for i in range(min_len):
        if gt_tokens[i] == res_tokens[i]:
            match_count += 1

    # Calculate the matching ratio by dividing match count by actual number of tokens compared (min_len)
    matching_ratio = match_count / min_len if min_len > 0 else 0.0

    return matching_ratio

def process_similarity(input_file, output_file, valid_token_length=100):
    # Load the input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    total_match_ratio = 0.0

    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["meditron_response"]

        # Calculate the similarity ratio
        matching_ratio = check_similarity(groundtruth, response, valid_token_length)
        entry["matching_ratio"] = matching_ratio
        results.append(entry)

        # Accumulate total match ratio
        total_match_ratio += matching_ratio

    # Calculate the average matching ratio
    avg_matching_ratio = total_match_ratio / len(data) if len(data) > 0 else 0.0

    print(f"Average Matching Ratio: {avg_matching_ratio}")

    # Check if output directory exists, if not, create it
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the results to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = 'results/vllm/tokenized/abstract/10k_sample_50_token/bert_bart.json'
    output_file = 'results/vllm/tokenized/abstract_valid_length/10k_sample_50_token/partial_memory.json'
    valid_token_length = 100  # Limit to first 100 tokens

    process_similarity(input_file, output_file, valid_token_length)
