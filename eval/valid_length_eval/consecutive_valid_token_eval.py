import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer

# Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b") # local path

def tokenize_text(text):
    # Use the tokenizer to split text into tokens
    tokens = tokenizer.tokenize(text)
    return tokens

def check_similarity(groundtruth, response, n=50, valid_token_length=None):
    gt_tokens = tokenize_text(groundtruth)
    res_tokens = tokenize_text(response)

    # Apply the valid token length limit if provided
    if valid_token_length is not None:
        gt_tokens = gt_tokens[:valid_token_length]
        res_tokens = res_tokens[:valid_token_length]

    gt_len = len(gt_tokens)
    res_len = len(res_tokens)

    if gt_len < n or res_len < n:
        return False, None

    for i in range(gt_len - n + 1):
        gt_ngram = gt_tokens[i:i+n]
        for j in range(res_len - n + 1):
            res_ngram = res_tokens[j:j+n]
            if gt_ngram == res_ngram:
                matched_string = tokenizer.convert_tokens_to_string(gt_ngram)
                return True, matched_string
    return False, None

def process_similarity(input_file, output_file, n=50, valid_token_length=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    match_count = 0
    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["meditron_response"]

        is_similar, matched_string = check_similarity(groundtruth, response, n, valid_token_length)
        entry["similarity"] = is_similar
        entry["matched_string"] = matched_string
        results.append(entry)
        
        if is_similar:
            match_count += 1
            print(f"Match found: {matched_string}")
    
    print(f"Total entries with at least {n} consecutive matching tokens: {match_count}")

    # Check if output directory exists, if not, create it
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    input_file = 'response/full_text_70B/10k_sample_50_token.json'
    output_file = 'results/70B_vllm/full_text/10k_sample_50_token/consecutive_token_50.json'
    n = 50  # Number of consecutive matching tokens
    valid_token_length = 100  

    process_similarity(input_file, output_file, n, valid_token_length)

