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

def check_similarity(groundtruth, response, n=50):
    gt_tokens = tokenize_text(groundtruth)
    res_tokens = tokenize_text(response)

    gt_len = len(gt_tokens)
    res_len = len(res_tokens)

    if gt_len < n or res_len < n:
        return False, None

    for i in range(gt_len - n + 1):
        gt_ngram = gt_tokens[i:i+n]
        for j in range(res_len - n + 1):
            res_ngram = res_tokens[j:j+n]
            if gt_ngram == res_ngram:
                matched_string = ' '.join(gt_ngram)
                return True, matched_string
    return False, None

def process_similarity(input_file, output_file, n=50):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    match_count = 0
    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["meditron_response"]

        is_similar, matched_string = check_similarity(groundtruth, response, n)
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
    input_file = 'response/vllm/tokenized/4k_sample_50_token.json'
    output_file = 'results/vllm/tokenized/clinical_guideline/meditron7B/4k_sample_50_token/consecutive_token_50.json'
    n = 50  # Number of consecutive matching tokens

    process_similarity(input_file, output_file, n)
