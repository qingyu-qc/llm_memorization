import json
from tqdm import tqdm
import os
import transformers

# Initialize the tokenizer
tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')

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

def tokenize_text(text):
    # Use the tokenizer to split text into tokens
    tokens = tokenizer.tokenize(text)
    return tokens

def check_similarity(groundtruth, response, n=50, first_k_sentences=None):
    # Truncate both groundtruth and response to the first k sentences
    groundtruth = truncate_sentences(groundtruth, first_k_sentences)
    response = truncate_sentences(response, first_k_sentences)

    # Tokenize the truncated texts
    gt_tokens = tokenize_text(groundtruth)
    res_tokens = tokenize_text(response)

    gt_len = len(gt_tokens)
    res_len = len(res_tokens)

    if gt_len < n or res_len < n:
        return False, None

    # Check for n consecutive matching tokens
    for i in range(gt_len - n + 1):
        gt_ngram = gt_tokens[i:i+n]
        for j in range(res_len - n + 1):
            res_ngram = res_tokens[j:j+n]
            if gt_ngram == res_ngram:
                matched_string = tokenizer.convert_tokens_to_string(gt_ngram)
                return True, matched_string
    return False, None

def process_similarity(input_file, output_file, n=50, first_k_sentences=None):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    match_count = 0
    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry["groundtruth"]
        response = entry["pmc_response"]

        # Check similarity based on the first k sentences
        is_similar, matched_string = check_similarity(groundtruth, response, n, first_k_sentences)
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
    first_k_sentences = 3  # Set the number of first k sentences to evaluate

    process_similarity(input_file, output_file, n, first_k_sentences)
