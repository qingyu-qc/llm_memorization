import os
import argparse
import transformers
from consecutive_sen_token_eval import process_similarity
from rouge_sen_eval import evaluate_rouge
from bleu_sen_eval import evaluate_bleu
from bert_bart_sen_eval_tokenizer import compute_scores

# Initialize the tokenizer
tokenizer = transformers.LlamaTokenizer.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")#"epfl-llm/meditron-7b" and "chaoyi-wu/PMC_LLAMA_7B"

def tokenize_text(text):
    # Use the tokenizer to split text into tokens
    tokens = tokenizer.tokenize(text)
    return tokens
def find_continuous_matches(gt_ft, res_ft):
    match_length = 0
    gt_len = len(gt_ft)
    res_len = len(res_ft)

    for i in range(min(gt_len, res_len)):
        if gt_ft[i] == res_ft[i]:
            match_length += 1
        else:
            break  

    if gt_ft == res_ft and gt_len > 0 and res_len > 0:
        first_match = 'true'
        matched_tokens = gt_ft  
    else:
        first_match = 'false'
        matched_tokens = gt_ft[:match_length] if match_length > 0 else []

    return first_match, matched_tokens, match_length

    
def process_data(data):
    total_match_length = 0
    num_records = 0

    for record in data:
        if 'groundtruth' in record and 'pmc_response' in record:
            # Extract first sentences
            gt_first = record['groundtruth'].split('.')[0].strip()
            res_first = record['pmc_response'].split('.')[0].strip()
            
            # Tokenize the first sentences
            gt_ft = tokenize_text(gt_first)
            res_ft = tokenize_text(res_first)
            
            # Find continuous matches starting from the beginning
            first_match, matched_tokens, match_length = find_continuous_matches(gt_ft, res_ft)
            
            # Store the matches in the record for further analysis
            record['first_match']=first_match
            record['matched_tokens'] = matched_tokens
            record['match_length'] = match_length
            
            # Accumulate the match length for averaging
            if first_match=='true':
                total_match_length += match_length
                num_records += 1
    
    # Calculate the average match length
    average_match_length = total_match_length / num_records if num_records > 0 else 0
    return data, average_match_length,total_match_length,num_records

def main(input_file, output_dir, n, valid_token_length):
    # Ensure output directory exists
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
     # Load and process input file
    with open(input_file, 'r') as f:
        data = json.load(f)
        processed_data, average_match_length,total_match_length,num_records = process_data(data)
    
    # Save processed input to a new file
    processed_input_file = os.path.join(output_dir, 'processed_input.json')
    with open(processed_input_file, 'w') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"Average match length: {average_match_length}")
    print(f"Total_match_length: {total_match_length}")
    print(f"Num_records: {num_records}")
    # 1. Consecutive Evaluation
    cons_output = os.path.join(output_dir, 'consecutive.json')
    if not os.path.exists(cons_output):
        process_similarity(processed_input_file, cons_output, n, first_k_sentences)
    else:
        print(f"Skipping Consecutive Evaluation, {cons_output} already exists.")

    # 2. ROUGE Evaluation
    rouge_output = os.path.join(output_dir, 'rouge.json')
    if not os.path.exists(rouge_output):
        evaluate_rouge(cons_output, rouge_output, first_k_sentences)
    else:
        print(f"Skipping ROUGE Evaluation, {rouge_output} already exists.")

    # 3. BLEU Evaluation
    bleu_output = os.path.join(output_dir, 'bleu.json')
    if not os.path.exists(bleu_output):
        evaluate_bleu(rouge_output, bleu_output, first_k_sentences)
    else:
        print(f"Skipping BLEU Evaluation, {bleu_output} already exists.")

    # 4. BERT & BART Scores
    bert_bart_output = os.path.join(output_dir, 'bert_bart.json')
    if not os.path.exists(bert_bart_output):
        compute_scores(bleu_output, bert_bart_output, first_k_sentences)
    else:
        print(f"Skipping BERT & BART Scores, {bert_bart_output} already exists.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all evaluations.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSON file with data.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output JSON files.")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of consecutive matching words for similarity check.")
    parser.add_argument("--first_k_sentences", type=int, default=None,
                        help="Maximum valid sentences to consider for evaluation.")

    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.n, args.valid_token_length)

# Example command:
# python eval_all.py --input_file response/replay_70B/10k_sample_50_token.json --output_dir results/vllm/tokenized/replay/meditron70B/10k_sample_50_token/ --n 50 --first_k_sentences 3
