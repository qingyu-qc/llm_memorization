import os
import argparse
from consecutive_valid_token_eval import process_similarity
from rouge_valid_eval import evaluate_rouge
from bleu_valid_eval import evaluate_bleu
from bert_bart_valid_eval_tokenizer import compute_scores


def main(input_file, output_dir, n, valid_token_length):
    # Ensure output directory exists
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Consecutive Evaluation
    cons_output = os.path.join(output_dir, 'consecutive.json')
    if not os.path.exists(cons_output):
        process_similarity(input_file, cons_output, n, valid_token_length)
    else:
        print(f"Skipping Consecutive Evaluation, {cons_output} already exists.")

    # 2. ROUGE Evaluation
    rouge_output = os.path.join(output_dir, 'rouge.json')
    if not os.path.exists(rouge_output):
        evaluate_rouge(cons_output, rouge_output, valid_token_length)
    else:
        print(f"Skipping ROUGE Evaluation, {rouge_output} already exists.")

    # 3. BLEU Evaluation
    bleu_output = os.path.join(output_dir, 'bleu.json')
    if not os.path.exists(bleu_output):
        evaluate_bleu(rouge_output, bleu_output, valid_token_length)
    else:
        print(f"Skipping BLEU Evaluation, {bleu_output} already exists.")

    # 4. BERT & BART Scores
    bert_bart_output = os.path.join(output_dir, 'bert_bart.json')
    if not os.path.exists(bert_bart_output):
        compute_scores(bleu_output, bert_bart_output, valid_token_length)
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
    parser.add_argument("--valid_token_length", type=int, default=None,
                        help="Maximum valid token length to consider for evaluation.")

    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.n, args.valid_token_length)

# Example command:
# python eval_all.py --input_file response/replay_70B/10k_sample_50_token.json --output_dir results/vllm/tokenized/replay/meditron70B/10k_sample_50_token/ --n 50 --valid_token_length 100
