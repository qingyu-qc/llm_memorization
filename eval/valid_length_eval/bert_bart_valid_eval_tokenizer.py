import json
import evaluate
import numpy as np
from BARTScore.bart_score import BARTScorer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Initialize BERTScore and BARTScore evaluators
bertscore = evaluate.load("bertscore")
bart_scorer = BARTScorer(device='cuda', checkpoint="facebook/bart-large-cnn")
bart_scorer.load(path="/rds/general/user/yy923/home/meditron/eval/bart_score.pth")  # Replace with your path

# Initialize tokenizer with a maximum length
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
max_length = 512

def truncate_tokens(text, valid_token_length=None):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    if valid_token_length:
        tokens = tokens[:valid_token_length]
    return tokenizer.convert_tokens_to_string(tokens)

def compute_scores(input_file, output_file, valid_token_length=None):
    """
    Compute and write scores including BERT and BART scores to the output file.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    bert_scores = []
    bart_scores = []

    for entry in tqdm(data, desc="Processing entries"):
        groundtruth = entry.get('groundtruth', '')
        response = entry.get('meditron_response', '')
        
        # Apply valid token length truncation if provided
        groundtruth = truncate_tokens(groundtruth, valid_token_length)
        response = truncate_tokens(response, valid_token_length)

        # Compute BERT score
        bert_result = bertscore.compute(predictions=[response], references=[groundtruth], model_type="bert-base-multilingual-cased")
        bert_score_f1 = round(bert_result["f1"][0], 4)
        bert_scores.append(bert_score_f1)

        # Compute BART score
        bart_result = bart_scorer.score(srcs=[response], tgts=[groundtruth], batch_size=1)
        bart_score = round(bart_result[0], 4)
        bart_scores.append(bart_score)

        # Store results for each entry
        entry['bert_score_f1'] = bert_score_f1
        entry['bart_score'] = bart_score

    # Calculate average scores
    avg_bert_score = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0
    avg_bart_score = sum(bart_scores) / len(bart_scores) if bart_scores else 0.0

    print(f"Average BERT Score F1: {avg_bert_score}")
    print(f"Average BART Score: {avg_bart_score}")

    # Write the updated entries with scores to the output file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_file = 'results/vllm/tokenized/replay/meditron70B/10k_sample_100_token/bleu.json'
    output_file = 'results/vllm/tokenized/replay/meditron70B/10k_sample_100_token/10k_sample_100_eval_results.json'
    valid_token_length = 100  # Set the valid token length here
    compute_scores(input_file, output_file, valid_token_length)
