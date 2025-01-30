import random
import json
import argparse
import os
from transformers import AutoTokenizer

# Function to process the dataset with specified input and output token lengths
def process_dataset(sampled_dataset, tokenizer, input_length, output_length, target_sample_size):
    results = []
    for i in range(len(sampled_dataset)):
        if len(results) >= target_sample_size:
            break  # Stop processing if we've reached the desired sample size
        
        text = sampled_dataset[i]['abstract']

        # Tokenize the text
        tokens = tokenizer.tokenize(text)

        # Check the total length of tokens
        total_length = len(tokens)

        if total_length <= input_length:
            print(f'Skipping data {i} with token length {total_length} because it has insufficient total length')
            continue  # Skip this entry if the total length is less than or equal to input_length

        input_tokens = tokens[:input_length]

        if total_length <= input_length + output_length:
            # Use all remaining tokens as output
            output_tokens = tokens[input_length:]
            # Ensure the output length is at least 50 tokens
            if len(output_tokens) < 50:
                print(f'Skipping data {i} with output token length {len(output_tokens)} because it has insufficient output length')
                continue
        else:
            output_tokens = tokens[input_length:input_length + output_length]

        input_text = tokenizer.convert_tokens_to_string(input_tokens)
        groundtruth_text = tokenizer.convert_tokens_to_string(output_tokens)
        original_text = tokenizer.convert_tokens_to_string(tokens[:input_length + output_length])

        # Construct the data entry
        data_entry = {
            "prompt": "Generate the next 500 tokens for the following text:",
            "input": input_text,
            "groundtruth": groundtruth_text,
            "original_text": original_text
        }

        results.append(data_entry)

    return results

def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load the dataset
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Sample a large number of indices from the dataset to ensure we can get the required number after processing
    initial_sample_size = int(args.sample_size * 10)  # Double the sample size as a precaution
    sampled_indices = random.sample(range(len(dataset)), initial_sample_size)
    sampled_dataset = [dataset[i] for i in sampled_indices]

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")

    # Process the dataset until we get the desired number of valid samples
    results = process_dataset(sampled_dataset, tokenizer, args.input_length, args.output_length, args.sample_size)

    # Save the results to a JSON file
    output_file = args.output_file
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"JSON file has been generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--sample_size', type=int, default=10000, help='Number of samples to retain after processing')
    parser.add_argument('--input_length', type=int, default=50, help='Number of tokens in the input text')
    parser.add_argument('--output_length', type=int, default=500, help='Number of tokens in the output text')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='1k_sampled_data.json', help='Output file to save the results')

    args = parser.parse_args()
    main(args)
# python Meditron/replay_process_token.py --input_file replay.jsonl --sample_size 10000 --input_length 50 --output_length 500 --seed 42 --output_file Meditron/dataset/replay_tokenized/10k_sample_50_token.json