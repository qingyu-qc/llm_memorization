import random
import json
from datasets import load_dataset
import argparse
import os


def process_dataset(sampled_dataset, input_length, output_length, random_start):
    results = []
    for i in range(len(sampled_dataset)):
        clean_text = sampled_dataset[i]['clean_text']

        # Split the text into words
        words = clean_text.split()

        # Check the total length of words
        total_length = len(words)

        if total_length < input_length + 50:
            # Skip if there aren't enough words for at least minimal lengths
            print(
                f'Skipping data {i} with word length {total_length} because it has insufficient total length')
            continue

        if random_start:
            # Randomly select a start position for input_length
            start_idx = random.randint(0, total_length - input_length)
        else:
            # Start from the beginning
            start_idx = 0

        # Ensure there's enough length for input
        if start_idx + input_length > total_length:
            print(
                f'Skipping data {i} because starting index exceeds text length')
            continue

        input_end_idx = start_idx + input_length
        output_end_idx = min(input_end_idx + output_length, total_length)

        input_words = words[start_idx:input_end_idx]
        output_words = words[input_end_idx:output_end_idx]

        if len(output_words) < 50:
            print(
                f'Skipping data {i} with output word length {len(output_words)} because it has insufficient output length')
            continue

        input_text = ' '.join(input_words)
        groundtruth_text = ' '.join(output_words)
        original_text = ' '.join(words[:output_end_idx])

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
    dataset = load_dataset("epfl-llm/guidelines")

    # Sample specified number of indices from the dataset
    sampled_indices = random.sample(
        range(len(dataset["train"])), args.sample_size)
    sampled_dataset = dataset["train"].select(sampled_indices)

    # Process the dataset
    results = process_dataset(
        sampled_dataset, args.input_length, args.output_length, args.random_start)

    # Save the results to a JSON file
    output_file = args.output_file

    # Check if output directory exists, if not, create it
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"JSON file has been generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help = 'Number of samples to draw from the dataset')
    parser.add_argument('--input_length', type=int, default=50,
                        help = 'Number of words in the input text')
    parser.add_argument('--output_length', type=int, default=500,
                        help = 'Number of words in the output text')
    parser.add_argument('--seed', type=int, default=42,
                        help = 'Random seed for reproducibility')
    parser.add_argument('--output_file', type=str,
                        default = '1k_sampled_data.json', help = 'Output file to save the results')
    parser.add_argument('--random_start', action='store_true',
                        help = 'Randomly select the start position for input text')

    args = parser.parse_args()
    main(args)
