import random
import json
from datasets import load_dataset
import argparse
import os
import transformers


# Process the dataset with or without truncating the options from input
def process_dataset(sampled_dataset, tokenizer, target_sample_size, remove_option):
    results = []
    for i in range(len(sampled_dataset)):
        if len(results) >= target_sample_size:
            break  # Stop processing if we've reached the desired sample size

        input_text = sampled_dataset[i]['input']
        groundtruth_text = sampled_dataset[i]['output']
        prompt=sampled_dataset[i]['instruction']

        # If remove_option is set to 'yes', remove everything after '\n###Options'
        if remove_option == 'yes':
            option_index = input_text.find('\n###Options')
            if option_index != -1:
                options_text = input_text[option_index:]
                input_text = input_text[:option_index]
                groundtruth_text = options_text + groundtruth_text

        input_tokens = tokenizer.tokenize(input_text)
        groundtruth_tokens = tokenizer.tokenize(groundtruth_text)

        # Check if the output token length is at least 50
        if len(groundtruth_tokens) < 50:
            continue  # Skip this entry if the output length is less than 50 tokens

        # Convert tokens back to text
        input_text = tokenizer.convert_tokens_to_string(input_tokens)
        groundtruth_text = tokenizer.convert_tokens_to_string(groundtruth_tokens)

        # Construct the data entry
        data_entry = {
            "prompt": prompt,
            "input": input_text,
            "groundtruth": groundtruth_text,
            "original_text": input_text + " " + groundtruth_text
        }

        results.append(data_entry)

    return results

def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load the dataset
    if args.source == "chatdoc":
        with open("HealthCareMagic-100k.json", "r") as f:
            data = json.load(f)
        dataset = [{"input": entry["input"], "output": entry["output"]} for entry in data]
        filtered_dataset = dataset
    else:
        dataset = load_dataset("axiong/pmc_llama_instructions")
        filtered_dataset = dataset['train'].filter(lambda x: x['source'] == args.source)

    print(f"Total number of samples in the dataset: {len(filtered_dataset)}")

    # Sample a large number of indices from the dataset to ensure we can get the required number after processing
    initial_sample_size = int(args.sample_size * 5)
    sampled_indices = random.sample(range(len(filtered_dataset)), initial_sample_size)
    sampled_dataset = [filtered_dataset[i] for i in sampled_indices]

    # Load the tokenizer
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.tokenizer)

    # Process the dataset until we get the desired number of valid samples
    results = process_dataset(sampled_dataset, tokenizer, args.sample_size, args.remove_option)

    output_file = args.output_file
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"JSON file has been generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to retain after processing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='pmc_finetune_dataset/1k_sampled_data.json', help='Output file to save the results')
    parser.add_argument('--tokenizer', type=str, default="axiong/PMC_LLaMA_13B", help='Tokenizer model to use for tokenization')
    parser.add_argument('--remove_option', type=str, choices=['yes', 'no'], default='no', help="Whether to remove options from the input text ('yes' or 'no')")
    parser.add_argument('--source', type=str, choices=['medmcqa', 'medqa_train', 'pubmedqa.ori_pqaa', 'umls', 'umls_relation', 'chatdoc'], required=True, help="Specify the source of the dataset to filter")

    args = parser.parse_args()
    main(args)
