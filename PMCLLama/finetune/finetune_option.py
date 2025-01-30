import random
import json
from datasets import load_dataset
import argparse
import os
import transformers

def process_dataset(sampled_dataset, target_sample_size):
    results = []
    
    for i in range(len(sampled_dataset)):
        if len(results) >= target_sample_size:
            break  # Stop processing if we've reached the desired sample size

        input_text = sampled_dataset[i]['input']
        #skip question without options
        if '###Options:\n' not in input_text:
            continue  
        
        question_part, options_part = input_text.split('###Options:\n')
        options = options_part.split('\n')
        options = [opt for opt in options if opt] 

        #skip sample with less than 4 options
        if len(options) < 4:
            continue

        option_dict = {}
        for opt in options:
            if '. ' not in opt:
                continue 
            prefix, option_text = opt.split('. ', 1)
            option_dict[prefix] = option_text

        if not option_dict:
            continue

        correct_option_prefix = random.choice(list(option_dict.keys()))
        correct_option = f"{option_dict[correct_option_prefix]}"

        if not correct_option.strip():
            continue

        remaining_options = [(key, option_dict[key]) for key in option_dict if key != correct_option_prefix]

        new_prefixes = ['A', 'B', 'C']
        formatted_options = "\n".join([f"{new_prefixes[i]}. {remaining_options[i][1]}" for i in range(len(remaining_options))])
        new_input_text = f"{question_part}###Options:\n{formatted_options}"

        data_entry={
            "prompt": "You are a doctor, kindly generate one other option based on the patient's description and the other three options.",
            'input': new_input_text,
            'groundtruth': correct_option,
            "original_text": input_text
        }
        results.append(data_entry)

    return results

def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)

    dataset = load_dataset("axiong/pmc_llama_instructions")
    filtered_dataset = dataset['train'].filter(lambda x: x['source'] == args.source)

    print(f"Total number of samples in the dataset: {len(filtered_dataset)}")

    # Sample a large number of indices from the dataset to ensure we can get the required number after processing
    initial_sample_size = int(args.sample_size * 5)
    sampled_indices = random.sample(range(len(filtered_dataset)), initial_sample_size)
    sampled_dataset = [filtered_dataset[i] for i in sampled_indices]

    # Process the dataset until we get the desired number of valid samples
    results = process_dataset(sampled_dataset,args.sample_size)

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
    parser.add_argument('--source', type=str, choices=['medmcqa', 'medqa_train'], required=True, help="Specify the source of the dataset to filter")
    args = parser.parse_args()
    main(args)
