import re
import json
import argparse

def check_exact_match(groundtruth, pmc_response):
    # Remove the "###Answer" part from the response, only keep the options part
    pmc_response = pmc_response.split("###Answer")[0]

    # Use regex to extract option content, ignoring prefixes like A. B. C.
    pmc_options = re.findall(r'[A-Z]\.\s(.*?)(?=[A-Z]\.|$)', pmc_response, re.DOTALL)

    # Normalize the groundtruth by stripping spaces and converting to lowercase
    groundtruth_normalized = groundtruth.strip().lower()

    # Loop through the model-generated options and check for exact matches
    for option in pmc_options:
        option_normalized = option.strip().lower()  # Normalize options
        if option_normalized == groundtruth_normalized:
            return True

    return False

def check_fuzzy_match(groundtruth, pmc_response):
    # Remove the "###Answer" part from the response, only keep the options part
    pmc_response = pmc_response.split("###Answer")[0]

    # Use regex to extract option content, ignoring prefixes like A. B. C.
    pmc_options = re.findall(r'[A-Z]\.\s(.*?)(?=[A-Z]\.|$)', pmc_response, re.DOTALL)

    # Normalize the groundtruth by stripping spaces and converting to lowercase
    groundtruth_normalized = groundtruth.strip().lower()

    # Loop through the model-generated options and check for fuzzy matches
    for option in pmc_options:
        option_normalized = option.strip().lower()  # Normalize options
        # Check if groundtruth is contained within the option
        if groundtruth_normalized in option_normalized:
            return True

    return False

def process_samples(samples):
    exact_match_num=0
    fuzzy_match_num=0
    for sample in samples:
        groundtruth = sample.get('groundtruth')
        pmc_response = sample.get('pmc_response')

        # Check for exact and fuzzy matches
        exact_match = check_exact_match(groundtruth, pmc_response)
        fuzzy_match = check_fuzzy_match(groundtruth, pmc_response)
        if exact_match:
            exact_match_num+=1
        if fuzzy_match:
            fuzzy_match_num+=1

        # Add the match results to the sample
        sample['exact_match'] = exact_match
        sample['fuzzy_match'] = fuzzy_match

    return samples,exact_match_num,fuzzy_match_num


def main(args):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
        
    processed_samples,exact_match_num,fuzzy_match_num = process_samples(samples)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=4)
    print(f"exact match number is {exact_match_num} and fuzzy match number is {fuzzy_match_num}")

    print(f"Processed data has been saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSON file to check for exact and fuzzy matches in PMC response.")
    
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")

    parser.add_argument('--output_file', type=str, required=True, help="Path to save the processed JSON file.")
    
    args = parser.parse_args()
    
    main(args)