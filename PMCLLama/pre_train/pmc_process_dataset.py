import json
import argparse

def process_dataset(sampled_dataset, input_length, output_length):
    results = []
    for i in range(len(sampled_dataset)):
        clean_text = sampled_dataset[i]['abstract_and_body']
        pmcid = sampled_dataset[i]['pmcid']
        title = sampled_dataset[i]['title']

        # Split the text into words
        words = clean_text.split()

        # Check the total length of words
        total_length = len(words)

        if total_length <= input_length:
            print(f'Skipping data {i} with word length {total_length} because it has insufficient total length')
            continue  # Skip this entry if the total length is less than or equal to input_length

        input_words = words[:input_length]
        input_text = ' '.join(input_words)

        if total_length <= input_length + output_length:
            # Use all remaining words as output
            output_words = words[input_length:]
        else:
            output_words = words[input_length:input_length + output_length]

        groundtruth_text = ' '.join(output_words)
        original_text = ' '.join(words[:input_length + output_length])

        # Construct the data entry
        data_entry = {
            "pmcid": pmcid,
            "title": title,
            "prompt": "Generate the next 500 tokens for the following text:",
            "input": input_text,
            "groundtruth": groundtruth_text,
            "original_text": original_text
        }

        results.append(data_entry)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PMC articles dataset.")
    parser.add_argument('--input_json', type=str, default='pmc_intermediate.json', help='Input JSON file with extracted articles.')
    parser.add_argument('--output_json', type=str, required=True, help='Output JSON file with processed dataset.')
    parser.add_argument('--input_length', type=int, default=50, help='Input length for model.')
    parser.add_argument('--output_length', type=int, default=500, help='Output length for model.')

    args = parser.parse_args()

    # Load articles from intermediate JSON file
    with open(args.input_json, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    processed_articles = process_dataset(articles, args.input_length, args.output_length)

    # Save processed articles to a JSON file
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=4)

