import json
import argparse
import transformers
import torch

def process_dataset(sampled_dataset, tokenizer,input_length, output_length):
    results = []
    for i in range(len(sampled_dataset)):
        clean_text = sampled_dataset[i]['abstract_and_body']
        pmcid = sampled_dataset[i]['pmcid']
        title = sampled_dataset[i]['title']

        # Tokenize the text
        tokens = tokenizer.tokenize(clean_text)

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
    parser.add_argument('--output_json', type=str, default='pmc_article_500token.json', help='Output JSON file with processed dataset.')
    parser.add_argument('--input_length', type=int, default=500, help='Input length for model.')
    parser.add_argument('--output_length', type=int, default=500, help='Output length for model.')

    args = parser.parse_args()

    # Load articles from intermediate JSON file
    with open(args.input_json, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    tokenizer = transformers.LlamaTokenizer.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")

    processed_articles = process_dataset(articles, tokenizer,args.input_length, args.output_length)

    # Save processed articles to a JSON file
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_articles, f, ensure_ascii=False, indent=4)

