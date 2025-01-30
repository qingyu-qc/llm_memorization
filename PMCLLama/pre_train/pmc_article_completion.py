import json
import torch
import transformers
from tqdm import tqdm
import argparse

def perform_inference(input_text, tokenizer, model, max_input_length=1024, max_new_tokens=1024):
    # Encode the input text with a limit on maximum length
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length).to("cuda")

    # Generate the output token with attention mask
    with torch.no_grad():
        output_tokens = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False
        )
    
    # Decode the output tokens to text
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Remove the input_text part from the generated output
    generated_text = output_text[len(input_text):].strip()
    
    return generated_text

def main(input_file, output_file):
    # Fixed parameters
    model_name = "chaoyi-wu/PMC_LLAMA_7B"
    max_input_length = 1024
    max_new_tokens = 1024

    # Load the tokenizer and model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")

    # Loading dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Execute inference
    all_results = []
    for entry in tqdm(data, desc="Processing entries"):
        prompt = entry["prompt"]
        input_text = entry["input"]
        combined_text = f"{prompt}\n\n{input_text}"
        
        try:
            generated_text = perform_inference(combined_text, tokenizer, model, max_input_length, max_new_tokens)
            entry["pmc_response"] = generated_text
            all_results.append(entry)
        except torch.cuda.OutOfMemoryError:
            print(f"Skipping entry due to CUDA out of memory: {entry}")
            torch.cuda.empty_cache()
            continue

        # Save after each batch
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, ensure_ascii=False, indent=4)

    print("Inference completed and results saved to", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using a pre-trained language model on a dataset.")
    
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file.")
    
    args = parser.parse_args()

    main(input_file=args.input_file, output_file=args.output_file)
