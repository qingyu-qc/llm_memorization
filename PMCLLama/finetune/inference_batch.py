import json
import torch
import transformers
from tqdm import tqdm
import argparse

def perform_inference_batch(input_texts, tokenizer, model, max_input_length=1024, max_new_tokens=1024):
    # Encode the batch of input texts
    inputs = tokenizer(input_texts, return_tensors='pt', truncation=True, max_length=max_input_length, padding=True).to("cuda")

    # Generate the output tokens with attention mask
    with torch.no_grad():
        output_tokens = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False
        )
    
    # Decode the output tokens to text
    output_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
    
    # Remove the input_text part from the generated output
    generated_texts = [output_text[len(input_texts[i]):].strip() for i, output_text in enumerate(output_texts)]
    
    return generated_texts

def main(input_file, output_file, batch_size=8):
    # Fixed parameters
    model_name = "axiong/PMC_LLaMA_13B"
    max_input_length = 1024
    max_new_tokens = 1024

    # Load the tokenizer and model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    
    # Set pad_token to eos_token if no pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")

    # Ensure the model's pad_token_id is set correctly
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Loading dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Execute inference in batches
    all_results = []
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        batch_data = data[i:i+batch_size]
        combined_texts = [f"{entry['prompt']}\n\n{entry['input']}" for entry in batch_data]
        
        try:
            generated_texts = perform_inference_batch(combined_texts, tokenizer, model, max_input_length, max_new_tokens)
            for j, entry in enumerate(batch_data):
                entry["pmc_response"] = generated_texts[j]
                all_results.append(entry)
        except torch.cuda.OutOfMemoryError:
            print(f"Skipping batch due to CUDA out of memory")
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    
    args = parser.parse_args()

    main(input_file=args.input_file, output_file=args.output_file, batch_size=args.batch_size)
