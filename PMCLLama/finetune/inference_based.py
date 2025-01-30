import json
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load the tokenizer and model
tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B',torch_dtype=torch.bfloat16, device_map="cuda")

def perform_inference(input_text, max_input_length=1024, max_new_tokens=1024):
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

# Loading dataset
input_file = '2k_input_data_50medqa.json'
output_file = 'response/2k_input_data_50medqa_direct_answer.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Execute inference
all_results = []
for entry in tqdm(data, desc="Processing entries"):
    prompt = entry["prompt"]
    input_text = entry["input"]
    combined_text = f"{prompt}\n\n{input_text}"
    
    try:
        generated_text = perform_inference(combined_text)
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
