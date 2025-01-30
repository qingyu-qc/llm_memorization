import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Initialize the vLLM
llm = LLM(
    model="epfl-llm/meditron-70b",
    tensor_parallel_size=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_seq_len_to_capture=2048,
)

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=1024)


def perform_inference_batch(inputs):
    # Generate output using the VLLM's generate method
    generated = llm.generate(inputs, sampling_params)

    # Extract the generated text for each input
    return [output.outputs[0].text.strip() for output in generated]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run inference with Meditron model.')
    parser.add_argument('--input_file', type=str,
                        required=True, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str,
                        required=True, help='Path to the output JSON file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing inputs')

    args = parser.parse_args()

    # Load the dataset
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_results = []
    batch_size = args.batch_size
    batch_inputs = []

    # Execute inference in batches
    for i, entry in enumerate(tqdm(data, desc="Processing entries")):
        prompt = entry["prompt"]
        input_text = entry["input"]
        combined_text = f"{prompt}\n\n{input_text}"
        batch_inputs.append(combined_text)

        # If batch size is reached or it's the last entry, perform inference
        if len(batch_inputs) == batch_size or i == len(data) - 1:
            try:
                generated_texts = perform_inference_batch(
                    batch_inputs)  # Pass the batch inputs
                for j, generated_text in enumerate(generated_texts):
                    data_idx = i - len(batch_inputs) + 1 + j
                    data[data_idx]["meditron_response"] = generated_text
                all_results.extend(data[i - len(batch_inputs) + 1: i + 1])
                batch_inputs = []  # Reset batch inputs after processing
            except Exception as e:
                print(f"Skipping batch due to error: {e}")
                batch_inputs = []  # Reset batch inputs if an error occurs
                continue

            # Save the results after processing each batch
            with open(args.output_file, 'w', encoding='utf-8') as f_out:
                json.dump(all_results, f_out, ensure_ascii=False, indent=4)

    print("Inference completed and results saved to", args.output_file)
