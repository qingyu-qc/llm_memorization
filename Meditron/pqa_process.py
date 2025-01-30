import random
import json
from datasets import load_dataset
from tqdm import tqdm

# Load the dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial")

# Sample 1000 entries
sample_size = 1000
sampled_dataset = ds['train'].shuffle(seed=42).select(range(sample_size))

# Create the first dataset (split question dataset)
split_question_dataset = []

for entry in tqdm(sampled_dataset, desc="Processing split question dataset"):
    question = entry['question']
    words = question.split()
    mid_index = len(words) // 2
    input_text = ' '.join(words[:mid_index])
    groundtruth_text = ' '.join(words[mid_index:])
    
    split_question_entry = {
        "input": input_text,
        "groundtruth": groundtruth_text
    }
    split_question_dataset.append(split_question_entry)

# Save the first dataset
output_file_1 = 'question_split_dataset.json'
with open(output_file_1, 'w', encoding='utf-8') as f:
    json.dump(split_question_dataset, f, ensure_ascii=False, indent=4)

print(f"Split question dataset saved to {output_file_1}")

# Create the second dataset (context to question dataset)
context_question_dataset = []

for entry in tqdm(sampled_dataset, desc="Processing context question dataset"):
    context = ' '.join(entry['context']['contexts'])
    question = entry['question']
    
    context_question_entry = {
        "input": context,
        "groundtruth": question
    }
    context_question_dataset.append(context_question_entry)

# Save the second dataset
output_file_2 = 'context_question_dataset.json'
with open(output_file_2, 'w', encoding='utf-8') as f:
    json.dump(context_question_dataset, f, ensure_ascii=False, indent=4)

print(f"Context to question dataset saved to {output_file_2}")
