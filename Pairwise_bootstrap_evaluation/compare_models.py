import os
import json
import numpy as np
import csv
from tqdm import tqdm
from scipy.stats import ranksums
import argparse

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def compute_average_metrics(data):
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'bert_score_f1', 'bart_score']
    aggregated_metrics = {metric: [] for metric in metrics}
    similarity_count = 0

    for entry in data:
        rouge = entry.get('rouge', {})
        aggregated_metrics['rouge1'].append(rouge.get('rouge1', 0))
        aggregated_metrics['rouge2'].append(rouge.get('rouge2', 0))
        aggregated_metrics['rougeL'].append(rouge.get('rougeL', 0))
        aggregated_metrics['bleu'].append(entry.get('bleu', 0))
        aggregated_metrics['bert_score_f1'].append(entry.get('bert_score_f1', 0))
        aggregated_metrics['bart_score'].append(entry.get('bart_score', 0))
        if entry.get('similarity', False):  
            similarity_count += 1

    average_metrics = {metric: round(np.mean(values), 4) for metric, values in aggregated_metrics.items()}
    ratio = round(similarity_count / len(data), 4) if len(data) > 0 else 0.0
    average_metrics['similarity_ratio'] = ratio

    return average_metrics

def generate_and_save_bootstrap_samples(input_file, n_samples, sample_size, output_dir, seed=42):
    with open(input_file, 'r') as file:
        data = json.load(file)
        n_data_points = len(data)

    np.random.seed(seed)
    bootstrap_indices = [
        np.random.choice(n_data_points, sample_size, replace=True).tolist()
        for _ in range(n_samples)
    ]

    input_filename = os.path.basename(input_file).replace('.json', '')
    output_file = os.path.join(
        output_dir,
        f"{input_filename}_{n_samples}samples_{sample_size}size.csv"
    )

    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(bootstrap_indices)

    print(f"Bootstrap samples saved to {output_file}")
    return output_file

def load_bootstrap_indices(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        bootstrap_indices = [list(map(int, row)) for row in reader]
    return bootstrap_indices

def compute_percentile_confidence_interval(data, lower=2.5, upper=97.5):
    ci_lower, ci_upper = np.percentile(data, [lower, upper])
    return round(ci_lower, 4), round(ci_upper, 4)

def compare_multiple_models_with_bootstrap(model_files, output_csv, bootstrap_dir, n_samples, sample_size, seed=42):
    all_results = []

    for i in range(len(model_files)):
        for j in range(i + 1, len(model_files)):
            model_1 = model_files[i]
            model_2 = model_files[j]

            bootstrap_indices_file_1 = generate_and_save_bootstrap_samples(
                input_file=model_1,
                n_samples=n_samples,
                sample_size=sample_size,
                output_dir=bootstrap_dir,
                seed=seed
            )
            bootstrap_indices_file_2 = generate_and_save_bootstrap_samples(
                input_file=model_2,
                n_samples=n_samples,
                sample_size=sample_size,
                output_dir=bootstrap_dir,
                seed=seed
            )

            data_model_1 = load_data(model_1)
            data_model_2 = load_data(model_2)

            bootstrap_indices_1 = load_bootstrap_indices(bootstrap_indices_file_1)
            bootstrap_indices_2 = load_bootstrap_indices(bootstrap_indices_file_2)

            sampled_metrics_model_1 = []
            sampled_metrics_model_2 = []
            for indices_1, indices_2 in zip(bootstrap_indices_1, bootstrap_indices_2):
                sampled_data_1 = [data_model_1[idx] for idx in indices_1]
                sampled_data_2 = [data_model_2[idx] for idx in indices_2]

                sampled_metrics_model_1.append(compute_average_metrics(sampled_data_1))
                sampled_metrics_model_2.append(compute_average_metrics(sampled_data_2))

            metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'bert_score_f1', 'bart_score', 'similarity_ratio']
            for metric in metrics:
                model_1_values = [sample[metric] for sample in sampled_metrics_model_1]
                model_2_values = [sample[metric] for sample in sampled_metrics_model_2]

                model_1_mean = round(np.mean(model_1_values), 4)
                model_1_std = round(np.std(model_1_values), 4)
                model_1_ci_lower, model_1_ci_upper = compute_percentile_confidence_interval(model_1_values)

                model_2_mean = round(np.mean(model_2_values), 4)
                model_2_std = round(np.std(model_2_values), 4)
                model_2_ci_lower, model_2_ci_upper = compute_percentile_confidence_interval(model_2_values)

                t_stat, p_val = ranksums(model_1_values, model_2_values, nan_policy='omit')
                p_value = format(p_val, '.2e')

                all_results.append({
                    'Model 1': os.path.basename(model_1),
                    'Model 2': os.path.basename(model_2),
                    'Metric': metric,
                    'Model 1 Mean': model_1_mean,
                    'Model 1 Std': model_1_std,
                    'Model 1 CI Lower': model_1_ci_lower,
                    'Model 1 CI Upper': model_1_ci_upper,
                    'Model 2 Mean': model_2_mean,
                    'Model 2 Std': model_2_std,
                    'Model 2 CI Lower': model_2_ci_lower,
                    'Model 2 CI Upper': model_2_ci_upper,
                    'p-value': p_value
                })

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'Model 1', 'Model 2', 'Metric',
            'Model 1 Mean', 'Model 1 Std', 'Model 1 CI Lower', 'Model 1 CI Upper',
            'Model 2 Mean', 'Model 2 Std', 'Model 2 CI Lower', 'Model 2 CI Upper',
            'p-value'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

    print(f"All pairwise comparison results with percentile-based confidence intervals saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple models using bootstrap sampling.")
    parser.add_argument('--models', type=str, nargs='+', required=True, help="Paths to the JSON files of the models.")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--bootstrap_dir', type=str, required=True, help="Directory to save bootstrap indices.")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of bootstrap samples.")
    parser.add_argument('--sample_size', type=int, default=100, help="Size of each bootstrap sample.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    compare_multiple_models_with_bootstrap(
        model_files=args.models,
        output_csv=args.output_csv,
        bootstrap_dir=args.bootstrap_dir,
        n_samples=args.n_samples,
        sample_size=args.sample_size,
        seed=args.seed
    )
