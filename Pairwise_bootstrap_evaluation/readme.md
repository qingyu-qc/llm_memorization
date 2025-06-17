## Model Evaluation with Bootstrap Sampling

Compare multiple model outputs using bootstrap resampling and statistical significance testing.

### Pairwise Bootstrap Evaluation

To compare models using bootstrap sampling and generate a summary CSV file, run:

```bash
python compare_models.py \
  --models model_outputs/model1.json model_outputs/model2.json model_outputs/model3.json \
  --output_csv results/model_comparison.csv \
  --bootstrap_dir bootstrap_indices/ \
  --n_samples 100 \
  --sample_size 100 \
  --seed 42
This will compute the mean, standard deviation, 95% confidence interval, and p-value (Wilcoxon rank-sum test) for each metric, and save the results to model_comparison.csv.

