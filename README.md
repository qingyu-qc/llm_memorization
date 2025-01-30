# Data Memorization of Medical LLMs
This repository investigate the problem of privacy and memorization in large language models (LLMs).

## Install Meditron Environment
To set up the environment, follow the steps below:
```
conda create -n meditron python=3.10
conda activate meditron
pip install -r requirements.txt
```

## Originial datasets
### Meditron Pre-train Datasets 
The following datasets are used for Meditron pre-training: 
- Clinical Guidelines
- Paper Abstracts
- Medical Papers
- Replay dataset
You can find these datasets in the [gap-replay directory](https://github.com/epfLLM/meditron/tree/main/gap-replay).

### PMCLLaMA Datasets
For PMCLLaMa, the following datasets are available:
- [pre-trained dataset](./PMCLLama/readme.md). We provide the source and a detailed procedure for processing the pre-trained datasets.
- [fine-tuning dataset](https://huggingface.co/datasets/axiong/pmc_llama_instructions).

## Pre-processed datasets
We provide multiple links to pre-processed versions datasets for convenience:

### Meditron

1. Initial pre-processed datasets: [Access here](https://yaleedu-my.sharepoint.com/:f:/r/personal/anran_li_yale_edu/Documents/Medical%20LLMs%20Memorization/Experiment%20results/Pre-train/Initial_datasets?csf=1&web=1&e=abtPU6)

2. Processed datasets: [Access here](https://yaleedu-my.sharepoint.com/:f:/r/personal/anran_li_yale_edu/Documents/Medical%20LLMs%20Memorization/Experiment%20results/Pre-train/Meditron_split_dataset?csf=1&web=1&e=cxprDj)

3. Fine-tuned Sampled Datasets [Access here](https://yaleedu-my.sharepoint.com/:f:/r/personal/anran_li_yale_edu/Documents/Medical%20LLMs%20Memorization/Experiment%20results/Fine-tune/sampled_dataset?csf=1&web=1&e=lpQklH)


### PMCLLaMA
1. Prcessed fine-tuned datasets: [Access here](https://yaleedu-my.sharepoint.com/:f:/r/personal/anran_li_yale_edu/Documents/Medical%20LLMs%20Memorization/Experiment%20results/Fine-tune/sampled_dataset/PMCLLaMA_finetuned_dataset?csf=1&web=1&e=vpR1FJ)

2. Processed pre-trained datasets: [Access here](https://yaleedu-my.sharepoint.com/:f:/r/personal/anran_li_yale_edu/Documents/Medical%20LLMs%20Memorization/Experiment%20results/Pre-train/PMCLLaMA_split_dataset?csf=1&web=1&e=dplsms)

## Dataset Processing
We use two main strategies for sampling the text data:
1. Sequential Sampling from the First Word
   - Script: [guidelines_process_token.py](./Meditron/guidelines_process_token.py)
   - This script processes data starting from the very beginning of each document.

2. Sequential Sampling from a Random Starting Point
   - Script: [random_dataset.py](./Meditron/random_dataset.py)
   - This script chooses a random position within each document to begin sampling. 

## Model Inference
We leverage  [vLLM](https://github.com/vllm-project/vllm) to speed up the inference. For details, please refer to [vllm_inference.py](./Meditron/vllm_inference.py).

## Evaluation results:
Detailed evaluation results for all experiments can be found in this [Google spreadsheet](https://docs.google.com/spreadsheets/d/1cbOuZKMctm0PAj3LCwNYm2mJBz-tFvfkHrGIHNxxGow/edit?usp=sharing).

## Evaluation metrics:
After running model inference, the generated responses are stored in a dedicated response folder. We then apply various evaluation metrics using the script [eval_all.py](./eval/full_eval/eval_all.py), which includes:
1. Top-n Consecutive Token Match
2. ROUGE Score
3. BLEU Score
4. BERTScore
5. BARTScore

### BERT and BART evalution metric
To evaluate using BERTScore and BARTScore, you will need to:
1. Clone the [BARTScore repository](https://github.com/neulab/BARTScore) into your current directory.
2. Download the `bart_score.pth` model file from [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) for BART metrics evaluation. 


### Valid Length Evaluation
We add two kinds of evaluation here: evaluating first-K tokens and first-K sentences.

1. First-K tokens comparison
    - Script: [eval_valid_all.py](./eval/valid_length_eval/eval_valid_all.py)
    - Evaluates the similarity between model output and GT for the first K tokens.

2. First-K sentences comparison
    - Script: [eval_valid_firstsen.py](./eval/valid_sent_eval/eval_valid_firstsen.py)
    - Evaluates the similarity for the first K sentences of the text.

### Partial memorization metric
The partial memorization function measures how closely a model's output matches the groundtruth at the token level within the first 100 tokens, focusing on exact matches at corresponding positions. Please refer to [partial_memorization_eval](./partial_memorization_eval.py).


## License
This repository is provided under the [MIT License](#) (or whichever license applies). Please refer to the `LICENSE` file for details.