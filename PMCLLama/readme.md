# Dataset Processing
There are also two training corpora for the PMCLLama model: pre-training and fine-tuning. The model used for pretraining is from https://huggingface.co/chaoyi-wu/PMC_LLAMA_7B and model used for fine-tuning is from https://huggingface.co/axiong/PMC_LLaMA_13B.
## Pre-training data processing
The data used for pre-training include the PMC open source articles. The method for processing the pre-training data can also be devided into the sequential sampling from first word and sequential sampling from random start point。
### Sequential sampling from first word
To obtain the PMC open source articles we firstly need to download the csv file that include all the PMCid of the articles. 
```
wget ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv
```
Then we randomly select sample size PMCid and extract the setences from the abstract and body using the pmc_extract_articles.py
```
python pmc_extract_articles.py --csv_path ./oa_file_list.csv --download_dir ./downloads --extract_dir ./extracted_articles --num_samples 11000 --random_seed 42 --word_limit 2000 --intermediate_json ./pmc_intermediate.json
```
This command will run the script with the parameters we specify, download and process 11000 sample articles, download them to the ./downloads directory, extract them to the ./extracted_articles directory, and store the extracted text content in the ./pmc_intermediate.json file.

After selecting the acticles we can use the following code to process the data
```
python pmc_process_dataset_token.py --input_json ./pmc_intermediate.json --output_json ./pmc_article_50token.json --input_length 50 --output_length 500
```
This command will run the script with the parameters you specified, read the extracted articles from ./pmc_intermediate.json, save them to ./pmc_article_50token.json after processing, and use the model with an input length of 50 token and an output length of 500. If you want the input to be seperated using the space you can replace the pmc_process_dataset_token.py with the pmc_process_dataset.py 

# Model inference
## Pre-training model inference
The pmc_article_completion.py can be used to generate the pmc model answers and the pmc_article_completion_batch.py using the batch size as input which can speed up the inference.
```
python pmc_article_completion_batch.py --input_file ./pmc_article_50token.json --output_file response/pmc_response_50token.json --batch_size 8
```

## Fine-tuning model inference
For medqa dataset, we employ [fine-tuned_meditron70b_medqa](https://huggingface.co/hippocrates/fine-tuned_meditron70b_medqa) for inference. Since there is no tokenizer files in the Huggingface repository, we need to manually copy the meditron tokenizer or llama2 tokenizer files into the local huggingface hub.
```
# Huggingface login
huggingface-cli login

# Download the fine-tuned model
huggingface-cli download hippocrates/fine-tuned_meditron70b_medqa

# Manually copy the tokenizer files into the huggingface hub.

# Model inference
run finetune/finetune_inference.py --input_file /gpfs/home/yy923/LLM_privacy/Meditron/dataset/medqa/1k_sample_50.json --output_file /gpfs/home/yy923/LLM_privacy/Meditron/output/vllm/tokenized/medqa/1k_sample_50.json --batch_size 256
```
Processed medqa dataset can be found [onedrive](https://yaleedu-my.sharepoint.com/:f:/r/personal/anran_li_yale_edu/Documents/Medical%20LLMs%20Memorization/Experiment%20results/Finetuning/MedQA/MedQA_datasets?csf=1&web=1&e=tVg94i)
