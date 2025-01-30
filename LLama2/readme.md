# Dataset Processing
There are also four pre-training corpora for the Meditron model: clinical guideline, experience-replay, Pubmed abstract and Pubmed full-text.

# Pre-training dataset acquisition
Download from Onedrive repository, I'm uploading all the processed dataset [there](https://yaleedu-my.sharepoint.com/personal/anran_li_yale_edu/_layouts/15/onedrive.aspx?e=5%3A386697351f404935b77181913bc33c9c&sharingv2=true&fromShare=true&at=9&CID=fc553eb4%2D4dd3%2D469f%2Da64e%2Decde325dd87e&FolderCTID=0x012000EF8DD22F7A9DEF4AAA754653B3CEA2A8&id=%2Fpersonal%2Fanran%5Fli%5Fyale%5Fedu%2FDocuments%2FMedical%20LLMs%20Memorization%2FExperiment%20results%2FPre%2Dtrain%2FMeditron%5Fsplit%5Fdataset).

# Model inference
Replay the model in the `llama2_inference` script to llama2 model, then run the following code
```
python llama2_inference.py --input_file sampled_data.json --output_file sampled_data_chatcompletion.json --batch_size 256
```