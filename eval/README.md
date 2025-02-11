### BERT and BART evalution metric
To evaluate using BERTScore and BARTScore, you will need to follow the commands below:
1. Make a package
```
touch __init__.py
```
2. Clone the [BARTScore repository](https://github.com/neulab/BARTScore).
3. Create Initialization File for BARTScore
```
touch BARTScore/__init__.py
```
4. Add BARTScorer to the Initialization File
Edit BARTScore/`__init__.py` to include the following line:
```
from .bart_score import BARTScorer
```
5. Download the `bart_score.pth` model file from [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) for BART metrics evaluation. 
