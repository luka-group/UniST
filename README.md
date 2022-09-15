# UniST
Code for our paper [Unified Semantic Typing with Meaningful Label Inference](https://arxiv.org/abs/2205.01826) at NAACL 2022.

## Requirements
* pytorch == 1.9.1
* transformers == 4.10.2
* scikit-learn

## Data

### UFET
Download [UFET dataset](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html) and place the crowdsourced portion under `data/ufet`.

### TACRED
Download [TACRED dataset](https://nlp.stanford.edu/projects/tacred/) to `data/tacred`.

### MAVEN 
Download [MAVEN dataset](https://github.com/THU-KEG/MAVEN-dataset) to `data/maven`.

### FewRel
Download [FewRel dataset](https://github.com/thunlp/FewRel) to `data/fewrel` and run 
```bash
python data/fewrel/process_fewrel.py
```

## Experiments
The training scripts are provided under `./scripts`. For example, to train a UniST base model on TACRED, run
```bash
bash ./scripts/run_tacred.sh
```