# apa-site-choice-prediction

Predicting Alternative Polyadenylation Site Choice from mRNA Sequences

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

Our project aims to train the `zhihan1996/DNABERT-2-117M` machine learning model to predict where and which polyadenylation site a gene will use, based on features derived from the RNA sequence. By identifying sequence motifs (such as AAUAAA variants), nucleotide composition, and position within the transcript, the model will highlight key factors influencing site choice.

We will train and test our model using publicly available datasets from PolyASite 2.0 and PolyADB, curated atlases of experimentally validated polyadenylation sites, and GENCODE, which provides high-quality gene annotations. These datasets include APA information for human and mouse, covering multiple tissues and experimental conditions. The output will include both predictions and explanations (via SHAP plots) showing which features were most important.

This tool could help researchers understand APA regulation and potentially detect disease-associated changes in RNA processing.

## Background 

Messenger RNA (mRNA) molecules are made from DNA and serve as the instructions for building proteins. At the end of most mRNAs is a poly(A) tail: a stretch of adenines that protects the RNA and helps it function. Genes can use different polyadenylation sites to produce mRNAs with different tail lengths, a process called Alternative Polyadenylation (APA). APA changes can alter RNA stability, localization, and translation, and are often linked to diseases such as cancer.

## Workflow 

**Phase 1: Planning and Research** 

Tasks were delegated to members as follows: 
1. Raw data processing 
    - Figure out how to compile data from PolyASite 2.0, PolyA_DB, and GENCODE into one .csv file
    - Determine how to format the .csv file depending on what columns of data are important and relevant  
2. Neural network research 
    - Research CNN vs. transformers and decide what would be more suitable for our project
    - Choose a machine learning model to feed our data into 
    - Research said model, how it works, what it does, and the necessary format for the data we feed it 
3. Presentation and visuals 
    - Create presentation template 
    - Learn how to plot cool graphs (ex. heat maps)
    - Create visualizations of chosen machine learning model 

Tasks were completed, and the decided ml model was the transformer-based genome foundation model `zhihan1996/DNABERT-2-117M`. 

Why a transformer: Self-attention captures long-range motif interactions (AAUAAA, UGUA, U/G-rich) across tens–hundreds of bases; pretrained genomic representations and parallelism make training fast and interpretable for k-mer attributions.

Why `DNABERT-2-117M`: Genome-native k-mer tokenizer, ~117M params (hackathon-friendly), easy loading (`AutoTokenizer`/`AutoModelForSequenceClassification`), strong prior for motif-based classification, and token-level attributions map cleanly to biological k-mers.

INSERT WHAT COLUMNS OF DATA WE KEPT HERE AND WHY


**Phase 2: Preparation for Model Training** 

`prep_rna_for_model.ipynb` was written with functions to prep raw RNA sequences for the chosen ml model. 

INSERT WHAT BORIS DID HERE 

INSERT WHAT AKSHITA DID HERE 


**Phase 3: Model Training** 


**Phase 4: Evaluation and Interpreting** 


**Phase 5: Wrapping Up** 


## Installation and Dependencies 
Download all project files in a directory.

**Required Packages**
- Python **3.10–3.12**
- `torch` (CPU or CUDA)
- `transformers`
- `pandas`, `numpy`
- `GenomeKit`

*Standard library* (no install needed): `re`, `typing`  
*Included with torch*: `torch.utils.data` (`Dataset`, `DataLoader`)

## Quick Start 


## License

This project is licensed under the [MIT License](LICENSE).
