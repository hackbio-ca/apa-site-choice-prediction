# apa-site-choice-prediction

Predicting Alternative Polyadenylation Site Choice from mRNA Sequences

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

Our project aims to train the `zhihan1996/DNABERT-2-117M` machine learning model to predict where and which polyadenylation site a gene will use, based on features derived from the RNA sequence. By identifying sequence motifs (such as AAUAAA variants), nucleotide composition, and position within the transcript, the model will highlight key factors influencing site choice.

We will train and test the model using publicly available datasets from **PolyASite 2.0** and **PolyA_DB**, which provide experimentally validated catalogs of polyadenylation sites, and **GENCODE**, which supplies comprehensive gene annotations. Together, these resources give us a high-confidence, genome-wide reference of polyadenylation sites and their transcript contexts. _The model’s output will include both predictions and explanations showing which sequence features were most important for site choice._ ** check what outputs are included, and modify based off of that (placeholder)

This tool could help researchers understand APA regulation and potentially detect disease-associated changes in RNA processing.

Submitted as part of the [Toronto Bioinformatics Hackathon 2025](https://hackbio.ca/)

## Background 

Messenger RNA (mRNA) molecules are made from DNA and serve as the instructions for building proteins. At the end of most mRNAs is a poly(A) tail: a stretch of adenosinese that protects the RNA and helps it function. Genes can use different polyadenylation sites to produce mRNAs with different tail lengths, a process called Alternative Polyadenylation (APA). APA changes can alter RNA stability, localization, and translation, and are often linked to diseases such as cancer.

## Workflow 

**Phase 1: Planning and Research** 

Tasks were delegated to members as follows: 
1. Raw data processing 
    - Figure out how to compile data from PolyASite 2.0, PolyA_DB, and GENCODE into one `.csv` file
    - Determine how to format the `.csv` file depending on what columns of data are important and relevant  
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

We chose **PolyASite 2.0** and **PolyA_DB** because they provide experimentally validated catalogs of human and mouse polyadenylation sites across multiple tissues and experimental conditions. These are the gold-standard references for APA studies, ensuring our model learns from real, biologically relevant examples rather than predictions.  

**GENCODE** was used for high-quality gene annotations (including transcript boundaries, strand orientation, and exon/intron structures). This ensured that each polyadenylation site could be contextualized relative to its host gene and transcript.

From these datasets, we kept the following columns because they are most relevant for predicting APA site choice:  
- **chromosome, strand, start/end position** → defines site location  
- **gene_id / transcript_id** → links the site to its gene context  
- **PAS motif (sequence window)** → captures canonical and non-canonical motifs  
- **distance to stop codon / transcript end** → reflects positional bias in APA usage  

We compiled these sources into one standardized `.csv` using **genome_kit**, which allowed us to lift coordinates to hg38, extract surrounding RNA sequence windows, and harmonize the annotation fields. The result is a unified dataset ready for sequence-based model input.



**Phase 2: Preparation for Model Training** 

`prep_rna_for_model.ipynb` was written with functions to prep raw RNA sequences for the chosen ml model.


**Phase 3: Model Training** 

_INSERT WHAT BORIS DID HERE_ 

However, the model was taking too long to run when we fed it our dataset and we were running out of time to complete the hackathon. We made the decision to cut our data in half in a last ditch effort to finish the project on time.  


**Phase 4: Evaluation and Interpreting** 

_INSERT WHAT AKSHITA DID HERE_ 


## Installation and Dependencies 
Download all project files in a directory.

**Required Packages**
- Python **3.10–3.12**
- `torch` (CPU or CUDA)
- `transformers`
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
- `genome_kit`

*Standard library* (no install needed): `re`, `typing`, `math`, `os`, `sys`, `random`, `hashlib`, `collections`, `gzip`  
*Included with torch*: `torch.utils.data` (`Dataset`, `DataLoader`)

## Quick Start 


## License

This project is licensed under the [MIT License](LICENSE).
