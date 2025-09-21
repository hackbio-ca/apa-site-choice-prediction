# apa-site-choice-prediction

Predicting Alternative Polyadenylation Site Choice from mRNA Sequences

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

Our project aims to train the [`DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M) machine learning model to predict where and which polyadenylation site a gene will use, based on features derived from the RNA sequence (Qin et al., 2020). By identifying sequence motifs (such as AAUAAA variants), nucleotide composition, and position within the transcript, the model will highlight key factors influencing site choice (Tian et al., 2025).

We will train and test the model using publicly available datasets from **PolyASite 2.0** and **PolyA_DB**, which provide experimentally validated catalogs of polyadenylation sites, and **GENCODE**, which supplies comprehensive gene annotations. Together, these resources give us a high-confidence, genome-wide reference of polyadenylation sites and their transcript contexts (Zhang et al., 2021). The model’s output will include predictions and further visualizing aids are included.

This tool could help researchers understand APA regulation and potentially detect disease-associated changes in RNA processing.

Submitted as part of the [Toronto Bioinformatics Hackathon 2025](https://hackbio.ca/)

## Background 

Messenger RNA (mRNA) molecules are made from DNA and serve as the instructions for building proteins. At the end of most mRNAs is a poly(A) tail: a stretch of adenosinese that protects the RNA and helps it function. Genes can use different polyadenylation sites to produce mRNAs with different tail lengths, a process called Alternative Polyadenylation (APA) (Gallicchio et al., 2023). APA changes can alter RNA stability, localization, and translation, and are often linked to diseases such as cancer(Zhang et al., 2021).

## Workflow 

**Phase 1: Planning and Research** 

Tasks were delegated to members as follows: 
1. Raw data processing 
    - Figure out how to compile data from PolyA Site 2.0, PolyA_DB, and GENCODE into one `.csv` file
    - Determine how to format the `.csv` file depending on what columns of data are important and relevant  
2. Neural network research 
    - Research CNN vs. transformers and decide what would be more suitable for our project
    - Choose a machine learning model to feed our data into 
    - Research said model, how it works, what it does, and the necessary format for the data we feed it 
3. Presentation and visuals 
    - Create presentation template 
    - Learn how to plot cool graphs (ex. heat maps)
    - Create visualizations of chosen machine learning model 

Tasks were completed, and the decided ml model was the transformer-based genome foundation model `DNABERT-2-117M`. 

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

Trained the model on our dataset using amazon web service servers. However, the model was taking too long to run when we fed it our dataset and we were running out of time to complete the hackathon. We made the decision to cut our data in half in a last ditch effort to finish the project on time.  


**Phase 4: Evaluation and Interpreting** 

Used built-in functions from `transformers` and `scikit-learn` to evaluate model accuracy. Compiled raw BED/GTF files into one CSV (including data points like ±50-nt windows, AAUAAA/variants, GC%) for model checks. Curated a smaller set of data for demoing the tuned model and generating visualizations using BertViz. 


## Installation and Dependencies 
Follow repo_quickstart_ultra_short.md within this repository.

**Required Packages**
- Python **3.10–3.12**
- `torch` (CPU or CUDA)
- `transformers`
- `seaborn`
- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
- `genome_kit`

*Standard library* (no install needed): `re`, `typing`, `math`, `os`, `sys`, `random`, `hashlib`, `collections`, `gzip`  
*Included with torch*: `torch.utils.data` (`Dataset`, `DataLoader`)

## References 

Gallicchio, L., Olivares, G. H., Berry, C. W., & Fuller, M. T. (2023, January). Regulation and function of alternative polyadenylation in development and differentiation. RNA biology. https://pmc.ncbi.nlm.nih.gov/articles/PMC10730144/ 
Passmore, L. A., & Coller, J. (2022, February). Roles of mrna poly(a) tails in regulation of eukaryotic gene expression. Nature reviews. Molecular cell biology. https://pmc.ncbi.nlm.nih.gov/articles/PMC7614307/ 
Qin, H., Ni, H., Liu, Y., Yuan, Y., Xi, T., Li, X., & Zheng, L. (2020, July 11). RNA-binding proteins in tumor progression - journal of hematology & oncology. BioMed Central. https://jhoonline.biomedcentral.com/articles/10.1186/s13045-020-00927-w 
Tian, Q., Zou, Q., & Jia, L. (2025, May 14). Benchmarking of methods that identify alternative polyadenylation events in single-/multiple-polyadenylation site genes. NAR genomics and bioinformatics. https://pmc.ncbi.nlm.nih.gov/articles/PMC12076406/ 
Zhang, Y., Liu, L., Qiu, Q., Zhou, Q., Ding, J., Lu, Y., & Liu, P. (2021, February 1). Alternative polyadenylation: Methods, mechanism, function, and role in cancer - Journal of Experimental & Clinical Cancer Research. BioMed Central. https://jeccr.biomedcentral.com/articles/10.1186/s13046-021-01852-7 

## License

This project is licensed under the [MIT License](LICENSE).
