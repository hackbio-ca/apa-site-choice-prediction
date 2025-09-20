# apa-site-choice-prediction

Predicting Alternative Polyadenylation Site Choice from mRNA Sequences

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
test test 
## Abstract

Messenger RNA (mRNA) molecules are made from DNA and serve as the instructions for building proteins. At the end of most mRNAs is a poly(A) tail: a stretch of adenines that protects the RNA and helps it function. Genes can use different polyadenylation sites to produce mRNAs with different tail lengths, a process called Alternative Polyadenylation (APA). APA changes can alter RNA stability, localization, and translation, and are often linked to diseases such as cancer.

Our project aims to build an interpretable machine learning model to predict which polyadenylation site a gene will use, based on features derived from the RNA sequence. By identifying sequence motifs (such as AAUAAA variants), nucleotide composition, and position within the transcript, the model will highlight key factors influencing site choice.

We will train and test our model using publicly available datasets from PolyASite 2.0, a curated atlas of experimentally validated polyadenylation sites, and GENCODE, which provides high-quality gene annotations. These datasets include APA information for human and mouse, covering multiple tissues and experimental conditions. The output will include both predictions and explanations (via SHAP plots) showing which features were most important.

This tool could help researchers understand APA regulation and potentially detect disease- associated changes in RNA processing.

## License

This project is licensed under the [MIT License](LICENSE).
