import genome_kit as gk
from genome_kit import Genome
from genome_kit import Interval

ref = Genome("hg19")

import pandas as pd
pas = pd.read_csv("data/human.PAS.txt", sep="\t")
print(pas.head())

def make_window(chrom, pos, strand, flank=50):
    return Interval(chrom, strand, pos-flank, pos+flank, ref)

row = pas.iloc[0]
iv = make_window(row["Chromosome"], row["Position"], row["Strand"])
seq = ref.dna(iv)
print(seq)