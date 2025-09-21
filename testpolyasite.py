import genome_kit as gk
from genome_kit import Genome
from genome_kit import Interval

#!/usr/bin/env python3
import os

# --- 1) Load genomes ---
# Use hg19 because you're using gencode.v19 (which annotates hg19).
ref = Genome("hg19")             # reference FASTA (DNA comes from here)
gencode = Genome("gencode.v19")  # annotations (genes/transcripts/UTRs)

# Quick sanity check: get DNA + overlapping genes at an example window
iv_ref = Interval("chr1", "+", 100000, 100050, ref)
print("DNA sanity:", ref.dna(iv_ref))
iv_anno = Interval("chr1", "+", 100000, 100050, gencode)
print("Genes overlapping:", [g.id for g in gencode.genes.find_overlapping(iv_anno)])

# --- 2) Resolve BED path,get polyasite data
here = os.path.dirname(__file__)
bed_path = os.path.join(here, "data", "atlas.clusters.2.0.GRCh38.96.bed")

def pas_center(start: int, end: int, strand: str) -> int:
    # Common convention: end on '+' / start on '-', pass the center of strand
    return end if strand == "+" else start

# --- 3) Read the FIRST BED row, extract ±50nt window sequence, print info ---
with open(bed_path) as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        chrom, start, end, name, score, strand = line.strip().split()[:6]
        start, end = int(start), int(end)
        center = pas_center(start, end, strand)

        # Build interval for DNA from the reference genome
        iv = Interval(chrom, strand, center - 50, center + 50, ref)
        seq = ref.dna(iv)

        # Optional: annotate with gene(s) from GENCODE
        iv_g = Interval(chrom, strand, iv.start, iv.end, gencode)
        genes = gencode.genes.find_overlapping(iv_g)
        gene_id = genes[0].id if genes else None

        print("Name:", name)
        print("Chrom:", chrom, "Strand:", strand, "Center:", center)
        print("Gene:", gene_id)
        print("Sequence:", seq)
        break  # remove this break