#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, gzip
import pandas as pd
from genome_kit import Genome, Interval

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

BED_PATH = os.path.join(DATA, "atlas.clusters.2.0.GRCh38.96.bed")  # PolyASite (hg38)
GTF_PATH = os.path.join(DATA, "gencode.v49.annotation.gtf")        # uncompressed
OUT_CSV  = os.path.join(OUTD, "polyasite_annotated_hg38.csv")

ref = Genome("hg38")

CANON = {f"chr{i}" for i in range(1,23)} | {"chrX","chrY","chrM"}

def load_bed(path: str) -> pd.DataFrame:
    # Force chrom to str to avoid mixed dtype; keep first 6 BED fields
    cols = ["chrom","start","end","name","score","strand"]
    df = pd.read_csv(
        path, sep="\t", header=None, comment="#", usecols=range(6),
        names=cols, dtype={0: str}, low_memory=False
    )

    # Normalize chromosome names: make string, ensure 'chr' prefix
    def norm_chr(c):
        c = str(c).strip()
        return c if c.startswith("chr") else ("chr" + c)

    df["chrom"] = df["chrom"].map(norm_chr)

    # Keep only canonical chroms; PolyASite may include scaffolds
    df = df[df["chrom"].isin(CANON)].copy()

    # Ensure strand is '+' or '-'
    df["strand"] = df["strand"].astype(str).where(df["strand"].isin(["+","-"]), "+")

    # PAS position: end for '+', start for '-'
    df["pos"] = df.apply(lambda r: int(r["end"]) if r["strand"] == "+" else int(r["start"]), axis=1)
    return df

def parse_gtf(gtf_path: str) -> pd.DataFrame:
    def opener(p): 
        return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "r")
    rows = []
    with opener(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attrs = parts
            if chrom not in CANON:  # match filtering with sites
                continue
            a = {}
            for kv in attrs.strip().strip(";").split(";"):
                kv = kv.strip()
                if not kv or " " not in kv:
                    continue
                k, v = kv.split(" ", 1)
                a[k] = v.strip().strip('"')
            rows.append({
                "chrom": chrom,
                "start": int(start),
                "end": int(end),
                "strand": strand,
                "gene_id": a.get("gene_id", ""),
                "gene_name": a.get("gene_name", ""),
                "gene_type": a.get("gene_type", a.get("gene_biotype", "")),
            })
    return pd.DataFrame(rows)

def annotate_with_genes(sites: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    # Avoid column name collisions; we output gene columns as g_*.
    genes = genes.rename(columns={"start": "g_start", "end": "g_end"})
    out = []
    for chrom, dfc in sites.groupby("chrom"):
        g = genes[genes["chrom"] == chrom]
        if g.empty:
            out.append(dfc.assign(gene_id="", gene_name="", gene_type=""))
            continue
        g_sorted = g.sort_values(["g_start","g_end"]).reset_index(drop=True)
        m = pd.merge_asof(
            dfc.sort_values("pos"),
            g_sorted[["g_start","g_end","gene_id","gene_name","gene_type"]],
            left_on="pos", right_on="g_start", direction="backward"
        )
        m = m[(m["pos"] <= m["g_end"]) & (m["pos"] >= m["g_start"])]

        # Fallback for any misses
        miss = dfc.loc[~dfc.index.isin(m.index)]
        if len(miss):
            def row_hit(p):
                hit = g[(g.g_start <= p.pos) & (p.pos <= g.g_end)]
                if len(hit):
                    h = hit.iloc[0]
                    return pd.Series([h.gene_id, h.gene_name, h.gene_type])
                return pd.Series(["","",""])
            extra = miss.apply(row_hit, axis=1)
            miss = miss.assign(gene_id=extra[0], gene_name=extra[1], gene_type=extra[2])
            out.append(pd.concat([m, miss], axis=0).sort_index())
        else:
            out.append(m.sort_index())

    return pd.concat(out, axis=0).sort_index()


def seq_window(chrom: str, strand: str, center: int, flank: int = 101) -> str:
    # center must be int; chrom must be str
    center = int(center)
    start = max(0, center - flank)
    end   = center + flank + 1
    iv = Interval(str(chrom), strand, start, end, ref)
    return ref.dna(iv)

def main():
    print("Loading PolyASite BED…")
    bed = load_bed(BED_PATH)
    bed["pos"] = bed["pos"].astype(int)
    bed["chrom"] = bed["chrom"].astype(str)

    print("Parsing GENCODE (gtf)…")
    genes = parse_gtf(GTF_PATH)

    print("Annotating…")
    ann = annotate_with_genes(bed[["chrom","pos","strand"]], genes)

    print("Extracting sequences…")
    ann = ann.assign(
        sequence = ann.apply(lambda r: seq_window(r.chrom, r.strand, r.pos), axis=1),
        source   = "PolyASite"
    )

    cols = ["chrom","pos","strand","gene_id","gene_name","gene_type","source","sequence","name","start","end","score"]
    cols = [c for c in cols if c in ann.columns]
    ann[cols].to_csv(OUT_CSV, index=False)
    print(f"✔ Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
