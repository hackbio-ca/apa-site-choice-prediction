#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, gzip
import pandas as pd
from genome_kit import Genome, Interval

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

PAS_TXT  = os.path.join(DATA, "human.PAS.txt")              # PolyA_DB (hg19)
CHAIN    = os.path.join(DATA, "hg19ToHg38.over.chain")      # uncompressed
GTF_PATH = os.path.join(DATA, "gencode.v49.annotation.gtf") # uncompressed
OUT_CSV  = os.path.join(OUTD, "polyadb_annotated_hg38.csv")

ref_hg38 = Genome("hg38")

def liftover_batch(chrom, pos, strand):
    try:
        from pyliftover import LiftOver
    except Exception as e:
        print("pyliftover is required. `conda install -c bioconda pyliftover`.\n", e, file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(CHAIN):
        print(f"Missing chain file: {CHAIN}", file=sys.stderr); sys.exit(1)
    lo = LiftOver(CHAIN)
    out_c, out_p, out_s = [], [], []
    for c, p, s in zip(chrom, pos, strand):
        lifts = lo.convert_coordinate(c, int(p)-1, strand=s if s in ["+","-"] else None)
        if not lifts:
            out_c.append(None); out_p.append(None); out_s.append(None); continue
        c2, p0, s2, _ = lifts[0]
        out_c.append(c2); out_p.append(int(p0)+1); out_s.append(s2 if s2 in ["+","-"] else s)
    return pd.DataFrame({"chrom_hg38": out_c, "pos_hg38": out_p, "strand_hg38": out_s})

def parse_gtf_genes(gtf_path: str) -> pd.DataFrame:
    def opener(p): 
        return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "r")
    rows = []
    with opener(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene": continue
            chrom, _, _, start, end, _, strand, _, attrs = parts
            a = {}
            for kv in attrs.strip().strip(";").split(";"):
                kv = kv.strip()
                if not kv or " " not in kv: continue
                k, v = kv.split(" ", 1)
                a[k] = v.strip().strip('"')
            rows.append({
                "chrom": chrom, "start": int(start), "end": int(end), "strand": strand,
                "gene_id": a.get("gene_id",""), "gene_name": a.get("gene_name",""),
                "gene_type": a.get("gene_type", a.get("gene_biotype","")),
            })
    return pd.DataFrame(rows)

def load_polyadb_pas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    chr_col   = pick("chr","chrom","chromosome")
    pos_col   = pick("pos","position","site","pas")
    strandcol = pick("strand","str")
    if pos_col is None:
        start_col, end_col = pick("start"), pick("end")
        if chr_col and start_col and end_col and strandcol:
            df["pos"] = df.apply(lambda r: int(r[end_col]) if r[strandcol] == "+" else int(r[start_col]), axis=1)
        else:
            raise ValueError("Could not infer PAS position in human.PAS.txt")
    else:
        df["pos"] = df[pos_col].astype(int)
    if strandcol is None or chr_col is None:
        raise ValueError("Missing chromosome/strand columns in human.PAS.txt")
    df = df.rename(columns={chr_col: "chrom", strandcol: "strand"})
    if not df["chrom"].iloc[0].startswith("chr"):
        df["chrom"] = "chr" + df["chrom"].astype(str)
    # Tag these as hg19 to avoid name collisions later
    df = df.rename(columns={"chrom":"chrom_hg19","pos":"pos_hg19","strand":"strand_hg19"})
    return df[["chrom_hg19","pos_hg19","strand_hg19"]].copy()

def annotate_overlap(sites: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    out = []
    for chrom, dfc in sites.groupby("chrom"):
        g = genes[genes["chrom"] == chrom]
        if g.empty:
            out.append(dfc.assign(gene_id="", gene_name="", gene_type="")); continue
        m = pd.merge_asof(
            dfc.sort_values("pos"),
            g.sort_values("start")[["start","end","gene_id","gene_name","gene_type"]].rename(columns={"start":"g_start"}),
            left_on="pos", right_on="g_start", direction="backward"
        )
        m = m[(m["pos"] <= m["end"]) & (m["pos"] >= m["g_start"])]
        miss = dfc.loc[~dfc.index.isin(m.index)]
        if len(miss):
            def row_hit(p):
                hit = g[(g.start <= p.pos) & (p.pos <= g.end)]
                if len(hit):
                    h = hit.iloc[0]; return pd.Series([h.gene_id,h.gene_name,h.gene_type])
                return pd.Series(["","",""])
            extra = miss.apply(row_hit, axis=1)
            miss = miss.assign(gene_id=extra[0], gene_name=extra[1], gene_type=extra[2])
            out.append(pd.concat([m, miss], axis=0).sort_index())
        else:
            out.append(m.sort_index())
    return pd.concat(out, axis=0).sort_index()

def seq_window(chrom, strand, center, flank=101):
    start = max(0, int(center) - flank)
    end   = int(center) + flank + 1
    iv = Interval(chrom, strand, start, end, ref_hg38)
    return ref_hg38.dna(iv)

def main():
    print("Loading PolyA_DB (hg19)…")
    pas_hg19 = load_polyadb_pas(PAS_TXT)  # columns: chrom_hg19, pos_hg19, strand_hg19
    print(f"PolyA_DB rows (hg19): {len(pas_hg19):,}")

    print("Lifting hg19 → hg38…")
    lifted = liftover_batch(pas_hg19["chrom_hg19"], pas_hg19["pos_hg19"], pas_hg19["strand_hg19"])

    # Combine and keep only hg38 coordinates for downstream steps
    merged = pd.concat([pas_hg19.reset_index(drop=True), lifted.reset_index(drop=True)], axis=1)
    merged = merged.dropna(subset=["chrom_hg38","pos_hg38"]).copy()

    # Canonical hg38 column names; hg19 columns remain as *_hg19 (no collisions)
    merged = merged.rename(columns={"chrom_hg38":"chrom","pos_hg38":"pos","strand_hg38":"strand"})
    merged["pos"] = merged["pos"].astype(int)

    print("Parsing GENCODE genes…")
    genes = parse_gtf_genes(GTF_PATH)

    print("Annotating…")
    sites = merged[["chrom","pos","strand"]].copy()
    ann = annotate_overlap(sites, genes)

    print("Extracting sequences…")
    ann = ann.assign(sequence=ann.apply(lambda r: seq_window(r.chrom, r.strand, r.pos), axis=1),
                     source="PolyA_DB")

    cols = ["chrom","pos","strand","gene_id","gene_name","gene_type","source","sequence"]
    ann[cols].to_csv(OUT_CSV, index=False)
    print(f"✔ Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
