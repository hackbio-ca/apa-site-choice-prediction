#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make_dataset.py — combine positives + negatives into one ML-ready CSV

import os
import sys
import pandas as pd

# ---------------- Paths (script-relative) ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

POS_CSV = os.path.join(OUTD, "apasites_positives.csv")
NEG_CSV = os.path.join(OUTD, "apasites_negatives.csv")
OUT_CSV = os.path.join(OUTD, "apasites_dataset.csv")

# Expected schema (negatives now match positives)
COLS = ["id","chrom","pos","strand","gene_name","source","sequence","label"]

def read_csv_or_die(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.stderr.write(f"Missing file: {path}\n")
        sys.exit(1)
    df = pd.read_csv(path)
    # add any missing expected columns
    for c in COLS:
        if c not in df.columns:
            df[c] = "" if c not in ("pos","label") else pd.NA
    # type fixes
    df["chrom"] = df["chrom"].astype(str)
    df["strand"] = df["strand"].astype(str)
    # coerce numerics safely
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    if "label" in df:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    return df[COLS]

def main():
    pos = read_csv_or_die(POS_CSV)
    neg = read_csv_or_die(NEG_CSV)

    # Ensure labels are correct (force, just in case)
    pos["label"] = 1
    neg["label"] = 0
    # Ensure sources are set
    pos.loc[pos["source"].isna() | (pos["source"]==""), "source"] = "positive"
    neg["source"] = "negative"

    # Basic cleanups: drop rows missing essential fields
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["chrom","pos","strand"]).copy()
        # normalize sequence to uppercase; allow blanks for negatives if you kept them empty
        df["sequence"] = df["sequence"].astype(str).str.upper()
        return df

    pos = clean(pos)
    neg = clean(neg)

    # Concatenate and de-dup by genomic coordinate + strand (+ label)
    out = pd.concat([pos, neg], ignore_index=True)
    out = out.drop_duplicates(subset=["chrom","pos","strand","label"]).reset_index(drop=True)

    # Final column order
    out = out[COLS]

    out.to_csv(OUT_CSV, index=False)
    print(f"✔ Combined dataset: {len(out):,} rows -> {OUT_CSV}")
    print(f"   Positives: {int((out['label']==1).sum()):,} | Negatives: {int((out['label']==0).sum()):,}")

if __name__ == "__main__":
    main()
