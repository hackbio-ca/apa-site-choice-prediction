#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
from hashlib import blake2b

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

IN1 = os.path.join(OUTD, "polyasite_annotated_hg38.csv")
IN2 = os.path.join(OUTD, "polyadb_annotated_hg38.csv")
OUT = os.path.join(OUTD, "apasites_positives.csv")

def make_id(row) -> str:
    s = f"{row.chrom}:{row.pos}:{row.strand}:{row.get('gene_name','')}"
    return blake2b(s.encode(), digest_size=8).hexdigest()

def main():
    a = pd.read_csv(IN1)
    b = pd.read_csv(IN2)
    keep = ["chrom","pos","strand","gene_name","source","sequence"]
    for df in (a,b):
        for c in keep:
            if c not in df.columns:
                df[c] = "" if c != "pos" else -1
        df["pos"] = df["pos"].astype(int)

    allx = pd.concat([a[keep], b[keep]], ignore_index=True)\
             .drop_duplicates(subset=["chrom","pos","strand","sequence"])\
             .reset_index(drop=True)
    allx["id"] = allx.apply(make_id, axis=1)
    allx = allx[["id","chrom","pos","strand","gene_name","source","sequence"]]
    allx.to_csv(OUT, index=False)
    print(f"✔ Wrote {OUT}  ({len(allx):,} rows)")

if __name__ == "__main__":
    main()
