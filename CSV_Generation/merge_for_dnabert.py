#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge PolyA_DB + PolyASite for DNABERT with strong de-dup and optional clustering.

Inputs (auto from ./output next to this script or repo root):
  - polyadb_annotated_hg38.csv
  - polyasite_annotated_hg38.csv

Outputs (to ./output):
  - merged_coord_dedup.csv         (exact coord de-dup + provenance)
  - merged_clustered_tol{TOL}.csv  (if DO_CLUSTER=True)
  - merged_unique_sequences.csv    (one row per sequence; adds n_coords_collapsed)
  - apasites_positives.csv         (legacy schema: id,chrom,pos,strand,gene_name,source,sequence)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import hashlib
import pandas as pd

# ===================== Config =====================
DO_CLUSTER = True      # collapse nearby PAS within ±TOL per (chrom,strand)
TOL = 12               # try 12–25 depending on how aggressively you want to merge clusters
WRITE_UNIQUE_SEQS = True
PDB_NAME = "polyadb_annotated_hg38.csv"
PAS_NAME = "polyasite_annotated_hg38.csv"
OUT_POSITIVES = "apasites_positives.csv"
# ==================================================


def find_io() -> Tuple[Optional[Path], Optional[Path], Path]:
    here = Path(__file__).resolve().parent
    candidates = [here / "output", here.parent / "output", Path.cwd() / "output"]
    pdb_path = pas_path = None
    out_dir: Optional[Path] = None
    for base in candidates:
        a = base / PDB_NAME
        b = base / PAS_NAME
        if pdb_path is None and a.exists():
            pdb_path = a
        if pas_path is None and b.exists():
            pas_path = b
        if out_dir is None and base.exists():
            out_dir = base
    if out_dir is None:
        out_dir = here / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
    return pdb_path, pas_path, out_dir


def require_cols(df: pd.DataFrame, needed: Iterable[str], label: str) -> None:
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"[ERROR] {label} missing columns: {miss}. Found: {list(df.columns)}")


def ensure_source(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "source" not in df.columns:
        df = df.assign(source=name)
    else:
        df["source"] = df["source"].astype(str).fillna(name)
    return df


def agg_sources(series: pd.Series) -> str:
    vals = set()
    for s in series.fillna("").astype(str):
        for x in s.split(","):
            x = x.strip()
            if x:
                vals.add(x)
    return ",".join(sorted(vals))


def internal_coord_dedup(df: pd.DataFrame) -> pd.DataFrame:
    key = ["chrom","pos","strand"]
    before = len(df)
    out = df.drop_duplicates(subset=key, keep="first").copy()
    print(f"    internal de-dup: {before:,} -> {len(out):,}  (removed {before-len(out):,})")
    return out


def merge_by_coord(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    key = ["chrom","pos","strand"]
    keep_cols = ["gene_id","gene_name","gene_type","sequence","source"]
    # carry any extra columns (first wins)
    for c in list(a.columns) + list(b.columns):
        if c not in key + keep_cols:
            keep_cols.append(c)
    df = pd.concat([a[key + [c for c in keep_cols if c in a.columns]],
                    b[key + [c for c in keep_cols if c in b.columns]]],
                   ignore_index=True)
    agg_map = {c: "first" for c in set(df.columns) - set(key)}
    agg_map["source"] = agg_sources
    merged = (df.sort_values(key).groupby(key, as_index=False).agg(agg_map))
    merged["pos"] = pd.to_numeric(merged["pos"], errors="coerce").fillna(0).astype("int64")
    merged["chrom"] = merged["chrom"].astype(str)
    merged["strand"] = merged["strand"].astype(str)
    return merged


def cluster_nearby(df: pd.DataFrame, tol: int) -> pd.DataFrame:
    # collapse positions within ±tol per (chrom,strand); keep representative ~ median
    rows: List[pd.DataFrame] = []
    for (chrom, strand), d in df.sort_values(["chrom","strand","pos"]).groupby(["chrom","strand"], dropna=False):
        if d.empty:
            continue
        d = d.reset_index(drop=True)
        cluster_id, prev = 0, None
        ids: List[int] = []
        for p in d["pos"].astype(int):
            if prev is None or (p - prev) > tol:
                cluster_id += 1
            ids.append(cluster_id)
            prev = p
        d["cluster_id"] = ids
        # representative closest to median position
        rep_idx = (d.groupby("cluster_id")["pos"]
                     .apply(lambda s: (s - s.median()).abs().idxmin()))
        reps = d.loc[rep_idx.values].copy()
        # aggregate sources within cluster
        src = d.groupby("cluster_id")["source"].apply(agg_sources).reset_index()
        reps = reps.merge(src, on="cluster_id", suffixes=("","_agg"))
        reps["source"] = reps["source_agg"]
        reps.drop(columns=["cluster_id","source_agg"], inplace=True)
        rows.append(reps)
    out = pd.concat(rows, ignore_index=True) if rows else df.copy()
    out["pos"] = out["pos"].astype("int64")
    return out


def dedup_by_sequence(df: pd.DataFrame) -> pd.DataFrame:
    if "sequence" not in df.columns:
        raise SystemExit("[ERROR] 'sequence' column required for sequence de-dup.")
    g = df.groupby("sequence", as_index=False)
    out = g.agg({
        "chrom":"first","pos":"first","strand":"first",
        "gene_id":"first","gene_name":"first","gene_type":"first",
        "source": agg_sources
    })
    out["n_coords_collapsed"] = g.size()["size"].values
    out["pos"] = out["pos"].astype("int64")
    return out


def make_id(chrom: str, pos: int, strand: str, seq: str) -> str:
    base = f"{chrom}|{pos}|{strand}|{seq}".encode("utf-8")
    return hashlib.blake2b(base, digest_size=8).hexdigest()  # 16-hex stable id


def main():
    pdb_path, pas_path, out_dir = find_io()
    print("="*70)
    print("MERGE FOR DNABERT — de-dup + optional clustering")
    print("="*70)
    if not (pdb_path and pas_path):
        print("[ERROR] Could not find inputs.")
        print(f"  - {PDB_NAME}: {pdb_path if pdb_path else 'NOT FOUND'}")
        print(f"  - {PAS_NAME}: {pas_path if pas_path else 'NOT FOUND'}")
        sys.exit(1)
    print(f"PolyA_DB:  {pdb_path}")
    print(f"PolyASite: {pas_path}")
    print(f"Output ->  {out_dir}\n")

    # Load
    a = pd.read_csv(pdb_path)
    b = pd.read_csv(pas_path)
    for label, df in (("PolyA_DB", a), ("PolyASite", b)):
        require_cols(df, ["chrom","pos","strand"], f"{label} CSV")

    # Ensure source labels
    a = ensure_source(a, "PolyA_DB")
    b = ensure_source(b, "PolyASite")

    # Internal de-dup
    print("[Step 1] Internal coordinate de-dup within each source:")
    a = internal_coord_dedup(a)
    b = internal_coord_dedup(b)
    print()

    # Merge by coordinate + provenance aggregation
    print("[Step 2] Merge across sources by (chrom,pos,strand) and aggregate provenance:")
    merged = merge_by_coord(a, b)
    merged_path = out_dir / "merged_coord_dedup.csv"
    merged.to_csv(merged_path, index=False)
    print(f"    wrote {merged_path}  (rows: {len(merged):,})\n")

    # Optional clustering
    base = merged
    if DO_CLUSTER:
        print(f"[Step 3] Cluster nearby PAS within ±{TOL} nt per (chrom,strand):")
        clustered = cluster_nearby(merged, tol=TOL)
        clustered_path = out_dir / f"merged_clustered_tol{TOL}.csv"
        clustered.to_csv(clustered_path, index=False)
        print(f"    wrote {clustered_path}  (rows: {len(clustered):,})\n")
        base = clustered

    # Unique sequence table for leakage-safe splits
    if WRITE_UNIQUE_SEQS:
        print("[Step 4] Build one-row-per-sequence table:")
        uniq = dedup_by_sequence(base)
        uniq_path = out_dir / "merged_unique_sequences.csv"
        uniq.to_csv(uniq_path, index=False)
        print(f"    wrote {uniq_path}  (rows: {len(uniq):,})\n")

    # Final positives in legacy schema
    print("[Step 5] Write legacy apasites_positives.csv:")
    for col in ("gene_name","source","sequence"):
        if col not in base.columns:
            base[col] = ""
    pos = base[["chrom","pos","strand","gene_name","source","sequence"]].copy()
    pos["id"] = [make_id(c, int(p), s, seq) for c,p,s,seq in zip(pos["chrom"], pos["pos"], pos["strand"], pos["sequence"])]
    pos = pos[["id","chrom","pos","strand","gene_name","source","sequence"]]
    out_path = out_dir / OUT_POSITIVES
    pos.to_csv(out_path, index=False)
    print(f"    wrote {out_path}  (rows: {len(pos):,})")
    print("\nDone.")


if __name__ == "__main__":
    main()
