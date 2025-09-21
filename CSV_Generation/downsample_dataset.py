#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downsample the combined dataset for faster training, keeping format identical.

What it does (in order):
  1) Loads ./output/apasites_dataset.csv (same folder as make_dataset output)
  2) Drops exact duplicates by (chrom,pos,strand,label)
  3) Drops sequence duplicates within the SAME label (optional across labels: see RESOLVE_SEQ_CONFLICT)
  4) Collapses near-duplicates within ±TOL nt per (chrom,strand,label) (keeps one representative)
  5) Balances classes and downsamples to ~50% total rows while keeping pos=neg
  6) Writes ./output/apasites_dataset_half.csv with the SAME columns and order

Edit config below to change the tolerance or the target fraction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
import pandas as pd
import numpy as np

# ===================== Config =====================
TOL = 12                 # near-duplicate radius (nt) to collapse within class
TARGET_FRACTION = 0.50   # keep ~50% of rows after de-dup/clustering
SEED = 1337              # for reproducible sampling
RESOLVE_SEQ_CONFLICT = "drop_neg"  # 'drop_neg', 'drop_pos', or 'keep_both'
IN_NAME = "apasites_dataset.csv"
OUT_NAME = "apasites_dataset_half.csv"
# ==================================================


def find_dataset() -> Tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    for base in (here / "output", here.parent / "output", Path.cwd() / "output"):
        p = base / IN_NAME
        if p.exists():
            return p, base / OUT_NAME
    # fallback
    base = here / "output"
    base.mkdir(parents=True, exist_ok=True)
    return base / IN_NAME, base / OUT_NAME


def require_cols(df: pd.DataFrame, needed: Iterable[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] dataset is missing columns: {missing}. Found: {list(df.columns)}")


def drop_exact_dups(df: pd.DataFrame) -> pd.DataFrame:
    key = ["chrom","pos","strand","label"]
    before = len(df)
    out = df.drop_duplicates(subset=key, keep="first").copy()
    print(f"  exact coord de-dup: {before:,} -> {len(out):,}  (removed {before-len(out):,})")
    return out


def drop_sequence_dups(df: pd.DataFrame) -> pd.DataFrame:
    if "sequence" not in df.columns:
        print("  [skip] sequence de-dup: no 'sequence' column")
        return df
    before = len(df)
    out = df.sort_values(["label"]).drop_duplicates(subset=["sequence","label"], keep="first")
    print(f"  sequence de-dup (within label): {before:,} -> {len(out):,}  (removed {before-len(out):,})")

    # Handle sequences that appear in BOTH labels (conflicts)
    if RESOLVE_SEQ_CONFLICT != "keep_both":
        seq_pos = set(out.loc[out["label"] == 1, "sequence"].dropna().astype(str))
        dup_both = out.loc[(out["label"] == 0) & (out["sequence"].astype(str).isin(seq_pos))]
        if len(dup_both) and RESOLVE_SEQ_CONFLICT == "drop_neg":
            out = out.drop(index=dup_both.index)
            print(f"  dropped {len(dup_both):,} negatives that duplicate positive sequences")
        elif len(dup_both) and RESOLVE_SEQ_CONFLICT == "drop_pos":
            # drop positives that collide with negatives
            seq_neg = set(out.loc[out["label"] == 0, "sequence"].dropna().astype(str))
            dup_pos = out.loc[(out["label"] == 1) & (out["sequence"].astype(str).isin(seq_neg))]
            out = out.drop(index=dup_pos.index)
            print(f"  dropped {len(dup_pos):,} positives that duplicate negative sequences")
    return out


def cluster_within_class(df: pd.DataFrame, tol: int) -> pd.DataFrame:
    # collapse near-duplicates within ±tol per (chrom,strand,label); keep 1 representative
    if not all(c in df.columns for c in ("chrom","pos","strand","label")):
        print("  [skip] clustering: missing chrom/pos/strand/label")
        return df
    rows: List[pd.DataFrame] = []
    for (chrom, strand, label), d in df.sort_values(["chrom","strand","label","pos"]).groupby(["chrom","strand","label"], dropna=False):
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
        # representative: closest to median position in cluster
        rep_idx = (d.groupby("cluster_id")["pos"]
                     .apply(lambda s: (s - s.median()).abs().idxmin()))
        reps = d.loc[rep_idx.values].copy()
        reps.drop(columns=["cluster_id"], inplace=True)
        rows.append(reps)
    out = pd.concat(rows, ignore_index=True) if rows else df.copy()
    print(f"  clustering (±{tol} nt): {len(df):,} -> {len(out):,}  (removed {len(df)-len(out):,})")
    return out


def balance_and_downsample(df: pd.DataFrame, target_fraction: float, seed: int) -> pd.DataFrame:
    # Keep even counts; aim for target_fraction of total rows
    rng = np.random.default_rng(seed)
    n_total = len(df)
    n_target = max(2, int(round(n_total * target_fraction)))  # at least 2 rows
    # enforce even total
    if n_target % 2 == 1:
        n_target -= 1
    # class sizes after de-dup/clustering
    idx_pos = df.index[df["label"] == 1]
    idx_neg = df.index[df["label"] == 0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)
    # target per class
    per_class = min(n_pos, n_neg, n_target // 2)
    if per_class <= 0:
        raise SystemExit("[ERROR] Not enough rows to sample evenly after de-dup/clustering.")
    take_pos = rng.choice(idx_pos.to_numpy(), size=per_class, replace=False)
    take_neg = rng.choice(idx_neg.to_numpy(), size=per_class, replace=False)
    out = df.loc[np.concatenate([take_pos, take_neg])].sort_index().reset_index(drop=True)
    print(f"  balanced downsample: target ~{n_target:,} → kept {len(out):,} (pos={per_class:,}, neg={per_class:,})")
    return out


def main():
    in_path, out_path = find_dataset()
    print("="*70)
    print("DOWNSAMPLE DATASET (half size, de-dup, even classes)")
    print("="*70)
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    if not in_path.exists():
        print("[ERROR] Could not find the combined dataset. Run make_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(in_path)
    require_cols(df, ["chrom","pos","strand","label"])

    # Preserve the original column order
    col_order = list(df.columns)

    print(f"Rows in: {len(df):,}")
    df = drop_exact_dups(df)
    df = drop_sequence_dups(df)
    df = cluster_within_class(df, tol=TOL)

    # Recompute and ensure label column is int 0/1
    df["label"] = df["label"].astype(int)

    # Balance and downsample
    df_small = balance_and_downsample(df, target_fraction=TARGET_FRACTION, seed=SEED)

    # Restore column order and write
    df_small = df_small[[c for c in col_order if c in df_small.columns] + [c for c in df_small.columns if c not in col_order]]
    df_small.to_csv(out_path, index=False)
    print(f"✔ Wrote {out_path}  (rows: {len(df_small):,})")

if __name__ == "__main__":
    main()
