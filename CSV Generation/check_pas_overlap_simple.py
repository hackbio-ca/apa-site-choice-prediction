#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-argument PAS checker (updated):

- If it finds `output/apasites_positives.csv`, it runs **single-file** checks:
    * internal coord duplicates (chrom,pos,strand)
    * sequence duplicates
    * id duplicates (if 'id' present)
    * near-neighbors within ±12 nt (clusteriness inside the file)
    * sequence quality (length stats, non-ACGT rate)

- Otherwise, it falls back to the original **two-file** overlap check between:
    * output/polyadb_annotated_hg38.csv
    * output/polyasite_annotated_hg38.csv

Run:
    python check_pas_overlap_simple.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Optional
import sys
import pandas as pd

# ======= Config (edit here if you want) =======
TOL = 12           # near-duplicate tolerance in nucleotides
MAX_EXAMPLES = 25  # how many example pairs to print
PDB_NAME = "polyadb_annotated_hg38.csv"
PAS_NAME = "polyasite_annotated_hg38.csv"
POS_NAME = "apasites_positives.csv"
# ==============================================


def find_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Try to locate the positives CSV and/or the two source CSVs."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "output",
        here.parent / "output",
        Path.cwd() / "output",
        here,
        here.parent,
        Path.cwd(),
    ]
    pdb_path = pas_path = pos_path = None
    for base in candidates:
        a = base / PDB_NAME
        b = base / PAS_NAME
        p = base / POS_NAME
        if pos_path is None and p.exists():
            pos_path = p
        if pdb_path is None and a.exists():
            pdb_path = a
        if pas_path is None and b.exists():
            pas_path = b
    return pos_path, pdb_path, pas_path


def require_cols(df: pd.DataFrame, needed: Iterable[str], label: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] {label} is missing required columns: {missing}.\n"
                         f"Columns found: {list(df.columns)}")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"[ERROR] failed reading {label} at {path}: {e}")
    # Standardize dtypes
    if "pos" in df.columns:
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str)
    if "strand" in df.columns:
        df["strand"] = df["strand"].astype(str)
    return df


def exact_coord_overlap(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    key = ["chrom", "pos", "strand"]
    return a[key].merge(b[key], on=key, how="inner")


def internal_exact_dups(df: pd.DataFrame) -> int:
    key = ["chrom", "pos", "strand"]
    return int(df.duplicated(subset=key).sum())


def near_overlap_pairs(a: pd.DataFrame, b: pd.DataFrame, tol: int, max_examples: int = 50):
    """Count near-duplicate overlaps within ±tol nt between sources using a sweep."""
    def to_groups(df):
        g: Dict[Tuple[str, str], List[int]] = {}
        for (chrom, strand), d in df.groupby(["chrom", "strand"], dropna=False):
            if pd.isna(chrom) or pd.isna(strand):
                continue
            pos_list = sorted(int(p) for p in d["pos"].dropna().astype(int).tolist())
            if pos_list:
                g[(chrom, strand)] = pos_list
        return g

    A = to_groups(a)
    B = to_groups(b)

    pair_count = 0
    a_matched_total: Set[Tuple[str, str, int]] = set()
    b_matched_total: Set[Tuple[str, str, int]] = set()
    examples: List[Tuple[str, str, int, int, int]] = []

    for key in sorted(set(A.keys()).intersection(B.keys())):
        chrom, strand = key
        aa = A[key]
        bb = B[key]
        j_start = 0
        for ap in aa:
            j = j_start
            while j < len(bb) and bb[j] < ap - tol:
                j += 1
            j_start = j
            k = j
            while k < len(bb) and bb[k] <= ap + tol:
                bp = bb[k]
                pair_count += 1
                a_matched_total.add((chrom, strand, ap))
                b_matched_total.add((chrom, strand, bp))
                if len(examples) < max_examples:
                    examples.append((chrom, strand, ap, bp, ap - bp))
                k += 1

    return pair_count, len(a_matched_total), len(b_matched_total), examples


def near_neighbors_within(df: pd.DataFrame, tol: int = TOL) -> Tuple[int, int]:
    """Return (#rows with >=1 neighbor within ±tol, total #pairs) within same file."""
    cnt_rows = 0
    total_pairs = 0
    for (chrom, strand), d in df.groupby(["chrom","strand"], dropna=False):
        if d.empty:
            continue
        pos = sorted(int(p) for p in d["pos"].dropna().astype(int).tolist())
        if not pos:
            continue
        j = 0
        for i, p in enumerate(pos):
            while j < len(pos) and pos[j] < p - tol:
                j += 1
            k = j
            had = False
            while k < len(pos) and pos[k] <= p + tol:
                if pos[k] != p:
                    total_pairs += 1
                    had = True
                k += 1
            if had:
                cnt_rows += 1
    return cnt_rows, total_pairs


def report_positives(path: Path) -> None:
    df = load_csv(path, "APAS positives")
    print("=" * 70)
    print("APASITES POSITIVES CHECK")
    print("=" * 70)
    print(f"File: {path}")
    print(f"Rows: {len(df):,}\n")

    # Core columns?
    has_core = all(c in df.columns for c in ["chrom","pos","strand"])
    has_seq  = "sequence" in df.columns
    has_id   = "id" in df.columns

    if has_core:
        coord_dups = int(df.duplicated(subset=["chrom","pos","strand"]).sum())
        print(f"[Coord duplicates] (chrom,pos,strand): {coord_dups:,}")
        nn_rows, nn_pairs = near_neighbors_within(df, tol=TOL)
        print(f"[Near-neighbors] rows with ≥1 within ±{TOL} nt: {nn_rows:,}")
        print(f"[Near-neighbors] total pairs (±{TOL} nt): {nn_pairs:,}")
    else:
        print("Missing chrom/pos/strand; skipping coord checks.")

    if has_seq:
        seq_dups = int(df.duplicated(subset=["sequence"]).sum())
        uniq_seq = int(df["sequence"].nunique())
        print(f"[Sequences] duplicates: {seq_dups:,}")
        print(f"[Sequences] unique:     {uniq_seq:,}")
        # Length stats
        lens = df["sequence"].astype(str).str.len()
        print(f"[Sequences] length: min={lens.min()}, max={lens.max()}, median={int(lens.median())}")
        non_acgt = df["sequence"].astype(str).str.contains(r"[^ACGT]").sum()
        print(f"[Sequences] rows with non-ACGT chars: {int(non_acgt):,}")
    else:
        print("No 'sequence' column; skipping sequence checks.")

    if has_id:
        id_dups = int(df.duplicated(subset=["id"]).sum())
        print(f"[IDs] duplicate ids: {id_dups:,}")
        print(f"[IDs] unique ids:    {df['id'].nunique():,}")
    else:
        print("No 'id' column; skipping id checks.")

    # Quick samples if duplicates exist
    if has_core and coord_dups > 0:
        print("\nExamples of coord duplicates (first 5 groups):")
        dup_keys = df[df.duplicated(subset=['chrom','pos','strand'], keep=False)]                     .sort_values(['chrom','pos','strand'])                     .groupby(['chrom','pos','strand']).head(2)
        print(dup_keys.head(10)[['chrom','pos','strand']])

    if has_seq and seq_dups > 0:
        print("\nExamples of sequence duplicates (first 5):")
        dup_seq = df[df.duplicated(subset=['sequence'], keep=False)].head(5)
        cols = ['chrom','pos','strand','sequence']
        print(dup_seq[cols if all(c in dup_seq.columns for c in cols) else dup_seq.columns].head(5))

    print("\nDone.")


def main():
    pos_path, pdb_path, pas_path = find_paths()
    if pos_path:
        # Single-file mode
        report_positives(pos_path)
        return

    # Fall back to two-file overlap mode (original behavior)
    print("=" * 70)
    print("PAS OVERLAP CHECK (fallback two-file mode)")
    print("=" * 70)
    if not (pdb_path and pas_path):
        print("[ERROR] Could not find both source CSVs.")
        print(f"  - {PDB_NAME}: {pdb_path if pdb_path else 'NOT FOUND'}")
        print(f"  - {PAS_NAME}: {pas_path if pas_path else 'NOT FOUND'}")
        print("Put them under an 'output/' folder next to this script (or the repo root).")
        sys.exit(1)

    a = load_csv(pdb_path, "PolyA_DB CSV")
    b = load_csv(pas_path, "PolyASite CSV")

    # Require core columns
    require_cols(a, ["chrom", "pos", "strand"], "PolyA_DB CSV")
    require_cols(b, ["chrom", "pos", "strand"], "PolyASite CSV")

    n_a, n_b = len(a), len(b)
    print(f"PolyA_DB file:    {pdb_path}")
    print(f"PolyASite file:   {pas_path}")
    print(f"Tolerance (nt):   ±{TOL}\n")
    print(f"PolyA_DB rows:    {n_a:,}")
    print(f"PolyASite rows:   {n_b:,}\n")

    inner = exact_coord_overlap(a, b)
    n_exact = len(inner)
    print(f"[Exact coord overlap] same (chrom,pos,strand): {n_exact:,}")
    if n_a:
        print(f"  - Coverage of PolyA_DB:  {n_exact / n_a:.2%}")
    if n_b:
        print(f"  - Coverage of PolyASite: {n_exact / n_b:.2%}")
    print()

    dup_a = internal_exact_dups(a)
    dup_b = internal_exact_dups(b)
    print(f"[Internal exact duplicates] (chrom,pos,strand)")
    print(f"  - PolyA_DB internal dups:  {dup_a:,}")
    print(f"  - PolyASite internal dups: {dup_b:,}")
    print()

    pairs, a_hit, b_hit, examples = near_overlap_pairs(a, b, tol=TOL, max_examples=MAX_EXAMPLES)
    print(f"[Near-duplicate overlaps] within ±{TOL} nt across sources")
    print(f"  - Pair count (all matches):              {pairs:,}")
    print(f"  - PolyA_DB rows with ≥1 near match:     {a_hit:,} ({(a_hit/n_a if n_a else 0):.2%})")
    print(f"  - PolyASite rows with ≥1 near match:    {b_hit:,} ({(b_hit/n_b if n_b else 0):.2%})")
    if examples:
        print(f"  - Examples (up to {MAX_EXAMPLES}):")
        print(f"      chrom   strand   pos_A   pos_B   delta(A-B)")
        for chrom, strand, ap, bp, d in examples[:MAX_EXAMPLES]:
            print(f"      {chrom:<7} {strand:^6} {ap:>8} {bp:>8} {d:>10}")
    print()

    seq_stats = None
    if "sequence" in a.columns and "sequence" in b.columns:
        both = pd.concat([a[["sequence"]].assign(_src="A"),
                          b[["sequence"]].assign(_src="B")], ignore_index=True)
        seq_dups = int(both.duplicated(subset=["sequence"]).sum())
        uniqA = int(a["sequence"].nunique())
        uniqB = int(b["sequence"].nunique())
        common = pd.merge(a[["sequence"]].drop_duplicates(), b[["sequence"]].drop_duplicates(),
                          on="sequence", how="inner")
        print("[Sequence duplicates] (across sources)")
        print(f"  - total_rows_concat: {len(both):,}")
        print(f"  - dup_rows_concat:   {seq_dups:,}")
        print(f"  - unique_sequences_in_A: {uniqA:,}")
        print(f"  - unique_sequences_in_B: {uniqB:,}")
        print(f"  - unique_sequences_in_both: {len(common):,}")
    else:
        print("[Sequence duplicates] Skipped (no 'sequence' column found in one or both files).")
    print()

    print("Done.")
    

if __name__ == "__main__":
    main()
