#!/usr/bin/env python3
"""
make_test_samples.py

Generate small CSV test files from the portion of the dataset that is NOT in the
downsampled half. Tailored for this repo layout:

CSV_Generation/
  ├─ output/
  │   ├─ apasites_dataset.csv
  │   └─ apasites_dataset_half.csv
  └─ Model_Test_Samples/   (created if missing)

Usage:
  cd CSV_Generation
  python make_test_samples.py
"""

import os
import hashlib
import pandas as pd

# --- Paths fixed to your layout ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "Model_Test_Samples")
FULL_PATH = os.path.join(THIS_DIR, "output", "apasites_dataset.csv")
HALF_PATH = os.path.join(THIS_DIR, "output", "apasites_dataset_half.csv")

# Sampling config (adjust if you want)
SAMPLE_PER_CLASS = 100
RNG_SEED = 1337


def _stable_row_key(row: pd.Series) -> str:
    """
    Robust, layout-agnostic key to identify a row across CSVs.

    Priority:
      1) uid/site_id/id
      2) chrom,strand,pos
      3) chrom,strand,start,end
      4) chrom,pos
      5) SHA1 over (sequence,label,strand)
    """
    for col in ("uid", "site_id", "id"):
        if col in row and pd.notna(row[col]):
            return f"ID::{row[col]}"

    chrom = row.get("chrom", None)
    strand = row.get("strand", None)
    if pd.notna(strand):
        s = str(strand).strip()
        if s in {"1", "+1", "plus", "Plus", "POS", "pos"}:
            strand = "+"
        elif s in {"-1", "minus", "Minus", "NEG", "neg"}:
            strand = "-"
        else:
            strand = str(strand)

    if "chrom" in row and "pos" in row and pd.notna(chrom) and pd.notna(row["pos"]):
        try:
            return f"CSP::{chrom}::{strand}::{int(row['pos'])}"
        except Exception:
            pass

    for st, en in (("start", "end"), ("Start", "End")):
        if st in row and en in row and pd.notna(row.get(st)) and pd.notna(row.get(en)) and pd.notna(chrom):
            try:
                return f"CSE::{chrom}::{strand}::{int(row[st])}::{int(row[en])}"
            except Exception:
                pass

    if "chrom" in row and "pos" in row and pd.notna(chrom) and pd.notna(row["pos"]):
        try:
            return f"CP::{chrom}::{int(row['pos'])}"
        except Exception:
            pass

    seq = str(row.get("sequence", "")).upper()
    label = str(row.get("label", "NA"))
    payload = f"{seq}::{label}::{strand}"
    return "HASH::" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def anti_join(full_df: pd.DataFrame, half_df: pd.DataFrame) -> pd.DataFrame:
    """Rows in full_df that are NOT in half_df (by stable key)."""
    full = full_df.copy()
    half = half_df.copy()
    full["_rk"] = full.apply(_stable_row_key, axis=1)
    half["_rk"] = half.apply(_stable_row_key, axis=1)
    diff = full[~full["_rk"].isin(set(half["_rk"]))].drop(columns=["_rk"])
    return diff


def main():
    if not os.path.exists(FULL_PATH):
        raise FileNotFoundError(f"Missing full dataset: {FULL_PATH}")
    if not os.path.exists(HALF_PATH):
        raise FileNotFoundError(f"Missing half dataset: {HALF_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Full dataset: {FULL_PATH}")
    print(f"[INFO] Half dataset: {HALF_PATH}")

    full_df = pd.read_csv(FULL_PATH)
    half_df = pd.read_csv(HALF_PATH)

    if "label" not in full_df.columns or "label" not in half_df.columns:
        raise ValueError("Both CSVs must contain a 'label' column (0/1).")

    other_half = anti_join(full_df, half_df)
    if other_half.empty:
        raise ValueError("No remaining rows after removing half; inputs may be identical.")

    # Save the entire remainder for traceability
    remainder_path = os.path.join(OUT_DIR, "other_half_full.csv")
    other_half.to_csv(remainder_path, index=False)
    print(f"[OK] Wrote {remainder_path} (rows={len(other_half)})")

    # Build small labeled samples
    df = other_half.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)

    # Map label strings to 0/1 if needed
    if df["label"].dtype == object:
        mapped = df["label"].astype(str).str.lower().map({
            "1": 1, "pos": 1, "positive": 1, "true": 1,
            "0": 0, "neg": 0, "negative": 0, "false": 0
        })
        if mapped.notna().any():
            df["label"] = mapped.fillna(df["label"])

    pos = df[df["label"].isin([1, "1", True, "true", "pos", "positive"])]
    neg = df[df["label"].isin([0, "0", False, "false", "neg", "negative"])]

    n_pos = min(SAMPLE_PER_CLASS, len(pos))
    n_neg = min(SAMPLE_PER_CLASS, len(neg))

    pos_sample = pos.head(n_pos).copy()
    neg_sample = neg.head(n_neg).copy()
    combined = pd.concat([pos_sample, neg_sample], ignore_index=True).sample(frac=1, random_state=RNG_SEED)

    pos_out = os.path.join(OUT_DIR, "positives_sample.csv")
    neg_out = os.path.join(OUT_DIR, "negatives_sample.csv")
    comb_out = os.path.join(OUT_DIR, "combined_sample.csv")

    pos_sample.to_csv(pos_out, index=False)
    neg_sample.to_csv(neg_out, index=False)
    combined.to_csv(comb_out, index=False)

    print(f"[OK] Wrote {pos_out} (rows={len(pos_sample)})")
    print(f"[OK] Wrote {neg_out} (rows={len(neg_sample)})")
    print(f"[OK] Wrote {comb_out} (rows={len(combined)})")
    print("[DONE] Test CSVs are ready in Model_Test_Samples/")

if __name__ == "__main__":
    main()
