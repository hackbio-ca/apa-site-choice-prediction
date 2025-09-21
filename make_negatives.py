#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make_negatives.py  — hybrid negatives with sequences, same schema as positives
import os
import sys
import random
import pandas as pd
from collections import defaultdict
from hashlib import blake2b
from genome_kit import Genome, Interval

# ---------------- Paths (script-relative) ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "output")
os.makedirs(OUTD, exist_ok=True)

POSITIVES_CSV = os.path.join(OUTD, "apasites_positives.csv")
NEGATIVES_CSV = os.path.join(OUTD, "apasites_negatives.csv")

# ---------------- Config ----------------
RANDOM_SEED = 42
NEGATIVES_PER_POSITIVE = 2          # total negatives per positive (across all buckets)

# Hybrid mix (should sum ~1.0). Adjust if desired.
MIX_WINDOW = 0.50                   # window-shift ±100..400 nt within same UTR
MIX_RANDOM_UTR = 0.35               # uniform inside 3'UTRs
MIX_MOTIF = 0.15                    # motif-matched (PAS-like hexamer nearby)

EXCLUSION_RADIUS = 50               # forbid negatives within ±N nt of any PAS
WIN_NEAR_MIN = 100                  # window-shift min offset
WIN_NEAR_MAX = 400                  # window-shift max offset
DEFAULT_SEQ_LEN = 101               # used if positives lack 'sequence'
REQUIRED_UTR_MARGIN = 60            # keep sequence window fully inside UTR

# PAS-like hexamers (DNA alphabet); both strands are checked by revcomp
PAS_HEXAMERS = {
    "AATAAA","ATTAAA","AGTAAA","TATAAA","CATAAA","GATAAA",
    "AATACA","AATAGA","AATATA","AATGAA","ACTAAA","AAGAAA"
}

random.seed(RANDOM_SEED)

# ---------------- Helpers ----------------
def load_gencode_hg38():
    tried = []
    for name in ["gencode.v44", "gencode.v43", "gencode.v42", "gencode.v41", "gencode"]:
        try:
            return Genome(name)
        except Exception as e:
            tried.append((name, str(e)))
    sys.stderr.write("Failed to load GENCODE hg38 annotations. Tried: {}\n".format(
        ", ".join(n for n,_ in tried)))
    sys.exit(1)

def load_ref_hg38():
    try:
        return Genome("hg38")
    except Exception as e:
        sys.stderr.write(f"Failed to load hg38 reference: {e}\n")
        sys.exit(1)

def read_positives(path):
    if not os.path.exists(path):
        sys.stderr.write(f"Missing positives file: {path}\n")
        sys.exit(1)
    df = pd.read_csv(path)
    need = {"chrom","pos","strand"}
    miss = need - set(df.columns)
    if miss:
        sys.stderr.write(f"Positives file missing columns: {miss}\n")
        sys.exit(1)
    df["chrom"] = df["chrom"].astype(str)
    df["strand"] = df["strand"].astype(str)
    df["pos"] = df["pos"].astype(int)
    return df

def infer_seq_len_from_positives(pos_df):
    if "sequence" in pos_df.columns:
        for s in pos_df["sequence"].astype(str):
            s = s.strip()
            if s:
                return len(s)
    return DEFAULT_SEQ_LEN

def build_pas_index(df):
    by_chrom = defaultdict(set)
    for c, p in zip(df["chrom"].values, df["pos"].values):
        by_chrom[c].add(int(p))
    return by_chrom

def pos_is_clear_of_pas(chrom, pos, pas_index, radius=EXCLUSION_RADIUS):
    for d in range(-radius, radius+1):
        if (pos + d) in pas_index.get(chrom, ()):
            return False
    return True

def iter_utr3s(tx):
    """Yield 3'UTR Interval(s) across genome_kit variants."""
    if hasattr(tx, "utr3") and getattr(tx, "utr3") is not None:
        yield tx.utr3
    if hasattr(tx, "utr3s") and getattr(tx, "utr3s"):
        for iv in tx.utr3s:
            if iv is not None:
                yield iv

def make_id(chrom, pos, strand, gene_name):
    s = f"{chrom}:{int(pos)}:{strand}:{gene_name or ''}"
    return blake2b(s.encode(), digest_size=8).hexdigest()

def revcomp(s: str) -> str:
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return s.translate(table)[::-1]

def has_pas_hexamer(seq: str) -> bool:
    s = seq.upper()
    rc = revcomp(s)
    return any(h in s or h in rc for h in PAS_HEXAMERS)

def fetch_seq(ref: Genome, chrom: str, strand: str, center_pos: int, seq_len: int) -> str:
    """Extract DNA of length seq_len centered at center_pos on given strand."""
    left = seq_len // 2
    right = seq_len - left
    start = center_pos - left
    end = center_pos + right
    iv = Interval(chrom, strand, start, end, ref)
    return ref.dna(iv)

# ---------------- Samplers ----------------
def sample_window_shift(utr, chrom, strand, anchor_pos, pas_index, want, seq_len, ref):
    """Sample positions ±(100..400) nt from anchor_pos, clamped to UTR, not near PAS."""
    out = []
    for _ in range(want * 6):  # generous attempts
        if len(out) >= want: break
        offset = random.randint(WIN_NEAR_MIN, WIN_NEAR_MAX)
        if random.random() < 0.5:
            offset = -offset
        cand = anchor_pos + offset
        # clamp to keep full sequence inside UTR
        margin = max(REQUIRED_UTR_MARGIN, seq_len // 2 + 1)
        lo = utr.start + margin
        hi = utr.end   - margin
        if not (lo <= cand <= hi):
            continue
        if not pos_is_clear_of_pas(chrom, cand, pas_index):
            continue
        try:
            seq = fetch_seq(ref, chrom, strand, cand, seq_len)
        except Exception:
            continue
        out.append((cand, seq))
    return out

def sample_random_utr(utr, chrom, strand, pas_index, want, seq_len, ref):
    out = []
    margin = max(REQUIRED_UTR_MARGIN, seq_len // 2 + 1)
    lo = utr.start + margin
    hi = utr.end   - margin
    if hi <= lo:
        return out
    for _ in range(want * 4):
        if len(out) >= want: break
        cand = random.randint(lo, hi)
        if not pos_is_clear_of_pas(chrom, cand, pas_index):
            continue
        try:
            seq = fetch_seq(ref, chrom, strand, cand, seq_len)
        except Exception:
            continue
        out.append((cand, seq))
    return out

def sample_motif_matched(utr, chrom, strand, pas_index, want, seq_len, ref, scan_half=50):
    out = []
    margin = max(REQUIRED_UTR_MARGIN, seq_len // 2 + 1)
    lo = utr.start + margin
    hi = utr.end   - margin
    if hi <= lo:
        return out
    for _ in range(want * 8):  # harder to find, so more tries
        if len(out) >= want: break
        cand = random.randint(lo, hi)
        if not pos_is_clear_of_pas(chrom, cand, pas_index):
            continue
        # scan small window for PAS-like hexamer
        w_lo = max(utr.start, cand - scan_half)
        w_hi = min(utr.end,   cand + scan_half)
        try:
            seq_scan = fetch_seq(ref, chrom, strand, (w_lo + w_hi)//2, (w_hi - w_lo))
        except Exception:
            continue
        if not has_pas_hexamer(seq_scan):
            continue
        try:
            seq = fetch_seq(ref, chrom, strand, cand, seq_len)
        except Exception:
            continue
        out.append((cand, seq))
    return out

# ---------------- Main ----------------
def main():
    ref = load_ref_hg38()
    gencode = load_gencode_hg38()

    pos_df = read_positives(POSITIVES_CSV)
    pas_index = build_pas_index(pos_df)
    seq_len = infer_seq_len_from_positives(pos_df)

    # Build UTR index & map positives to UTRs (same-chrom/strand containment)
    # utr_index: (chrom,strand) -> list of (utr_interval, gene_name)
    utr_index = defaultdict(list)
    for gene in gencode.genes:
        gene_name = getattr(gene, "name", getattr(gene, "id", ""))
        for tx in gene.transcripts:
            for utr in iter_utr3s(tx):
                if utr is not None and (utr.end - utr.start) > (seq_len + 2*REQUIRED_UTR_MARGIN):
                    utr_index[(utr.chrom, utr.strand)].append((utr, gene_name))

    # Helper to find a UTR that contains a given positive
    def find_containing_utr(chrom, strand, pos):
        for utr, gene_name in utr_index.get((chrom, strand), []):
            if utr.start <= pos <= utr.end:
                return utr, gene_name
        return None, ""

    total_negs = int(len(pos_df) * NEGATIVES_PER_POSITIVE)
    n_win_target  = int(total_negs * MIX_WINDOW)
    n_rand_target = int(total_negs * MIX_RANDOM_UTR)
    n_motif_target= max(0, total_negs - n_win_target - n_rand_target)

    negatives = []
    seen_sites = set()  # avoid dup (chrom,strand,pos)

    # -------- 1) Window-shift (anchor each positive) --------
    got_win = 0
    for row in pos_df.itertuples(index=False):
        if got_win >= n_win_target:
            break
        chrom, strand, ppos = row.chrom, row.strand, int(row.pos)
        utr, gene_name = find_containing_utr(chrom, strand, ppos)
        if utr is None:
            continue
        for cand_pos, seq in sample_window_shift(utr, chrom, strand, ppos, pas_index, want=1, seq_len=seq_len, ref=ref):
            key = (chrom, strand, cand_pos)
            if key in seen_sites: 
                continue
            seen_sites.add(key)
            negatives.append({
                "id": make_id(chrom, cand_pos, strand, gene_name),
                "chrom": chrom,
                "pos": int(cand_pos),
                "strand": strand,
                "gene_name": gene_name,
                "source": "negative",
                "sequence": seq,
                "label": 0,
            })
            got_win += 1
            if got_win >= n_win_target:
                break

    # -------- 2) Random UTR --------
    got_rand = 0
    if n_rand_target > 0:
        # Flatten UTR list for easier sampling
        flat_utrs = []
        for key, lst in utr_index.items():
            for utr, gene_name in lst:
                flat_utrs.append((utr, gene_name))
        tries = 0
        while got_rand < n_rand_target and tries < n_rand_target * 20 and flat_utrs:
            tries += 1
            utr, gene_name = random.choice(flat_utrs)
            chrom, strand = utr.chrom, utr.strand
            for cand_pos, seq in sample_random_utr(utr, chrom, strand, pas_index, want=1, seq_len=seq_len, ref=ref):
                key = (chrom, strand, cand_pos)
                if key in seen_sites:
                    continue
                seen_sites.add(key)
                negatives.append({
                    "id": make_id(chrom, cand_pos, strand, gene_name),
                    "chrom": chrom,
                    "pos": int(cand_pos),
                    "strand": strand,
                    "gene_name": gene_name,
                    "source": "negative",
                    "sequence": seq,
                    "label": 0,
                })
                got_rand += 1
                if got_rand >= n_rand_target:
                    break

    # -------- 3) Motif-matched --------
    got_motif = 0
    if n_motif_target > 0:
        flat_utrs = []
        for key, lst in utr_index.items():
            for utr, gene_name in lst:
                flat_utrs.append((utr, gene_name))
        tries = 0
        while got_motif < n_motif_target and tries < n_motif_target * 40 and flat_utrs:
            tries += 1
            utr, gene_name = random.choice(flat_utrs)
            chrom, strand = utr.chrom, utr.strand
            for cand_pos, seq in sample_motif_matched(utr, chrom, strand, pas_index, want=1, seq_len=seq_len, ref=ref):
                key = (chrom, strand, cand_pos)
                if key in seen_sites:
                    continue
                seen_sites.add(key)
                negatives.append({
                    "id": make_id(chrom, cand_pos, strand, gene_name),
                    "chrom": chrom,
                    "pos": int(cand_pos),
                    "strand": strand,
                    "gene_name": gene_name,
                    "source": "negative",
                    "sequence": seq,
                    "label": 0,
                })
                got_motif += 1
                if got_motif >= n_motif_target:
                    break

    if not negatives:
        sys.stderr.write("Failed to generate negatives. Check GENCODE/hg38 availability and positives file.\n")
        sys.exit(1)

    neg_df = pd.DataFrame(negatives).drop_duplicates(subset=["chrom","strand","pos"]).reset_index(drop=True)

    # Ensure exact column order to match positives
    cols = ["id","chrom","pos","strand","gene_name","source","sequence","label"]
    neg_df = neg_df.reindex(columns=cols)

    neg_df.to_csv(NEGATIVES_CSV, index=False)
    print(f"✔ Wrote {len(neg_df):,} negatives -> {NEGATIVES_CSV}  "
          f"(mix: win={got_win}, rand={got_rand}, motif={got_motif}, seq_len={seq_len})")

if __name__ == "__main__":
    main()
