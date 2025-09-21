
import re
import numpy as np
import pandas as pd
from pathlib import Path
from pyfaidx import Fasta
from Bio.Seq import Seq

DATA_DIR  = Path(r"C:\Users\akshi\Desktop\HACKATHON")
BED_FILE  = DATA_DIR / "atlas.clusters.2.0.GRCh38.96.bed"
GTF_FILE  = DATA_DIR / "gencode.v49.polyAs.gtf"
REF_FASTA = DATA_DIR / "GRCh38.primary_assembly.genome.fa"
OUT_CSV   = DATA_DIR / "PolyA_revised5.csv"

print("BED exists? ", BED_FILE.exists())
print("GTF exists? ", GTF_FILE.exists())
print("FASTA exists? ", REF_FASTA.exists())
print("FAI exists? ", REF_FASTA.with_suffix(REF_FASTA.suffix + ".fai").exists())

# ================== SETTINGS ==================
UPSTREAM   = 50
DOWNSTREAM = 50
PAS_FROM   = 10   # scan -10..-40 upstream of cut
PAS_TO     = 40

PAS_VARIANTS = {
    "AAUAAA", "AUUAAA", "AGUAAA", "AAGAAA", "AACAAA",
    "AAUAUA", "AAUUAA", "AAUAAG", "UAUAAA", "GAUAAA",
    "CAUAAA", "ACUAAA", "AAUGAA"
}


def parse_attr(attrs: str, key: str) -> str:
    if not isinstance(attrs, str): return ""
    m = re.search(rf'{re.escape(key)}\s+"([^"]+)"', attrs)
    if m: return m.group(1)
    m2 = re.search(rf'{re.escape(key)}=([^;]+)', attrs)
    return m2.group(1) if m2 else ""

def resolve_chrom_name(fa, chrom: str):
    """Return a contig name that exists in FASTA, trying chr/no-chr and chrM/MT."""
    if not isinstance(chrom, str): return None
    c = chrom.strip()
    if c in fa: return c
    if c.startswith("chr"):
        c2 = c[3:]
        if c2 in fa: return c2
        if c == "chrM" and "MT" in fa: return "MT"
        if c == "chrM" and "M"  in fa: return "M"
    else:
        c2 = "chr" + c
        if c2 in fa: return c2
        if c in ("MT","M") and "chrM" in fa: return "chrM"
    return None  # unresolved

def fetch_seq_window(fa: Fasta, chrom_resolved: str, start_1b: int, end_1b: int, strand: str) -> str:
    """
    Uses pyfaidx.get_seq(name, start, end) which is 1-based INCLUSIVE.
    Returns RNA-like sequence (T->U) in transcript orientation (left=upstream).
    If chrom not resolved or any error, returns "" (so features become zeros).
    """
    try:
        if not chrom_resolved or chrom_resolved not in fa.keys():
            return ""
        if strand == "+":
            cut = int(end_1b)
            left  = max(1, cut - UPSTREAM)
            right = cut + DOWNSTREAM - 1       
            seq = fa.get_seq(chrom_resolved, left, right).seq
            return seq.upper().replace("T","U")
        else:
            cut = int(start_1b)
            left  = max(1, cut - DOWNSTREAM)
            right = cut + UPSTREAM - 1          
            seq = fa.get_seq(chrom_resolved, left, right).seq
            return str(Seq(seq).reverse_complement()).upper().replace("T","U")
    except Exception:
        return ""

def base_comp(seq: str):
    s = (seq or "").upper()
    counts = {b: s.count(b) for b in ["A","U","G","C"]}
    n_count = s.count("N")
    total = (sum(counts.values()) + n_count) or 1
    pct   = {f"{b}_pct": (counts[b]/total)*100.0 for b in counts}
    gc_pct = ((counts["G"] + counts["C"]) / total) * 100.0
    return counts | pct | {"GC_pct": gc_pct, "N_count": n_count}

def scan_pas(seq: str):
    cut = UPSTREAM
    region = (seq or "")[max(0, cut - PAS_TO): max(0, cut - PAS_FROM)]
    found = 0; has_canonical = 0
    for i in range(0, max(0, len(region) - 5)):
        mer = region[i:i+6]
        if mer in PAS_VARIANTS:
            found += 1
            if mer == "AAUAAA":
                has_canonical = 1
    return {
        "presence_AAUAAA": has_canonical,
        "pas_variant_count": found,
        "presence_any_PAS_variant": int(found > 0)
    }


# BED (0-based half-open) -> convert to 1-based inclusive
bed = pd.read_csv(BED_FILE, sep="\t", header=None, usecols=[0,1,2,3,4,5], dtype={0:str})
bed.columns = ["chrom","start0","end0","name","score","strand"]
bed["start"] = pd.to_numeric(bed["start0"], errors="coerce").add(1)  # 0->1
bed["end"]   = pd.to_numeric(bed["end0"],   errors="coerce")
bed["source"] = "PolyASite2"
bed["feature"] = "polyA_site"
bed["attribute"] = bed["name"].fillna("").map(lambda s: f'name "{s}"; dataset "human_polyAsite";')
for k in ["gene_id","gene_name","transcript_id","transcript_biotype"]:
    bed[k] = ""

# GTF (already 1-based inclusive)
gtf = pd.read_csv(GTF_FILE, sep="\t", comment="#", header=None, usecols=list(range(9)), dtype={0:str})
gtf.columns = ["chrom","source","feature","start","end","score","strand","frame","attribute"]
for key in ["gene_id","gene_name","transcript_id","transcript_biotype"]:
    gtf[key] = gtf["attribute"].apply(lambda s, k=key: parse_attr(s, k))

need_cols = [
    "chrom","start","end","strand","score",
    "gene_id","gene_name","transcript_id",
    "feature","transcript_biotype","source","attribute"
]
df = pd.concat([bed[need_cols], gtf[need_cols]], ignore_index=True)

# basic clean
df["start"] = pd.to_numeric(df["start"], errors="coerce")
df["end"]   = pd.to_numeric(df["end"],   errors="coerce")
df = df.dropna(subset=["chrom","start","end","strand"]).copy()
df = df[df["end"] >= df["start"]]

# ================== SEQUENCE FEATURES (keep all rows) ==================
# Open FASTA WITHOUT as_raw, so we can use get_seq (1-based inclusive)
fasta = Fasta(str(REF_FASTA), sequence_always_upper=True)
print("FASTA contigs (first 10):", list(fasta.keys())[:10])

# Resolve contig names (but DO NOT drop unresolved)
df["chrom_resolved"] = df["chrom"].apply(lambda c: resolve_chrom_name(fasta, c))
unres = df["chrom_resolved"].isna().sum()
print("Unresolved chrom rows (kept with empty seq):", unres)

# Fetch windows; unresolved or errors → "" (will produce zeros)
seqs = []
for _, r in df.iterrows():
    seqs.append(
        fetch_seq_window(
            fasta,
            r["chrom_resolved"] if pd.notna(r["chrom_resolved"]) else "",
            int(r["start"]), int(r["end"]), r["strand"]
        )
    )
df["window_seq"] = seqs

# Features
comp = pd.DataFrame([base_comp(s) for s in df["window_seq"]], index=df.index)
pas  = pd.DataFrame([scan_pas(s)  for s in df["window_seq"]], index=df.index)

# ================== FINAL TABLE ==================
out = pd.concat([df, comp, pas], axis=1).rename(columns={
    "chrom_resolved": "Chromosome",
    "start": "Start_1based",
    "end":   "End_1based",
    "strand":"Strand",
    "score": "Score_or_ReadCount",
    "feature": "Feature_type"
})

final_cols = [
    "Chromosome","Start_1based","End_1based","Strand","Score_or_ReadCount",
    "gene_id","gene_name","transcript_id","Feature_type","transcript_biotype",
    "presence_AAUAAA","pas_variant_count","presence_any_PAS_variant",
    "A","U","G","C","A_pct","U_pct","G_pct","C_pct","GC_pct","N_count"
]
for c in final_cols:
    if c not in out.columns:
        out[c] = np.nan
out = out[final_cols]

out.to_csv(OUT_CSV, index=False)
print(f"Wrote: {OUT_CSV}")
print("Rows:", len(out), "Cols:", out.shape[1])
print(out.head(5))
