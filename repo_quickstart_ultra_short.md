# 🚀 APA Site Choice Prediction — Ultra‑Short Quickstart

From a fresh clone, this sets up folders, downloads reference/annotation files, and runs the CSV pipeline with default scripts in `CSV_Generation/`.

```bash
# 1) Get the repo
git clone https://github.com/hackbio-ca/apa-site-choice-prediction.git
cd apa-site-choice-prediction

# 2) (Optional) create env
# conda create -n apa python=3.12 -y && conda activate apa
# pip install -r requirements.txt  # or: pip install pandas numpy tqdm pyfaidx genome_kit

# 3) Bootstrap file layout + downloads
bash scripts/bootstrap_data.sh

# 4) Build datasets (uses defaults that write to CSV_Generation/output/)
cd CSV_Generation
python make_dataset.py           # writes apasites_dataset.csv
python downsample_dataset.py     # writes apasites_dataset_half.csv
python make_test_samples.py      # writes Model_Test_Samples/*
```

> If you already have the data files, just drop them into the folders created by the script and re‑run step 4.

---

## scripts/bootstrap_data.sh (add this file)
Create `scripts/bootstrap_data.sh`, make it executable (`chmod +x scripts/bootstrap_data.sh`). It creates the folders and (optionally) downloads hg38 + GENCODE. You can provide PolyASite/PolyA_DB URLs via env vars.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Root of the repo
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CSV_DIR="$ROOT_DIR/CSV_Generation"

# Directory layout
mkdir -p "$CSV_DIR/data/polyasite" \
         "$CSV_DIR/data/polyadb" \
         "$CSV_DIR/data/gencode" \
         "$CSV_DIR/data/reference" \
         "$CSV_DIR/output" \
         "$CSV_DIR/Model_Test_Samples"

# Touch .gitkeep so the folders exist in git
for d in polyasite polyadb gencode reference output Model_Test_Samples; do
  touch "$CSV_DIR/$d/.gitkeep" 2>/dev/null || true
  # also under data/
  [[ $d =~ ^(polyasite|polyadb|gencode|reference)$ ]] && touch "$CSV_DIR/data/$d/.gitkeep" 2>/dev/null || true
done

# Helper: download if missing
_dl() {
  local url="$1" dest="$2"; shift 2 || true
  if [[ -f "$dest" ]]; then
    echo "[skip] $dest (exists)"
  else
    echo "[get ] $url -> $dest"
    curl -L --fail --retry 3 --retry-delay 2 -o "$dest" "$url"
  fi
}

# --- Defaults (you can override with env vars before running) ---
: "${GENCODE_RELEASE:=46}"
: "${GENCODE_GTF_URL:=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_RELEASE}/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz}"
: "${HG38_FASTA_URL:=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz}"

# Optional: space-separated lists
: "${POLYASITE_URLS:=}"   # e.g., "https://.../polyasite_cluster_set1.bed.gz https://.../set2.bed.gz"
: "${POLYADB_URLS:=}"     # e.g., "https://.../PolyA_DB_human_PAS.txt.gz"

# Download reference + annotation
_dl "$HG38_FASTA_URL" "$CSV_DIR/data/reference/hg38.fa.gz"
_dl "$GENCODE_GTF_URL" "$CSV_DIR/data/gencode/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz"

# (Optional) PolyASite & PolyA_DB — only if URLs provided
if [[ -n "$POLYASITE_URLS" ]]; then
  for u in $POLYASITE_URLS; do
    base=$(basename "$u")
    _dl "$u" "$CSV_DIR/data/polyasite/$base"
  done
else
  echo "[info] Set POLYASITE_URLS to auto-download PolyASite files, or drop them into CSV_Generation/data/polyasite/"
fi

if [[ -n "$POLYADB_URLS" ]]; then
  for u in $POLYADB_URLS; do
    base=$(basename "$u")
    _dl "$u" "$CSV_DIR/data/polyadb/$base"
  done
else
  echo "[info] Set POLYADB_URLS to auto-download PolyA_DB files, or drop them into CSV_Generation/data/polyadb/"
fi

# Optional: decompress copies for convenience
if command -v gunzip >/dev/null 2>&1; then
  [[ -f "$CSV_DIR/data/reference/hg38.fa.gz" ]] && gunzip -kf "$CSV_DIR/data/reference/hg38.fa.gz"
  [[ -f "$CSV_DIR/data/gencode/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz" ]] && gunzip -kf "$CSV_DIR/data/gencode/gencode.v${GENCODE_RELEASE}.annotation.gtf.gz"
fi

echo "\n[done] Layout ready under $CSV_DIR"
echo "- reference : CSV_Generation/data/reference"
echo "- gencode   : CSV_Generation/data/gencode"
echo "- polyasite : CSV_Generation/data/polyasite"
echo "- polyadb   : CSV_Generation/data/polyadb"
echo "- output    : CSV_Generation/output"
```

That’s it. Set `POLYASITE_URLS` / `POLYADB_URLS` if you want the script to fetch inputs for you, otherwise just drop your files into the created folders and run step 4 above.

