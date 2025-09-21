# APA Site Choice Prediction — Quickstart

## 1) Project overview
Predict alternative polyadenylation (APA) site usage from sequence. The pipeline builds labeled examples from curated PAS catalogs, converts them into DNABERT‑ready format, and fine‑tunes a transformer to classify sites.

---

## 2) Requirements
- **OS:** macOS or Linux
- **Python:** 3.10–3.12
- **Conda or Mamba** (recommended)  
- **Packages:** `pandas`, `numpy`, `pybedtools` (optional), `genome_kit`, `tqdm`, `pyfaidx`, `transformers`, `datasets`, `accelerate`, `torch` (CPU or CUDA)

> If you already have a working environment for this repo, you can skip to **Step 4**.

### Create an environment
```bash
# from repo root
mamba create -n apa python=3.12 -y  # conda works too
mamba activate apa

# core deps
pip install pandas numpy tqdm pyfaidx

# genome & ML deps
pip install genome_kit transformers datasets accelerate
# choose one of the following torch builds
pip install torch --index-url https://download.pytorch.org/whl/cpu            # CPU-only
# pip install torch --index-url https://download.pytorch.org/whl/cu124        # CUDA 12.4 example
```

---

## 3) Data you’ll need (hg38)
Create the following layout (empty folders are fine to start):
```
apa-site-choice-prediction/
├─ data/
│  ├─ polyasite/           # PolyASite 2.0 BED clusters (hg38)
│  ├─ polyadb/             # PolyA_DB PAS tables (to be lifted to hg38 if needed)
│  ├─ gencode/             # GENCODE GTF (hg38) + annotation index
│  └─ reference/           # hg38 FASTA + .fai
└─ output/
```

**Placeholders:**
- `data/reference/GRCh38.primary_assembly.genome.fa` (+ `*.fai` index)
- `data/gencode/gencode.vXX.annotation.gtf` (hg38)  
- `data/polyasite/*.bed(.gz)` (hg38 clusters)
- `data/polyadb/*.txt` (human PAS tables; lift to hg38 downstream if not already)

> Exact file names don’t matter as long as the scripts can find them in the listed folders.

---

## 4) Build labeled CSVs
From repo root:

### 4.1 PolyASite → CSV
```bash
python polyasite_bed_to_csv.py \
  --bed data/polyasite/*.bed* \
  --gtf data/gencode/gencode.vXX.annotation.gtf \
  --ref data/reference/GRCh38.primary_assembly.genome.fa \
  --out output/polyasite_hg38.csv
```

### 4.2 PolyA_DB → CSV (lift to hg38 if needed)
```bash
python polyadb_to_csv.py \
  --pas data/polyadb/*.txt \
  --gtf data/gencode/gencode.vXX.annotation.gtf \
  --ref data/reference/GRCh38.primary_assembly.genome.fa \
  --out output/polyadb_hg38.csv
```

> These produce harmonized columns (chrom, strand, pos/start/end as applicable, gene, transcript, region, sequence, label).

### 4.3 Merge/clean for modeling
```bash
python merge_for_dnabert.py \
  --polyasite output/polyasite_hg38.csv \
  --polyadb   output/polyadb_hg38.csv \
  --out       output/apasites_positives.csv
```

**Tips**
- To de‑duplicate close sites, use the script’s `--merge-distance` (e.g., 24 nt) and `--keep=score|max|first` flags if available.

---

## 5) Add negatives + finalize dataset
```bash
# Build matched negatives from the same reference and annotations
python make_negatives.py \
  --positives output/apasites_positives.csv \
  --gtf       data/gencode/gencode.vXX.annotation.gtf \
  --ref       data/reference/GRCh38.primary_assembly.genome.fa \
  --out       output/apasites_negatives.csv

# Combine into one DNABERT-ready file
python make_dataset.py \
  --positives output/apasites_positives.csv \
  --negatives output/apasites_negatives.csv \
  --out       output/apasites_with_negatives.csv
```

**Optional accelerators**
```bash
# Halve the dataset for quick experiments while preserving class balance
python tools/subsample_dataset.py \
  --in  output/apasites_with_negatives.csv \
  --out output/apasites_small.csv \
  --ratio 0.5 --balance
```

---

## 6) Train DNABERT‑2 (baseline)
```bash
python train.py \
  --model zhihan1996/DNABERT-2-117M \
  --train output/apasites_with_negatives.csv \
  --val   output/apasites_with_negatives.csv \
  --val-split 0.1 \
  --seq-col sequence --label-col label \
  --epochs 3 --batch-size 16 --lr 3e-5 --seed 1337 \
  --outdir output/models/dnabert2_baseline
```

**Notes**
- The trainer handles tokenization and k‑merization internally if the script exposes `--model` and `--seq-col`.
- For GPU, ensure you installed a CUDA build of PyTorch and run `accelerate config` once.

---

## 7) Evaluate & predict
```bash
python eval.py \
  --model output/models/dnabert2_baseline \
  --test  output/apasites_with_negatives.csv \
  --seq-col sequence --label-col label \
  --out   output/metrics_baseline.json

python infer.py \
  --model output/models/dnabert2_baseline \
  --in    output/apasites_positives.csv \
  --seq-col sequence \
  --out   output/predictions.csv
```

---

## 8) Reproducibility checklist
- `--seed` set for all sampling/training steps
- Record exact file versions (gtf/fa) in `output/run_manifest.json`
- Log environment: `pip freeze > output/requirements.txt`

---

## 9) Troubleshooting
- **`KeyError: 'end'` when reading BED:** Ensure BED has at least `chrom start end name score strand`. If missing, add columns or set `--bed-cols` flags.
- **Genome mismatch:** All inputs must be **hg38**. If a PAS table is on hg19, run the included liftover logic (script flag) before merging.
- **Tokenizer errors:** Check `--seq-col` exists and contains only `ACGT` (uppercased). N’s are either filtered or mapped by the script.
- **Class imbalance:** Use `--balance` or weighted sampling flags in dataset or trainer scripts.

---

## 10) One‑liner smoke test
```bash
mamba activate apa && \
python polyasite_bed_to_csv.py --bed data/polyasite/*.bed* --gtf data/gencode/*.gtf \
  --ref data/reference/*.fa --out output/polyasite_hg38.csv && \
python polyadb_to_csv.py --pas data/polyadb/*.txt --gtf data/gencode/*.gtf \
  --ref data/reference/*.fa --out output/polyadb_hg38.csv && \
python merge_for_dnabert.py --polyasite output/polyasite_hg38.csv \
  --polyadb output/polyadb_hg38.csv --out output/apasites_positives.csv && \
python make_negatives.py --positives output/apasites_positives.csv \
  --gtf data/gencode/*.gtf --ref data/reference/*.fa --out output/apasites_negatives.csv && \
python make_dataset.py --positives output/apasites_positives.csv \
  --negatives output/apasites_negatives.csv --out output/apasites_with_negatives.csv && \
python train.py --model zhihan1996/DNABERT-2-117M \
  --train output/apasites_with_negatives.csv --val-split 0.1 --epochs 1 && \
python eval.py --model output/models/dnabert2_baseline --test output/apasites_with_negatives.csv
```

---

## 11) Maintainers’ checklist (for PRs)
- ✅ Scripts accept **paths via flags**, no hard‑coded directories
- ✅ Outputs go under `output/`
- ✅ Deterministic sampling with `--seed`
- ✅ CI smoke test uses the one‑liner above with a tiny subset

---

**You’re set.** If your layout already conforms to the folders above, start at **Step 4** and you’ll have a training run in minutes.

