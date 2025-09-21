#!/usr/bin/env python3
"""
Train DNABERT-2 to classify poly(A) sites from sequence windows.

Inputs
------
- CSV with at least columns: 'sequence' (A/C/G/T/N string) and 'label' (0 or 1)
  (Optionally: 'chrom' to enable chromosome-wise splitting)

Dependencies
------------
pip install transformers accelerate torch scikit-learn pandas

Usage
-----
python train_apa_dnabert2.py \
  --csv apasites_with_negatives.csv \
  --out-dir ./dnabert2_apa \
  --epochs 3 --lr 2e-5 --batch-train 16 --batch-eval 32 \
  --max-length 512 --window-nt 300 --val-frac 0.1 \
  --chromosome-split false
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# --- your prep utilities (imports tokenizer, dataset, collate, etc.) ---
import preplibrary as prep


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.astype(int)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

    out = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        out["auroc"] = roc_auc_score(labels, probs)
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = average_precision_score(labels, probs)
    except Exception:
        out["auprc"] = float("nan")
    return out


def split_train_val(df: pd.DataFrame, val_frac: float, seed: int, chromosome_split: bool):
    if chromosome_split and ("chrom" in df.columns):
        chroms = sorted(df["chrom"].dropna().unique().tolist())
        k = max(1, int(len(chroms) * val_frac))
        val_chroms = set(chroms[-k:])  # simple deterministic split
        df_train = df[~df["chrom"].isin(val_chroms)].reset_index(drop=True)
        df_val = df[df["chrom"].isin(val_chroms)].reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(len(df) * val_frac))
        df_val = df.iloc[:n_val].reset_index(drop=True)
        df_train = df.iloc[n_val:].reset_index(drop=True)

    return df_train, df_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="test.csv", help="Training CSV with columns: sequence,label[,chrom]")
    ap.add_argument("--out-dir", default="./dnabert2_apa", help="Directory to store checkpoints and final model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-train", type=int, default=16)
    ap.add_argument("--batch-eval", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length (use >=320; 512 is safe)")
    ap.add_argument("--window-nt", type=int, default=300, help="Expected nucleotide window length (your CSV windows)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chromosome-split", type=lambda s: s.lower() in {"1","true","yes","y"}, default=False,
                   help="If true and 'chrom' column exists, hold out whole chromosomes for validation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load CSV
    usecols = ["sequence", "label"]
    # include chrom if present to optionally enable chromosome-wise split
    try:
        df_tmp = pd.read_csv(args.csv, nrows=1)
        if "chrom" in df_tmp.columns:
            usecols.append("chrom")
    except Exception:
        pass

    df = pd.read_csv(args.csv, usecols=usecols).dropna()
    df["sequence"] = df["sequence"].astype(str)
    df["label"] = df["label"].astype(int)

    # 2) Train/val split
    df_train, df_val = split_train_val(df, args.val_frac, args.seed, args.chromosome_split)

    # 3) Tokenizer & model
    # tokenizer is already constructed in preplibrary as `tok`
    tokenizer = prep.tok
    model_name = "zhihan1996/DNABERT-2-117M"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "non_PAS", 1: "PAS"},
        label2id={"non_PAS": 0, "PAS": 1},
        trust_remote_code=True,
    )

    # 4) Build datasets using your APADataset (one window per row; no sliding)
    train_ds = prep.APADataset(
        sequences=df_train["sequence"].tolist(),
        tokenizer=tokenizer,
        labels=df_train["label"].tolist(),
        centers=[None] * len(df_train),
        window_nt=args.window_nt,
        stride=args.window_nt,      # stride==window => no sliding
        max_length=args.max_length, # IMPORTANT: override your default 300 to avoid truncation
        k=6,
    )
    val_ds = prep.APADataset(
        sequences=df_val["sequence"].tolist(),
        tokenizer=tokenizer,
        labels=df_val["label"].tolist(),
        centers=[None] * len(df_val),
        window_nt=args.window_nt,
        stride=args.window_nt,
        max_length=args.max_length,
        k=6,
    )

    # 5) TrainingArguments
    # pick evaluation/logging cadence roughly per ~500 steps or each epoch if small
    eval_steps = 500
    if len(train_ds) // max(1, args.batch_train) < 500:
        eval_steps = max(50, len(train_ds) // max(1, args.batch_train))

    targs = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_train,
        per_device_eval_batch_size=args.batch_eval,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=max(10, eval_steps // 5),
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=prep.apa_collate,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 7) Train & evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})

    # 8) Save final
    final_dir = os.path.join(args.out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[OK] Saved model + tokenizer to: {final_dir}")


if __name__ == "__main__":
    main()