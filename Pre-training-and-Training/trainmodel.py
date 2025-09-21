#!/usr/bin/env python3
"""
Train DNABERT-2 to classify poly(A) sites from sequence windows.
"""

import os
# quiet the tokenizers fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import random
import numpy as np
import pandas as pd
import torch
from inspect import signature

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# --- your prep utilities (tokenizer, dataset, collate, etc.) ---
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
    try: out["auroc"] = roc_auc_score(labels, probs)
    except Exception: out["auroc"] = float("nan")
    try: out["auprc"] = average_precision_score(labels, probs)
    except Exception: out["auprc"] = float("nan")
    return out


def split_train_val(df: pd.DataFrame, val_frac: float, seed: int, chromosome_split: bool):
    if chromosome_split and ("chrom" in df.columns):
        chroms = sorted(df["chrom"].dropna().unique().tolist())
        k = max(1, int(len(chroms) * val_frac))
        val_chroms = set(chroms[-k:])
        df_train = df[~df["chrom"].isin(val_chroms)].reset_index(drop=True)
        df_val   = df[df["chrom"].isin(val_chroms)].reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(len(df) * val_frac))
        df_val   = df.iloc[:n_val].reset_index(drop=True)
        df_train = df.iloc[n_val:].reset_index(drop=True)
    return df_train, df_val


# ---- version-agnostic TrainingArguments builder ----
def make_training_args(args, eval_steps):
    params = set(signature(TrainingArguments.__init__).parameters)

    kw = dict(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_train,
        per_device_eval_batch_size=args.batch_eval,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    # prefer staying on CPU if no CUDA
    if "no_cuda" in params and not torch.cuda.is_available():
        kw["no_cuda"] = True

    # add only if supported by the installed transformers version
    if "evaluation_strategy" in params: kw["evaluation_strategy"] = "steps"
    if "eval_strategy" in params:       kw["eval_strategy"] = "steps"  # very old versions

    if "eval_steps" in params:              kw["eval_steps"] = eval_steps
    if "logging_steps" in params:           kw["logging_steps"] = max(10, eval_steps // 5)
    if "save_steps" in params:              kw["save_steps"] = eval_steps
    if "save_total_limit" in params:        kw["save_total_limit"] = 2
    if "load_best_model_at_end" in params:  kw["load_best_model_at_end"] = True
    if "metric_for_best_model" in params:   kw["metric_for_best_model"] = "auprc"
    if "greater_is_better" in params:       kw["greater_is_better"] = True
    if "dataloader_num_workers" in params:  kw["dataloader_num_workers"] = 2

    return TrainingArguments(**kw)
# ---------------------------------------------------


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

    # 3) Tokenizer & model (disable FlashAttention on CPU; set labels on config)
    tokenizer = prep.tok
    model_name = "zhihan1996/DNABERT-2-117M"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_labels = 2
    config.id2label = {0: "non_PAS", 1: "PAS"}
    config.label2id = {"non_PAS": 0, "PAS": 1}

    if not torch.cuda.is_available():
        # DNABERT-2 variants may use any of these flags; turn them off on CPU
        for k in ("use_flash_attn", "flash_attn", "flash_attention"):
            if hasattr(config, k):
                setattr(config, k, False)
        os.environ["DISABLE_FLASH_ATTN"] = "1"
        os.environ["FLASH_ATTENTION_DISABLED"] = "1"
        os.environ["USE_FLASH_ATTENTION"] = "0"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,            # pass adjusted config; no num_labels/label maps as kwargs
        trust_remote_code=True,
    )

    # 4) Datasets (one window per row; no sliding)
    train_ds = prep.APADataset(
        sequences=df_train["sequence"].tolist(),
        tokenizer=tokenizer,
        labels=df_train["label"].tolist(),
        centers=[None] * len(df_train),
        window_nt=args.window_nt,
        stride=args.window_nt,
        max_length=args.max_length,
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

    # 5) TrainingArguments (robust to transformers version)
    steps_per_epoch = max(1, len(train_ds) // max(1, args.batch_train))
    eval_steps = max(50, steps_per_epoch)
    targs = make_training_args(args, eval_steps)

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,  # OK; deprecation warning is harmless
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