#!/usr/bin/env python3
"""
Train DNABERT-2 to classify poly(A) sites from sequence windows.
Optimized for GPU throughput (A100/Colab) and robust evaluation memory use.
"""

import os
# quieter tokenizers, better CUDA allocator behavior
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

# Your prep utilities (tokenizer, dataset, collate)
import preplibrary as prep

# ---- GPU math knobs: TF32 can give ~10–20% extra throughput on Ampere+ ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.astype(int)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    out = {"accuracy": accuracy_score(labels, preds),
           "f1": f1_score(labels, preds, zero_division=0)}
    try: out["auroc"] = roc_auc_score(labels, probs)
    except Exception: out["auroc"] = float("nan")
    try: out["auprc"] = average_precision_score(labels, probs)
    except Exception: out["auprc"] = float("nan")
    return out


def _preprocess_logits_for_metrics(logits, labels):
    """
    Critical for avoiding eval OOM:
    Detach, cast to fp32, and move logits to CPU before Trainer accumulates them.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.detach().float().cpu()


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
def make_training_args(args):
    """
    Build TrainingArguments robustly across transformers versions.
    We set eval/save strategies *after* init to avoid keyword errors,
    and only enable load_best_model_at_end when strategies match.
    """
    params = set(signature(TrainingArguments.__init__).parameters)

    # minimal safe kwargs for __init__
    kw = dict(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_train,
        per_device_eval_batch_size=args.batch_eval,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
    )

    # precision / optimizer
    if torch.cuda.is_available():
        if "bf16" in params:
            kw["bf16"] = True       # best on A100
        elif "fp16" in params:
            kw["fp16"] = True
        if "optim" in params:
            kw["optim"] = "adamw_torch_fused"
    else:
        if "no_cuda" in params:
            kw["no_cuda"] = True

    # dataloader runtime knobs
    if "dataloader_pin_memory" in params:
        kw["dataloader_pin_memory"] = True
    if "dataloader_persistent_workers" in params:
        kw["dataloader_persistent_workers"] = True if args.num_workers > 0 else False
    if "dataloader_prefetch_factor" in params:
        kw["dataloader_prefetch_factor"] = getattr(args, "prefetch", 2)

    # eval accumulation to limit GPU memory during eval concatenation
    if "eval_accumulation_steps" in params:
        kw["eval_accumulation_steps"] = args.eval_accum_steps

    # create with the safe core set
    targs = TrainingArguments(**kw)

    # ---- Force matching strategies post-init (handles old/new versions) ----
    got_eval_attr = False
    if hasattr(targs, "evaluation_strategy"):
        setattr(targs, "evaluation_strategy", "epoch")
        got_eval_attr = True
    elif hasattr(targs, "eval_strategy"):
        setattr(targs, "eval_strategy", "epoch")
        got_eval_attr = True

    got_save_attr = False
    if hasattr(targs, "save_strategy"):
        setattr(targs, "save_strategy", "epoch")
        got_save_attr = True

    # Best model settings: only enable if both strategies exist (and thus match)
    can_load_best = got_eval_attr and got_save_attr
    if hasattr(targs, "load_best_model_at_end"):
        setattr(targs, "load_best_model_at_end", bool(can_load_best))
    if hasattr(targs, "metric_for_best_model"):
        setattr(targs, "metric_for_best_model", "auprc")
    if hasattr(targs, "greater_is_better"):
        setattr(targs, "greater_is_better", True)

    # Optional: keep save_total_limit small
    if hasattr(targs, "save_total_limit"):
        setattr(targs, "save_total_limit", 2)

    # small debug print
    try:
        ev = getattr(targs, "evaluation_strategy", getattr(targs, "eval_strategy", "N/A"))
        sv = getattr(targs, "save_strategy", "N/A")
        lb = getattr(targs, "load_best_model_at_end", "N/A")
        print(f"[args] eval_strategy={ev}  save_strategy={sv}  load_best_model_at_end={lb}  "
              f"eval_accum_steps={getattr(targs,'eval_accumulation_steps','N/A')}")
    except Exception:
        pass

    return targs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/content/apasites_dataset.csv",
                    help="Training CSV with columns: sequence,label[,chrom]")
    ap.add_argument("--out-dir", default="./dnabert2_apa",
                    help="Directory to store checkpoints and final model")

    # Throughput knobs
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-train", type=int, default=64)
    ap.add_argument("--batch-eval", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="Gradient accumulation steps (keep if VRAM is tight)")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers (lower if RAM is tight)")
    ap.add_argument("--logging-steps", type=int, default=200)
    ap.add_argument("--eval-accum-steps", type=int, default=16,
                    help="Accumulate eval results on CPU every N steps to avoid GPU OOM")

    # Sequence sizing
    ap.add_argument("--max-length", type=int, default=256,
                    help="Tokenizer max_length (shorter = faster; 256 fits ~203nt)")
    ap.add_argument("--window-nt", type=int, default=203,
                    help="Expected nucleotide window length in your CSV")

    # Split & reproducibility
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chromosome-split",
                    type=lambda s: s.lower() in {"1","true","yes","y"}, default=False,
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

    # 3) Tokenizer & model (disable FlashAttention on CPU)
    tokenizer = prep.tok
    model_name = "zhihan1996/DNABERT-2-117M"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_labels = 2
    config.id2label = {0: "non_PAS", 1: "PAS"}
    config.label2id = {"non_PAS": 0, "PAS": 1}

    if not torch.cuda.is_available():
        for k in ("use_flash_attn", "flash_attn", "flash_attention"):
            if hasattr(config, k):
                setattr(config, k, False)
        os.environ["DISABLE_FLASH_ATTN"] = "1"
        os.environ["FLASH_ATTENTION_DISABLED"] = "1"
        os.environ["USE_FLASH_ATTENTION"] = "0"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, trust_remote_code=True
    )

    if torch.cuda.is_available():
        model.to("cuda")
        print("✅ Model on GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ GPU not detected; training will be much slower (CPU).")

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

    print(f"Samples: train={len(train_ds)}  val={len(val_ds)}  "
          f"batch_train={args.batch_train}  batch_eval={args.batch_eval}  "
          f"max_length={args.max_length}  workers={args.num_workers}  accum={args.grad_accum}  "
          f"eval_accum={args.eval_accum_steps}")

    # 5) TrainingArguments (epoch eval/save; fused optim; bf16/fp16; eval accumulation)
    targs = make_training_args(args)

    # 6) Trainer (with CPU-offload of logits)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,          # deprecation warning is harmless
        data_collator=prep.apa_collate,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics,   # <-- key fix
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