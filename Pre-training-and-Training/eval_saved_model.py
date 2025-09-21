# eval_saved_model.py
import os, json, torch, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import preplibrary as prep  # your dataset/collate/utils

@torch.no_grad()
def evaluate(model_dir: str, csv_path: str, batch_size: int = 128, max_length: int = 256, window_nt: int = 203):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True).to(device)
    model.eval()

    # load data
    df = pd.read_csv(csv_path, usecols=[c for c in ["sequence","label","chrom"] if c in pd.read_csv(csv_path, nrows=1).columns]).dropna()
    df["sequence"] = df["sequence"].astype(str)
    df["label"] = df["label"].astype(int)

    ds = prep.APADataset(
        sequences=df["sequence"].tolist(),
        tokenizer=tok,
        labels=df["label"].tolist(),
        centers=[None]*len(df),
        window_nt=window_nt,
        stride=window_nt,
        max_length=max_length,
        k=6,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available(), collate_fn=prep.apa_collate)

    all_probs, all_labels = [], []
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        logits = out.logits
        loss = loss_fn(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    import numpy as np
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    preds = (all_probs >= 0.5).astype(int)
    metrics = {
        "loss": total_loss / max(1, n_batches),
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "samples": int(len(all_labels)),
    }
    try: metrics["auroc"] = roc_auc_score(all_labels, all_probs)
    except Exception: pass
    try: metrics["auprc"] = average_precision_score(all_labels, all_probs)
    except Exception: pass

    # save next to the model
    out_path = os.path.join(model_dir, "metrics_test.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics ->", out_path)
    print(metrics)

if __name__ == "__main__":
    # Example:
    # python eval_saved_model.py /path/to/dnabert2_apa/final /path/to/test.csv
    import sys
    model_dir = sys.argv[1]
    csv_path  = sys.argv[2]
    evaluate(model_dir, csv_path)