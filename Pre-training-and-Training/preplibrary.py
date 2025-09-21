# imports
import re
from typing import Iterable, List, Dict, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load tokenizer/model 
tok = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# functions to prep one sequence

def prep_sequence(seq: str) -> str:
    """Clean an RNA string so it's safe for models. Removes whitespace, uppercases all, U->T, map non-ACGTN to N.
    
    >>> prep_sequence("aau uug-cag\n")
    'AATTTGCAG'
    >>> prep_sequence("AAUXX")
    'AATNN'

    """
    
    if seq is None:
        return ""
    s = "".join(str(seq).split()).upper()   
    s = s.replace("U", "T")                 
    s = re.compile(r"[^ACGTN]").sub("N", s)
    return s

def kmers(seq: str, k: int = 6, pad_char: str = "N") -> List[str]:
    """ Manual tokenizer for if DNABERT-2 fails. Breaks sequence into substrings of length k, since DNABERT-style models 
    read DNA as sliding windows of k bases. Adds N for padding if sequence has less than k nucleotides

    >>> kmers("AAAUUUG")
    ['AAATTT', 'AATTTG']
    >>> kmers("A")
    ['ANNNNN']
    
    """
    
    s = prep_sequence(seq)
    n = len(s)
    if n < k:
        s = s + pad_char * (k - n)
        return [s]
    return [s[i:i+k] for i in range(n - k + 1)]

def tokenize_dnabert2(seq: str, tokenizer, max_length: int = 300, k: int = 6) -> Dict[str, torch.Tensor]:
    """ Turn a sequence into model-ready tensors of form (input_ids, attention_mask). Tries the DNABERT-2 tokenizer 
    directly (it usually does k-merization under the hood when loaded with `trust_remote_code=True`). If that fails, 
    it falls back to manual k-mers joined by spaces.
    
    >>> enc = tokenize_dnabert2("AAAUUUGCAG", tok, max_length=64)
    >>> enc["input_ids"].shape  
    torch.Size([1, 64])
    
    """
    
    s = prep_sequence(seq)
    try:
        return tokenizer(s, add_special_tokens=True, padding="max_length", truncation=True, max_length=max_length,
            return_tensors="pt", return_token_type_ids=False)
    except Exception:
        toks = " ".join(kmers(s, k))
        return tokenizer(toks, add_special_tokens=True, padding="max_length", truncation=True, max_length=max_length,
            return_tensors="pt", return_token_type_ids=False)

def make_windows(seq: str, window_nt: int = 300, stride: int = 100, center_index: Optional[int] = None) -> List[str]:
    """Slice a (possibly long) sequence into fixed-length nucleotide windows.

    DNABERT-style models run on fixed-size inputs, so for APA tasks we want either one window centered on a 
    candidate site or sliding windows that scan the whole sequence.

    >>> make_windows("A"*20, window_nt=8, stride=4)
    ['AAAAAAAA', 'AAAAAAAA', 'AAAAAAAA', 'AAAAAAAA']
    
    """
    s = prep_sequence(seq)
    n = len(s)
    if n == 0:
        return []

    if center_index is not None:
        c = max(0, min(n - 1, int(center_index)))   
        half = window_nt // 2
        left = max(0, c - half)
        right = min(n, left + window_nt)
        if right - left < window_nt and left > 0:
            left = max(0, right - window_nt)
        return [s[left:right]]

    if n <= window_nt:
        return [s]
    return [s[i:i+window_nt] for i in range(0, n - window_nt + 1, stride)]

# scale up to work on lots of raw sequences at once

class APADataset(Dataset):
    """ PyTorch Dataset that:
      1) expands each raw sequence into one or more windows (via make_windows),
      2) tokenizes each window for DNABERT-2 (via tokenize_dnabert2),
      3) stores tensors + lightweight metadata for DataLoader batching.

    Each original sequence may yield multiple "samples" (one per window).

    Shapes 
    -----
    - B: batch size (how many windowed, tokenized sequences)
    - L: tokens per windowed, tokenized sequences after padding
    
    Items
    -----
    __getitem__(idx) returns a dict with:
        - 'input_ids'      : LongTensor[L]
        - 'attention_mask' : LongTensor[L]
        - 'meta'           : dict with {'seq_index', 'win_index', 'len_nt'}
        - 'labels'         : LongTensor[] (only if labels were provided)

    Example
    -----
    >>> seqs = ["AAAUUUGCAG", "ccauuuaaaGG"]
    >>> ds = APADataset(seqs, tok, window_nt=8, stride=4, max_length=64)
    >>> len(ds)                        
    4
    >>> sample = ds[0]
    >>> sorted(sample.keys())
    ['attention_mask', 'input_ids', 'meta']
    
    """
    def __init__(self, sequences: Iterable[str], tokenizer, labels: Optional[Iterable[int]] = None, 
                 centers: Optional[Iterable[Optional[int]]] = None, window_nt: int = 300, stride: int = 100, max_length: int = 300, k: int = 6):
        self.samples = []
        seq_list = list(sequences)
        lab_list = list(labels) if labels is not None else [None]*len(seq_list)
        cen_list = list(centers) if centers is not None else [None]*len(seq_list)

        for i, (seq, lab, cen) in enumerate(zip(seq_list, lab_list, cen_list)):
            for j, w in enumerate(make_windows(seq, window_nt=window_nt, stride=stride, center_index=cen)):
                enc = tokenize_dnabert2(w, tokenizer, max_length=max_length, k=k)
                item = {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                        "meta": {"seq_index": i, "win_index": j, "len_nt": len(w)}}
                if lab is not None:
                    item["labels"] = torch.tensor(int(lab), dtype=torch.long)
                self.samples.append(item)

    def __len__(self) -> int:
        # number of windowed samples
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # return one windowed and tokenized sample
        return self.samples[idx]

def apa_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of APADataset items into a single batch for the model.
      - Stacks 'input_ids' and 'attention_mask' into (B, L) tensors.
      - Preserves 'meta' as a Python list (useful for mapping predictions back).
      - If any item has 'labels', stacks them into shape (B,)
    
    Parameters
    ----------
    batch : List[Dict]
        What DataLoader hands us (a list of items from APADataset.__getitem__).

    Returns
    -------
    Dict[str, Any]
        {
          'input_ids'     : LongTensor[B, L],
          'attention_mask': LongTensor[B, L],
          'labels'        : LongTensor[B]   (only if present),
          'meta'          : List[dict]
        }

    Example
    -------
    >>> seqs = ["AAAUUUGCAGUAAUGA", "ccauuuaaaGG"]
    >>> ds = APADataset(seqs, tok, window_nt=12, stride=6, max_length=64)
    >>> loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=apa_collate)
    >>> batch = next(iter(loader))
    >>> batch["input_ids"].shape  
    torch.Size([2, 64])
    
    """
    
    out = {"input_ids": torch.stack([b["input_ids"] for b in batch]), "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "meta": [b["meta"] for b in batch]}
    if any("labels" in b for b in batch):
        out["labels"] = torch.stack([b["labels"] for b in batch if "labels" in b])
    return out

