import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer

def plot_self_attention(model_dir, sequence, layer=-1, average_heads=True, save_path=None):
    """
    Generate and plot a self-attention heatmap for a given DNA sequence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, output_attentions=True)
    model.eval()

    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # (num_layers, batch, num_heads, seq_len, seq_len)
    attn_matrix = attentions[layer][0]  # select batch 0

    if average_heads:
        attn_matrix = attn_matrix.mean(dim=0).cpu().numpy()
    else:
        attn_matrix = attn_matrix[0].cpu().numpy()  # just head 0

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(f"Self-Attention Heatmap (Layer {layer}, {'avg heads' if average_heads else 'head 0'})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

