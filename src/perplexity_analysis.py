import torch
from torch.nn import Module
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import numpy as np
import matplotlib.pyplot as plt

from utils import load_model
from typing import List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def sentence_perplexity(sentence: str) -> float:
    """
    Returns exp(mean negative-log-likelihood) for `sentence`.
    Uses only the sentence’s own tokens as context.
    """
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    # labels == input_ids ⇒ model predicts *next* token for each position
    loss = model(**enc, labels=enc.input_ids).loss     # mean CE in nats
    return float(torch.exp(loss))                      # perplexity


def compute_perplexity(model: Module, tokenizer: AutoTokenizer, sentences: List[str], batch_size: int = 8):

    perplexities = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc   = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(DEVICE)

    # Use -100 to mask pads so they don't affect loss
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100

    # Convert to perplexities for each sentence separately
    # loss is averaged over *all* tokens in the batch, so instead
    # compute token-wise NLL to keep sentences separate:
    with torch.no_grad():
        logits = model(**enc).logits                   # [B,L,V]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(2, enc.input_ids.unsqueeze(-1)).squeeze(-1)  # [B,L]

    # Mask pads, then mean-pool per sentence
    masked_nll = nll * enc.attention_mask
    sent_lens  = enc.attention_mask.sum(dim=1)
    sent_nll   = masked_nll.sum(dim=1) / sent_lens
    perplexities.extend(torch.exp(sent_nll).cpu().tolist())

    perplexities = np.array(perplexities)      # length == n_sentences
    return perplexities


if __name__=="__main__":

    # Load model and tokenizer
    model, tokenizer = load_model(model_name="sentence-transformers/all-mpnet-base-v2", causal=False)
    pmodel, ptokenizer = load_model(model_name="gpt2", causal=True)


    # Load data and tokenize
    with open("data/story.txt", encoding="utf-8") as f:
        text = f.read()
    sentences: List[str] = sent_tokenize(text)

    perplexities = compute_perplexity(pmodel, ptokenizer, sentences)
    threshold = np.percentile(perplexities, 90)

    # Top 10% perplexities
    print(f"\n=== Possible turning points (perplexity > {threshold:.1f}) ===")
    for idx, (sent, ppl) in enumerate(zip(sentences, perplexities)):
        if ppl > threshold:
            print(f"[{idx:>4}] PPL={ppl:6.1f}  {sent[:]}…")

    plt.figure(figsize=(10, 3))
    plt.plot(perplexities, marker=".", linestyle="-", linewidth=0.8)
    plt.axhline(threshold, ls="--", label="90th-pct threshold")
    plt.xlabel("Sentence index")
    plt.ylabel("Perplexity")
    plt.title("Sentence-level perplexity trajectory")
    plt.legend()
    plt.tight_layout()
    plt.show()
