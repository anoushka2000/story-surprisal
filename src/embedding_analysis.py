import torch
from torch.nn import Module
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from utils import load_model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embed_sentence(sentence: str, tokenizer: AutoTokenizer, model: Module) -> torch.Tensor:
    """Return a 1-D pooled embedding for a single sentence."""
    
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)  # last_hidden_state: [1, seq_len, hidden]

    # --- mean pooling (mask out padding) --------------------
    token_embeddings = outputs.last_hidden_state            # [1, L, H]
    attention_mask   = inputs["attention_mask"].float()     # [1, L]
    summed = (token_embeddings * attention_mask.unsqueeze(-1)).sum(dim=1)
    counts = attention_mask.sum(dim=1)
    sentence_embedding = summed / counts                    # [1, H]
    return sentence_embedding.squeeze()                     # [H]


def compute_embeddings(sentences: List[str], batch_size: int = 8):

    embeds = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        # batch_inputs['input_ids']  [B, L]

        with torch.no_grad():
            outputs = model(**batch_inputs) # BaseModelOutputWithPoolingAndCrossAttentions

        mask  = batch_inputs["attention_mask"].float()
        token_emb = outputs.last_hidden_state        # [B, L, H]
        summed = (token_emb * mask.unsqueeze(-1)).sum(dim=1) # [B, H]: one 384-dim vector for each sentence in the batch

        counts = mask.sum(dim=1, keepdim=True) # shape [B, 1]
        embeds.append(summed / counts)

    embeddings = torch.cat(embeds, dim=0).cpu()      # shape: [N_sentences, H]
    return embeddings

def sentencewise_cosine_similarity(embeddings: torch.Tensor):
    prev = embeddings[:-1]
    curr = embeddings[1:]
    sims = cosine_similarity(curr, prev).diagonal()  # length = N-1
    return sims



if __name__=="__main__":

    # Load model and tokenizer
    model, tokenizer = load_model()

    # Load data and tokenize
    with open("data/story.txt", encoding="utf-8") as f:
        text = f.read()
    sentences: List[str] = sent_tokenize(text) 

    embeddings = compute_embeddings(sentences)
    sims = sentencewise_cosine_similarity(embeddings)

    # Flag low-similarity points (semantic "jumps")
    # Use 10th-percentile as a heuristic cut-off.
    threshold = np.percentile(sims, 10)

    print(f"\n=== Potential narrative turning points (similarity < {threshold:.3f}) ===")
    for idx, sim in enumerate(sims, start=1):   # idx == sentence index
        if sim < threshold:
            print(f"[{idx:>4}] sim={sim:.3f} → {sentences[idx][:]}")


    plt.plot(range(1, len(sims)+1), sims)
    plt.axhline(threshold, ls="--")
    plt.xlabel("Sentence index")
    plt.ylabel("Cosine similarity to prev.")
    plt.title("Semantic continuity curve")
    plt.savefig("semantic_continuity_curve.png")
