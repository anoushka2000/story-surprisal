# ------------------------------------------------------------
# 0.  Setup
#      pip install torch transformers accelerate nltk matplotlib
# ------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import sent_tokenize
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download("punkt")

# ------------------------------------------------------------
# 1.  Load an open causal LM
#    (GPT-2 small keeps VRAM low; swap in a larger or quantised
#     model such as "tiiuae/falcon-7b-instruct" or "mistral-7b") 
# ------------------------------------------------------------
MODEL_NAME = "gpt2"                      # 124 M params, open licence
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# GPT-2 has no PAD token by default ➔ reuse EOS to avoid errors
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# ------------------------------------------------------------
# 2.  Segment a story into sentences
# ------------------------------------------------------------
with open("data/story.txt", encoding="utf-8") as f:
    text = f.read()

sentences = sent_tokenize(text)

# ------------------------------------------------------------
# 3.  Function: sentence-level perplexity
# ------------------------------------------------------------
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
    ).to(device)

    # labels == input_ids ⇒ model predicts *next* token for each position
    loss = model(**enc, labels=enc.input_ids).loss     # mean CE in nats
    return float(torch.exp(loss))                      # perplexity

# ------------------------------------------------------------
# 4.  Compute perplexity for every sentence (batched for speed)
# ------------------------------------------------------------
batch_size = 16
perplexities = []

for i in range(0, len(sentences), batch_size):
    batch = sentences[i : i + batch_size]
    enc   = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    # Use -100 to mask pads so they don't affect loss
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100

    loss = model(**enc, labels=labels).loss            # mean over batch
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

# ------------------------------------------------------------
# 5.  Flag high-surprise sentences (top 10 %)
# ------------------------------------------------------------
threshold = np.percentile(perplexities, 90)

print(f"\n=== Possible turning points (perplexity > {threshold:.1f}) ===")
for idx, (sent, ppl) in enumerate(zip(sentences, perplexities)):
    if ppl > threshold:
        print(f"[{idx:>4}] PPL={ppl:6.1f}  {sent[:120]}…")

# ------------------------------------------------------------
# 6.  Optional: plot the perplexity curve
# ------------------------------------------------------------
plt.figure(figsize=(10, 3))
plt.plot(perplexities, marker=".", linestyle="-", linewidth=0.8)
plt.axhline(threshold, ls="--", label="90th-pct threshold")
plt.xlabel("Sentence index")
plt.ylabel("Perplexity")
plt.title("Sentence-level perplexity trajectory")
plt.legend()
plt.tight_layout()
plt.show()
