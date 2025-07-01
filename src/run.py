import os
import glob
import numpy as np
import pandas as pd

TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID") or 1)

from typing import List
from data_loading import get_text 
from utils import load_model
from embedding_analysis import sent_tokenize, compute_embeddings, sentencewise_cosine_similarity
from perplexity_analysis import compute_perplexity

datadirs = glob.glob("/u/abhutani/story-surprisal/StoryComplexity/**-**")
datadir = datadirs[TASK_ID]
text_files =  glob.glob(os.path.join(datadir, "text-files", "**.txt"))
save_dir = os.path.join("/u/abhutani/story-surprisal/results", os.path.basename(datadir))
os.makedirs(save_dir, exist_ok=True)
emb_dir =  os.path.join(save_dir, "embedding-files")
os.makedirs(emb_dir, exist_ok=True)
perplexity_dir =  os.path.join(save_dir, "perplexity-files")
os.makedirs(perplexity_dir, exist_ok=True)

model, tokenizer = load_model(model_name="sentence-transformers/all-mpnet-base-v2", causal=False)
pmodel, ptokenizer = load_model(model_name="gpt2", causal=True)

for text_file in text_files:
    # Load text
    with open(text_file, encoding="utf-8") as f:
        text = f.read()
    file_num: int  = os.path.splitext(os.path.basename(text_file))[0]
    sentences: List[str] = sent_tokenize(text) 

    # Compute embeddings 
    embeddings = compute_embeddings(tokenizer, model, sentences)
    sims = sentencewise_cosine_similarity(embeddings)
    file = os.path.join(emb_dir, f"{file_num}.npy")
    ## Save embeddings
    with open(file, 'wb') as f:
        np.save(file, embeddings.numpy())
    ## Save similarity
    df = pd.DataFrame(sims)
    df.to_csv(os.path.join(emb_dir, f"{file_num}.csv"))
    
    # Compute preplexity
    perplexities = compute_perplexity(pmodel, ptokenizer, sentences)
    df = pd.DataFrame(sims)
    ## Save preplexity
    df.to_csv(os.path.join(perplexity_dir, f"{file_num}.csv"))
