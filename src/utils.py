import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", causal: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if causal:
        model     = AutoModelForCausalLM.from_pretrained(model_name)
    else: 
        model = AutoModel.from_pretrained(model_name)
    # GPT-2 has no PAD token by default ➔ reuse EOS to avoid errors
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()                                # inference mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer