import logging, torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("nickatcher")

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model     = AutoModel.from_pretrained("AIDA-UPM/star")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_embeddings(messages: list, min_tokens: int = 1):
    enc = tokenizer(
        [m.content for m in messages],
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_length=True,
    ).to(device)

    lengths = enc["attention_mask"].sum(dim=1)
    keep = lengths >= min_tokens
    if keep.sum() == 0:
        hidden = model.config.hidden_size
        return torch.empty(0, hidden), 0

    for k in ("input_ids", "attention_mask", "length"):
        enc[k] = enc[k][keep]

    kept_messages = [m for m, flag in zip(messages, keep.tolist()) if flag]

    with torch.no_grad():
        style_emb = model(
            enc.input_ids,
            attention_mask=enc.attention_mask,
        ).pooler_output

    return style_emb, sum(enc["attention_mask"].sum(dim=1))
