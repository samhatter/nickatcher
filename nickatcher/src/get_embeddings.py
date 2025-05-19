from collections import defaultdict
import logging, torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("nickatcher")

tokenizer = AutoTokenizer.from_pretrained("StyleDistance/styledistance")
model     = AutoModel.from_pretrained("StyleDistance/styledistance")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_embeddings(messages: list, *, max_tokens: int = 1):
    if not messages:
        hidden = model.config.hidden_size
        return torch.empty((0, hidden), device=device), 0

    grouped_messages = group_messages(messages=messages, max_tokens=max_tokens)

    enc = tokenizer(
        grouped_messages,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )
        emb = outputs.pooler_output

    token_count = int(enc.attention_mask.sum())
    return emb.cpu(), token_count

def group_messages(messages: list, max_tokens: int):
    grouped_messages = []
    current_chunk = []
    current_tokens = 0
    for message in messages:
        enc = tokenizer(
            message.content,
            return_tensors="pt",
            return_length=True,
        ).to(device)
        num_tokens = enc["attention_mask"].sum().item()
        if current_tokens + num_tokens > max_tokens:
            grouped_messages.append("\n".join(current_chunk))
            current_chunk = [message.content]
            current_tokens = num_tokens
        else:
            current_chunk.append(message.content)
            current_tokens += num_tokens
    if current_chunk:
        grouped_messages.append("\n".join(current_chunk))
    logger.debug(f"Grouped {len(messages)} messages to {len(grouped_messages)}")
    return grouped_messages