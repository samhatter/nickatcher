import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
import os

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("nickatcher")

tokenizer = AutoTokenizer.from_pretrained("StyleDistance/styledistance")
model     = AutoModel.from_pretrained("StyleDistance/styledistance")
model.eval()

device = torch.device(os.getenv('DEVICE', 'cpu'))
model.to(device)

EMBEDDING_MAX_TOKENS = 512

executor = ThreadPoolExecutor(max_workers=int(os.getenv('EXECUTOR_THREADS', '4')))
    
async def get_embeddings(messages: list, *, batch_size=100):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _get_embeddings_sync, messages, batch_size)

def _get_embeddings_sync(messages: list, *, batch_size=100):
    if not messages:
        hidden = model.config.hidden_size
        return torch.empty((0, hidden), device=device), 0

    all_embeddings = []
    total_tokens = 0
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=EMBEDDING_MAX_TOKENS,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
            )
            emb = outputs.pooler_output

        all_embeddings.append(emb.cpu())
        total_tokens += int(enc.attention_mask.sum())

    final_embeddings = torch.cat(all_embeddings, dim=0)
    return final_embeddings, total_tokens

def group_messages(messages: list):
    grouped_messages = []
    current_chunk = []
    current_tokens = 0
    for message in messages:
        enc = tokenizer(
            message.content,
            truncation=True,
            max_length=EMBEDDING_MAX_TOKENS,
            return_tensors="pt",
            return_length=True,
        ).to(device)
        num_tokens = enc["attention_mask"].sum().item()
        if current_tokens + num_tokens > EMBEDDING_MAX_TOKENS:
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
