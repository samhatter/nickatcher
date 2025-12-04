import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading

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

tokenizer_lock = threading.Lock()

_embedding_cache = {}
_embedding_inflight = {}
_cache_lock = asyncio.Lock()

async def get_embeddings(messages: list, batch_size=100):
    loop = asyncio.get_running_loop()
    cache_key = tuple(messages)

    async with _cache_lock:
        if cache_key in _embedding_cache:
            return _embedding_cache[cache_key]

        future = _embedding_inflight.get(cache_key)
        if future is None:
            future = loop.run_in_executor(executor, _get_embeddings_sync, messages, batch_size)
            _embedding_inflight[cache_key] = future

    try:
        embeddings = await future
    except Exception:
        async with _cache_lock:
            _embedding_inflight.pop(cache_key, None)
        raise

    embeddings = embeddings.clone()

    async with _cache_lock:
        _embedding_inflight.pop(cache_key, None)
        _embedding_cache.setdefault(cache_key, embeddings)
        return _embedding_cache[cache_key]

def _get_embeddings_sync(messages: list, batch_size=100):
    if not messages:
        hidden = model.config.hidden_size
        return torch.empty((0, hidden), device=device)

    all_embeddings = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]

        with tokenizer_lock:
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

    final_embeddings = torch.cat(all_embeddings, dim=0)
    return final_embeddings

def group_messages(messages: list):
    grouped_messages = []
    current_chunk = []
    current_tokens = 0
    total_tokens_seen = 0
    
    for message in messages:
        with tokenizer_lock:
            enc = tokenizer(
                message.content,
                truncation=True,
                max_length=EMBEDDING_MAX_TOKENS,
                return_tensors="pt",
                return_length=True,
            ).to(device)
        num_tokens = enc["attention_mask"].sum().item()
        total_tokens_seen += num_tokens
        
        if current_tokens + num_tokens > EMBEDDING_MAX_TOKENS:
            grouped_messages.append("\n".join(current_chunk))
            current_chunk = [message.content]
            current_tokens = num_tokens
        else:
            current_chunk.append(message.content)
            current_tokens += num_tokens
    if current_chunk:
        grouped_messages.append("\n".join(current_chunk))
    logger.debug(
        "Grouped %d messages into %d chunks",
        len(messages),
        len(grouped_messages),
    )
    
    return grouped_messages, total_tokens_seen
