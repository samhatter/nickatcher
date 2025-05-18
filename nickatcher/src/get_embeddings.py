import logging
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger('nickatcher')

def get_embeddings(messages: list):
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    model = AutoModel.from_pretrained('AIDA-UPM/star')
    encoded_texts = tokenizer([message['message'] for message in messages],  padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        style_embeddings = model(encoded_texts.input_ids,
                                attention_mask=encoded_texts.attention_mask).pooler_output
    return style_embeddings