import logging
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger('nickatcher')
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
model = AutoModel.from_pretrained('AIDA-UPM/star')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_embeddings(messages: list):
    encoded_texts = tokenizer([message.content for message in messages],  padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        style_embeddings = model(encoded_texts.input_ids,
                                attention_mask=encoded_texts.attention_mask).pooler_output
    mask = encoded_texts["attention_mask"]      # shape (batch, seq_len)
    tokens = sum(mask.sum(dim=1))
    return style_embeddings, tokens