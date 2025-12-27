import torch
from transformers import BertTokenizer, BertModel

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=BertModel.from_pretrained("bert-base-uncased")

sentences={"Hi there!"}

inputs=tokenizer(sentences,return_tensors=True, padding=True,truncation=True)
with torch.no_grad():
    outputs=model(**inputs)
    sentence_embeddings=outputs.last_hidden_state.mean(dim=1)

    print(sentence_embeddings)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sacled_dot_product(q,k,v,mask=None):
    d_k=q.size()[-1]
    scaled=torch.matmul(q,k.transpose(-1,-2)/math.sqrt(d_k))
    if mask is not None:
        scaled+=mask
        attention=F.softmax(scaled, dim=-1)
        values=torch.matmul(attention,v)
        return values, attention