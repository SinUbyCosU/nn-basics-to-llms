from transformers import BertTokenizer, BertModel
import torch

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=BertModel.from_pretrained("bert-base-uncased")

sentences=["Hello world!", "I am Tanushree.","Welcome to my github", "I hope you love transformers as much as i do"]

inputs=tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    outputs=model(**inputs)
    sentence_embeddings=outputs.last_hidden_state.mean(dim=1)

print(sentence_embeddings)