# Import BERT tokenizer and model from HuggingFace Transformers
from transformers import BertTokenizer, BertModel
# Import PyTorch for tensor operations
import torch

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load the pre-trained BERT model
model = BertModel.from_pretrained("bert-base-uncased")

# Example sentences to encode
sentences = [
    "Hello world!",  # Example sentence 1
    "I am Tanushree.",  # Example sentence 2
    "Welcome to my github",  # Example sentence 3
    "I hope you love transformers as much as i do"  # Example sentence 4
]

# Tokenize sentences and convert to PyTorch tensors
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

# Disable gradient calculation (inference mode)
with torch.no_grad():
    # Get model outputs for the input sentences
    outputs = model(**inputs)
    # Compute mean-pooled sentence embeddings
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1)

# Print the resulting sentence embeddings
print(sentence_embeddings)