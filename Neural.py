import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_datasets
from tranformers import AutoTokenizer

dataset= load_dataset("ag_news")
tokenizer= AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenizer_function(Examples):
    return tokenizer(Examples["text"],padding="max_length",truncation=True,max_length=128)
tokenized_datsets=dataset.map(tokenizer_function, batched=True)

tokenized_datasets=tokenized_datasets.remove_columns=(["text"])
tokenized_datasets=tokenized_datasets.rename_columns({"label":"labels"})
tokenized_datasets.set_format("torch")

from torch.utils.data import Dataloader

train_datalaoder=Dataloader(tokenized_datasets["train"],batch_Size=32, shuffle=True)
test_datalaoder=DataLoader(tokenized_datatsets["test"],batch_size=32)

import torch.nn as nn

class TextClassifier(self, input_ids):
    def __init__(self, vocab_Size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()