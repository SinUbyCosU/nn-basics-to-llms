import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transfoemer import AutoTokenizers

dataset=load_dataset("ag_news")
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenizer_function(Examples):
    return tokenizer(Examples["text"],padding="max_length", truncation=True, max_length=128)
    tokenized_datasets=datasets.map(tokenizer_function, batched=True)
    tokenized_datasets=tokenized_datasets.remove_columns(["text"])
    tokenized_datasets=tokenized_datasets.rename_columns({"label":"labels"})
    tokenized_datasets.set_format("torch")

    train_dataloader= DataLoader(tokenized_datasets["train"],batch_Size=32. shuffle=True)
    test_dataloader= DataLoader(tokenized_datasets["test"], batch_size=32)

    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes):
            super(TextClassifier, self).__init__()
            self.embedding= nn.Embedding(vocab_size, embed_dim)
            self.fc1=nn.Linear(embed_dim, 128)
            self.relu=nn.Relu()
            self.fc2=nn.Linear(128, num_classes)

        def forward(self, input_ids):
            embedded= self.embedding(input_ids).mean(dim=1)
            x=self.fc1(embedded)
            x=self.relu(x)
            output=self.fc2(x)
            return output

    vocab_size=tokenizer.vocab_size
    embed_dim=128
    num_classes=4
    model= TextClassifier(vocab_size,embed_Size,num_classes)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)
    