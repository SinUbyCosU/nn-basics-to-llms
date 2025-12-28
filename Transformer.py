
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import MultiHeadAttention from the other file
from Multi-Head-Attention-Mechnaism import MultiHeasAttention
# Import positional encoding from the other file
from Positional_Embeddings import positional_encoding

class SimpleTransformerBlock(nn.Module):
	def __init__(self, d_model, num_heads, d_ff, input_dim=None):
		super().__init__()
		self.attn = MultiHeasAttention(input_dim or d_model, d_model, num_heads)
		self.norm1 = nn.LayerNorm(d_model)
		self.ff = nn.Sequential(
			nn.Linear(d_model, d_ff),
			nn.ReLU(),
			nn.Linear(d_ff, d_model)
		)
		self.norm2 = nn.LayerNorm(d_model)

	def forward(self, x, mask=None):
		attn_out = self.attn(x, mask)
		x = self.norm1(x + attn_out)
		ff_out = self.ff(x)
		x = self.norm2(x + ff_out)
		return x

class SimpleTransformer(nn.Module):
	def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, max_seq_len, vocab_size=None):
		super().__init__()
		self.embedding = nn.Linear(input_dim, d_model) if vocab_size is None else nn.Embedding(vocab_size, d_model)
		self.pos_encoding = torch.tensor(positional_encoding(max_seq_len, d_model).numpy(), dtype=torch.float32)
		self.layers = nn.ModuleList([
			SimpleTransformerBlock(d_model, num_heads, d_ff, input_dim=None)
			for _ in range(num_layers)
		])
		self.out = nn.Linear(d_model, vocab_size) if vocab_size is not None else nn.Identity()

	def forward(self, x, mask=None):
		# x: (batch, seq_len, input_dim) or (batch, seq_len) if using embedding
		if isinstance(self.embedding, nn.Embedding):
			x = self.embedding(x)
		else:
			x = self.embedding(x)
		seq_len = x.size(1)
		x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
		for layer in self.layers:
			x = layer(x, mask)
		return self.out(x)

# Example usage (for demonstration, not for training):
if __name__ == "__main__":
	batch_size = 2
	seq_len = 10
	d_model = 32
	num_heads = 4
	d_ff = 64
	num_layers = 2
	input_dim = 32
	max_seq_len = 20
	# Random input
	x = torch.randn(batch_size, seq_len, input_dim)
	model = SimpleTransformer(input_dim, d_model, num_heads, d_ff, num_layers, max_seq_len)
	out = model(x)
	print("Output shape:", out.shape)
