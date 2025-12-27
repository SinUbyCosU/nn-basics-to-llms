def scaled_dot_product_attemtion(query, key, value, mask=None):
import torch  # Import the PyTorch library for building neural networks
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.nn.functional as F  # Import functional tools like softmax
import math  # Import math for mathematical operations

# This function calculates attention scores and weighted values
def scaled_dot_product_attemtion(query, key, value, mask=None):
    # Get the size of the last dimension (head dimension)
    d_k = query.size()[-1]
    # Multiply query and key, then scale by sqrt of head dimension
    scaled = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
    # If a mask is given, add it to the scaled scores (to ignore certain positions)
    if mask is not None:
        scaled += mask
    # Apply softmax to get attention weights (make them sum to 1)
    attention = F.softmax(scaled, dim=-1)
    # Multiply attention weights by value to get the output
    values = torch.matmul(attention, value)
    # Return the output values and the attention weights
    return values, attention

# This class does multi-head attention, a key part of transformer models
class MultiHeasAttention(nn.Module):
    # The constructor sets up the layers and parameters
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()  # Call the parent class constructor
        self.inputs_dim = input_dim  # Input feature size
        self.d_model = d_model  # Model feature size (output size)
        self.num_heads = num_heads  # Number of attention heads
        # Each head gets a part of the model size
        self.head_dim = d_model // num_heads
        # Linear layer to create queries, keys, and values from input
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        # Linear layer to combine all heads' outputs
        self.linear_layer = nn.Linear(d_model, d_model)

    # This method does the forward pass (actual computation)
    def forward(self, x, mask=None):
        # x is the input tensor: (batch_size, sequence_length, input_dim)
        batch_size, sequence_length, input_dim = x.size()  # Get sizes

        # Step 1: Make queries, keys, and values from input
        # Output shape: (batch_size, sequence_length, 3 * d_model)
        qkv = self.qkv_layer(x)
        print(f"qkv.size():{qkv.size()}")  # Show the shape for debugging

        # Step 2: Reshape to split into heads and QKV
        # New shape: (batch_size, sequence_length, num_heads, 3 * head_dim)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size() after reshape:{qkv.size()}")

        # Step 3: Move the heads dimension forward for easier processing
        # New shape: (batch_size, num_heads, sequence_length, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        print(f"qkv.size(): {qkv.size()}")

        # Step 4: Split the last dimension into query, key, and value
        # Each will have shape: (batch_size, num_heads, sequence_length, head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}")

        # Step 5: Calculate attention output and weights
        # values: output after attention, attention: weights
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size: {attention.size()}")

        # Step 6: Move heads back to original position and merge them
        # Shape after permute: (batch_size, sequence_length, num_heads, head_dim)
        values = values.permute(0, 2, 1, 3)
        # Shape after reshape: (batch_size, sequence_length, d_model)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        print(f"values.size(): {values.size()}")

        # Step 7: Final linear layer to mix all heads' outputs
        # Output shape: (batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        print(f"out.size(): {out.size()}")
        # Return the final output
        return out