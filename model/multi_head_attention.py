import torch
from torch import nn
from model.attention import ScaledDotAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embedding: int, d_k: int, n_heads: int):
        """
        d_k: dimension of keys, querries, values (equal)
        n_heads: number of attention head
        """
        super().__init__()
        self.dim_embedding = dim_embedding
        self.d_k = d_k
        self.n_heads = n_heads
        self.dropout = nn.Dropout(p=0.5)

        # Linear layer to pay attention to different context. Model can learn which word in the sentence has more impact to each
        # Instead of only using embedding layer, adding linear layer which has learnable parameters
        self.weights_q = nn.Linear(self.dim_embedding, self.n_heads * self.d_k, bias=False)
        self.weights_k = nn.Linear(self.dim_embedding, self.n_heads * self.d_k, bias=False)
        self.weights_v = nn.Linear(self.dim_embedding, self.n_heads * self.d_k, bias=False)

        # Attention only wants to learn its context in the sentence -> no bias needed
        self.attention = ScaledDotAttention(d_k=self.d_k)

        # Final linear layer
        self.project = nn.Linear(self.n_heads * self.d_k, self.dim_embedding, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Shape of q, k, v: (batch_size, sequence_length, dim_embedding)
        batch_size, sequence_length, _ = q.size()
        q = self.weights_q(q)
        k = self.weights_k(k)
        v = self.weights_v(v)
        # Output of q, k, v now is (batch_size, sequence_length, n_heads * d_k), we want to have (batch_size, n_heads, sequence_length, d_k) for attention calculation
        q = q.reshape(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, sequence_length, self.n_heads, self.d_k).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        # Perform attention
        attention_output = self.attention(q, k, v, mask)

        # Concatenate heads, transpose back for final linear layer
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, self.n_heads * self.d_k)

        # Final layer projection
        output = self.project(attention_output)
        output = self.dropout(output)
        return output


def test_multi_head_attention():
    batch_size = 2
    seq_length = 5
    dim_embedding = 40
    d_k = 10
    n_heads = 4

    q = torch.rand(batch_size, seq_length, dim_embedding)
    k = torch.rand(batch_size, seq_length, dim_embedding)
    v = torch.rand(batch_size, seq_length, dim_embedding)

    mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.bool)

    attention_layer = MultiHeadAttention(dim_embedding, d_k, n_heads)

    output = attention_layer(q, k, v, mask)

    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        seq_length,
        dim_embedding,
    ), "Output shape is incorrect"

    print("Test passed!")


# if __name__ == "__main__":
#     test_multi_head_attention()
