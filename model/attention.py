from torch import nn
import torch

class ScaledDotAttention(nn.Module):
    def __init__(self, d_k: int):
        """
        d_k: Dimension of key and queries
        """
        super().__init__()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.5)
        self.d_k = d_k
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        scores_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores_weights = scores_weights.masked_fill_(~mask, -torch.inf)
        scores_weights = self.softmax(scores_weights)
        scores_weights = self.dropout(scores_weights)
        outputs = torch.matmul(scores_weights, v)
        return outputs

def test_scaled_dot_attention():
    batch_size = 2
    n_heads = 4
    seq_length = 5
    d_k = 16
    d_v = 16

    q = torch.rand(batch_size, n_heads, seq_length, d_k)
    k = torch.rand(batch_size, n_heads, seq_length, d_k)
    v = torch.rand(batch_size, n_heads, seq_length, d_v)

    mask = torch.ones(batch_size, 1, seq_length, seq_length, dtype=torch.bool)
    mask[:, :, 2:, :] = 0

    attention_layer = ScaledDotAttention(d_k)

    output = attention_layer(q, k, v, mask)

    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, n_heads, seq_length, d_v), "Output shape is incorrect"

    print("Test passed!")

# if __name__ == "__main__":
#     test_scaled_dot_attention()
