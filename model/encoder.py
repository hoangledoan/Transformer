from torch import nn
import torch
from model.multi_head_attention import MultiHeadAttention
from model.feedforward import FeedForwardNeuralNetwork


class EncoderBlock(nn.Module):
    def __init__(self, dim_embedding: int, sequence_length: int, n_heads: int, d_ff: int):
        super().__init__()
        self.multi_head = MultiHeadAttention(dim_embedding, sequence_length, n_heads)
        # self.multi_head output: (n_heads * sequence_length, dim_embedding)
        # only want to do normalization on embedding
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.ffn = FeedForwardNeuralNetwork(dim_embedding, d_ff)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

    def forward(
        self, inputs: torch.Tensor, pad_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Multihead -> residual connection + layer normalization -> feed forward -> residual connection -> layer normalization
        outputs = self.multi_head(inputs, inputs, inputs, pad_mask) + inputs
        outputs = self.layer_norm1(outputs)
        outputs = self.ffn(outputs) + outputs
        outputs = self.layer_norm2(outputs)
        return outputs


class Encoder(nn.Module):
    def __init__(self, dim_embedding: int, sequence_length: int, n_heads: int, d_ff: int, n: int):
        super().__init__()

        self.stack = nn.ModuleList(
            [
                EncoderBlock(
                    dim_embedding=dim_embedding, sequence_length=sequence_length, n_heads=n_heads, d_ff=d_ff
                )
                for _ in range(n)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, encoder_mask: torch.Tensor = None
    ) -> torch.Tensor:
        outputs = inputs
        for encoder in self.stack:
            outputs = encoder(outputs, encoder_mask)
        # Output has shape (batch_size, seq_length, dim_embedding)
        return outputs


def test_encoder():
    batch_size = 2
    seq_length = 5
    dim_embedding = 40
    sequence_length = 10
    n_heads = 4
    d_ff = 100
    n_layers = 3

    inputs = torch.rand(batch_size, seq_length, dim_embedding)


    encoder = Encoder(
        dim_embedding=dim_embedding,
        sequence_length=sequence_length,
        n_heads=n_heads,
        d_ff=d_ff,
        n=n_layers,
    )

    output = encoder(inputs)

    print(f"Inputs shape: {inputs.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        seq_length,
        dim_embedding,
    ), "Output shape is incorrect"

    print("Test passed!")


# if __name__ == "__main__":
#     test_encoder()
