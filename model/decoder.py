import torch
from torch import nn
from model.multi_head_attention import MultiHeadAttention
from model.feedforward import FeedForwardNeuralNetwork


class DecoderBlock(nn.Module):
    def __init__(self, dim_embedding: int, d_k: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_multi_head = MultiHeadAttention(dim_embedding, d_k, n_heads)
        self.cross_multi_head = MultiHeadAttention(dim_embedding, d_k, n_heads)
        self.ffn = FeedForwardNeuralNetwork(dim_embedding, d_ff)
        self.bn = nn.LayerNorm(dim_embedding)

    def forward(self, inputs: torch.Tensor, encoder_context: torch.Tensor, causual_mask: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:   
        # Self-attention with mask, letting the decoder to understand its previous generated token
        x  = self.self_multi_head(inputs, inputs, inputs, causual_mask) + inputs
        x = self.bn(x)

        # Cross-attention with input from the encoder, giving context to the decoder
        # Padding mask to ensure all sentence have the same length
        print(encoder_context.shape)
        x = self.cross_multi_head(x, encoder_context, encoder_context, pad_mask) + x
        x = self.bn(x)

        # Feed forward 
        x = self.ffn(x) + x
        x = self.bn(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dim_embedding: int, d_k: int, n_heads: int, d_ff: int, n: int, ):
        super().__init__()
        self.stack = nn.ModuleList([
            DecoderBlock(dim_embedding, d_k, n_heads, d_ff)
         for _ in range(n)])

    def create_causual_mask(self, batch_size, decoder_sequence_length):
        output = torch.ones((decoder_sequence_length, decoder_sequence_length))
        output = torch.tril(output, diagonal=0).bool()
        return output.unsqueeze_(0).expand(batch_size, -1, -1)
    
    def forward(self, inputs: torch.Tensor, encoder_context: torch.Tensor, decoder_padding_mask: torch.Tensor= None, encoder_padding_mask: torch.Tensor= None) -> torch.Tensor:
        # Input is decoder input, has shape: (batch, sequence_length, dim_embedding)
        batch_size, seq_length, _ = inputs.shape
        casual_mask = self.create_causual_mask(batch_size, seq_length) 
        if decoder_padding_mask is not None:
            casual_mask = casual_mask * decoder_padding_mask
        
        outputs = inputs
        for decoder in self.stack:
            outputs = decoder(outputs, encoder_context, casual_mask, encoder_padding_mask)
        return outputs
            

    
def test_decoder():
    batch_size = 2
    seq_length = 5
    dim_embedding = 40
    d_k = 10
    n_heads = 4
    d_ff = 100
    n_layers = 3

    inputs = torch.rand(batch_size, seq_length, dim_embedding)
    encoder_context = torch.rand(batch_size, seq_length, dim_embedding)
    decoder_padding_mask = (torch.rand(batch_size, 1, seq_length) > 0.5).bool()
    encoder_padding_mask = (torch.rand(batch_size, 1, seq_length) > 0.5).bool()

    decoder = Decoder(
        dim_embedding=dim_embedding,
        d_k=d_k,
        n_heads=n_heads,
        d_ff=d_ff,
        n=n_layers,
    )

    output = decoder(inputs, encoder_context, decoder_padding_mask, encoder_padding_mask)

    print(f"Inputs shape: {inputs.shape}")
    print(f"Encoder context shape: {encoder_context.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        seq_length,
        dim_embedding,
    ), "Output shape is incorrect"

    print("Test passed!")


if __name__ == "__main__":
    test_decoder()