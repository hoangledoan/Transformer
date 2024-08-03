from torch import nn
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim_embedding: int, max_length: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.dropout = nn.Dropout(p=0.5)
        self.dim_embedding = dim_embedding
        self.max_length = max_length

    def positional_encoding(self, dim_embedding: int, max_length: int):
        i = torch.arange(0, dim_embedding, 2) / dim_embedding
        pos = torch.arange(0, max_length)[:, None]

        angle_freq = torch.exp(i * (-torch.log(torch.Tensor([10000]))))

        output = torch.zeros((max_length, dim_embedding))

        output[:, 0::2] = torch.sin(pos * angle_freq)
        output[:, 1::2] = torch.cos(pos * angle_freq)

        return output
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input has shape (batch_size, sequence_length)
        sequence_length = input.shape[-1]
        embedding = self.embedding(input) * self.dim_embedding**1/2
        pos_encoding = self.positional_encoding(self.dim_embedding, self.max_length)
        # pos_encoding = nn.Parameter(data=pos_encoding, requires_grad = False)
        pos_encoding = pos_encoding[:sequence_length]
        output = embedding + pos_encoding
        output = self.dropout(output)
        return output


def test_embedding():
    vocab_size = 100  
    dim_embedding = 16          
    max_length = 50   
    batch_size = 4    
    sequence_length = 10  

    input_sequences = torch.randint(0, vocab_size, (batch_size, sequence_length))
    embedding_layer = Embedding(vocab_size, dim_embedding, max_length)
    output = embedding_layer(input_sequences)

    print(f"Input shape: {input_sequences.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, sequence_length, dim_embedding), "Output shape is incorrect"

    print("Test passed!")

# if __name__ == "__main__":
#     test_embedding()