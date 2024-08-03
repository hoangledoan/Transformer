from torch import nn
import torch


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, dim_embedding: int, d_ff: int):
        """
        dff: dimension of hidden layer.
        """
        super().__init__()
        self.linear_1 = nn.Linear(dim_embedding, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, dim_embedding)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear_1(inputs)
        outputs = self.relu(outputs)
        outputs = self.linear_2(outputs)
        outputs = self.dropout(outputs)
        # Output shape: ([input], dim_embedding)
        return outputs
