import torch
from torch import nn
from model.embedding import Embedding
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocal_size: int, eos_token_id: int, hparams: dict = None, sequence_max_length: int = 2048):
        super().__init__()
        """
        eos_token_id: id of the end-of-sentence token
        hparams: contains the input paramters, if not take default
        """
        self.vocal_size = vocal_size
        self.eos_token_id = eos_token_id
        self.sequence_max_length = sequence_max_length

        self.dim_embedding = hparams.get("dim_embedding", 512)
        self.sequence_length = hparams.get("sequence_length", 64)
        self.n_heads = hparams.get("n_heads", 8)
        self.d_ff = hparams.get("d_ff", 2048)
        self.n = hparams.get("n", 6)

        self.embedding = Embedding(self.vocal_size, self.dim_embedding, sequence_max_length)
        self.encoder = Encoder(self.dim_embedding, self.sequence_length, self.n_heads, self.d_ff, self.n)
        self.decoder = Decoder(self.dim_embedding, self.sequence_length, self.n_heads, self.d_ff, self.n)
        self.output_layer = nn.Linear(self.dim_embedding, self.vocal_size, bias=False)

        # Weight sharing as stated in the paper, the embedding weights that are used to encode token, are now used for generating output
        self.output_layer.weight = self.embedding.embedding.weight

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask = torch.Tensor) -> torch.Tensor:
        """
        Shape:
        - encoder_inputs: (batch_size, sequence_length_encoder)
        - decoder_inputs: (batch_size, sequence_length_decoder)
        - encoder_mask: (batch_size, sequence_length_encoder, sequence_length_encoder)
        - decoder_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
        - outputs: (batch_size, sequence_length_decoder, vocab_size)
        """
        encoder_embeddings = self.embedding(encoder_inputs)
        encoder_outputs = self.encoder(encoder_embeddings, encoder_mask)
        decoder_embeddings = self.embedding(decoder_inputs)
        decoder_outputs = self.decoder(decoder_embeddings, encoder_outputs, decoder_mask, encoder_mask)
        outputs = self.output_layer(decoder_outputs)
        return outputs
    
    def predict(self, encoder_input: torch.Tensor, ):
        """
        Shape:
        - encoder_input: (sequence_length, dim_embedding)
        """
        # Add batch
        encoder_input = encoder_input.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            encoder_input = self.embedding(encoder_input)
            encoder_output = self.encoder(encoder_input)
            output_sequence = []
            for _ in range(100):
                decoder_input = torch.Tensor([self.eos_token_id] + output_sequence).unsqueeze(0)
                decoder_input = self.embedding(decoder_input)
                output = self.decoder(decoder_input, encoder_output)
                logits = self.output_layer(output).squeeze(0)
                # logits has shape (sequence_length, vocab_size)
                # Extract the probability of the last token -> get the probability of the next token based on this!
                last_logit = logits[-1]
                # Next token is the one with highest probability 
                output = torch.argmax(last_logit).item()
                output_sequence.append(output)

                if output_sequence[-1] is self.eos_token_id:
                    break
        return output_sequence
