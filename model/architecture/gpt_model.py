from torch import nn
import torch
from typing import Dict
from .transformer_block import TransformerBlock
from .layer_norm import LayerNorm
from .model_config import cfg_124m

class GPTModel(nn.Module):
    def __init__(self, config: Dict = cfg_124m): 
        super().__init__()

        # Create the input embedding layer
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        # Create the positional embedding layer
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        # Store the context length for use in the forward pass
        self.context_length = config["context_length"]

        # Create the transformer layers based on the configuration
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(config["emb_dim"], config["n_heads"], config["hidden_dim"], config["dropout_prob"])
            for _ in range(config["n_layers"])
        ])

        # Final layer normalization
        self.final_norm = LayerNorm(config["emb_dim"])

        # Output projection layer: 
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias = False)

    def forward(self, x, causal_attn_mask=None):
        """
        Forward pass for the GPT model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            causal_attn_mask (torch.Tensor, optional): Causal attention mask of shape (seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size) containing the logits for each token in the vocabulary.
        """
        # Get the batch size and sequence length
        batch_size, seq_len = x.shape

        # Get the token embeddings for the input sequence
        x = self.tok_emb(x)    # [batch_size, seq_len, emb_dim]

        # Create position indices and get the positional embeddings
        pos_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)  # [batch_size, seq_len]

        # Add the positional embeddings to the token embeddings
        x = x + self.pos_emb(pos_indices)   # [batch_size, seq_len, emb_dim]

        # Pass the embeddings through each transformer layer
        for layer in self.trf_blocks:
            x = layer(x, causal_attn_mask=causal_attn_mask)  # [batch_size, seq_len, emb_dim]

        # Apply final layer normalization and the output projection
        x = self.final_norm(x)   # [batch_size, seq_len, emb_dim]
        logits = self.out_head(x)   # [batch_size, seq_len, vocab_size]
        return logits