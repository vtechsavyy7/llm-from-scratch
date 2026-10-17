from torch import nn
import torch

from .ffn import FeedForwardLayer
from .multihead_self_attention import MultiHeadSelfAttention
from .layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim, dropout_prob):
        """
        Transformer block consisting of a multi-head self-attention layer
        followed by a feed-forward network. Skip connections are used around both sub-layers.
        Each sub-layer has a Pre-LayerNorm applied before it.
        Each sub-layer has a Dropout applied after it for regularization.

        Args:
            emb_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of the hidden layer in the feed-forward network.
            dropout_prob (float): Dropout probability for the sub-layers.
        """

        super().__init__()
        self.norm1 = LayerNorm(emb_dim)
        self.att = MultiHeadSelfAttention(emb_dim, num_heads, dropout_prob)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm2 = LayerNorm(emb_dim)
        self.ff = FeedForwardLayer(emb_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, causal_attn_mask: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).
            causal_attn_mask (torch.Tensor): Causal attention mask of shape (seq_len, seq_len)
                                      Elements with value True indicate positions that should be masked.
        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        # Pre-LayerNorm for the attention sub-layer
        normed_x = self.norm1(x)
        attn_output = self.att(normed_x, attn_mask=causal_attn_mask)
        x = x + self.dropout1(attn_output)

        # Pre-LayerNorm for the feed-forward sub-layer
        normed_x = self.norm2(x)
        ffn_output = self.ff(normed_x)
        x = x + self.dropout2(ffn_output)

        return x