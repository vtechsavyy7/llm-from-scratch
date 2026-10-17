from torch import nn
import torch

from .ffn import FeedForwardLayer
from .multihead_attention import MultiHeadAttention
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
        self.att = MultiHeadAttention(emb_dim, num_heads, dropout_prob)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.norm2 = LayerNorm(emb_dim)
        self.ff = FeedForwardLayer(emb_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, query: torch.Tensor, attn_mask: torch.Tensor = None, use_kv_cache: bool = False):
        """
        Forward pass for the Transformer block.

        Args:
            query (torch.Tensor): Input query tensor of shape (batch_size, seq_len, emb_dim).
            attn_mask (torch.Tensor): Attention mask of shape (query_len, key_len)
                                      Elements with value True indicate positions that should be masked.
            use_kv_cache (bool): Whether to use the key-value cache for faster decoding during inference.
        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        # Pre-LayerNorm for the attention sub-layer
        normed_query = self.norm1(query)
        attn_output = self.att(
            normed_query,
            normed_query,
            normed_query,
            attn_mask=attn_mask,
            use_kv_cache=use_kv_cache
        )
        # Apply dropout and add the residual connection for the attention sub-layer
        query = query + self.dropout1(attn_output)

        # Pre-LayerNorm for the feed-forward sub-layer
        normed_query = self.norm2(query)
        ffn_output = self.ff(normed_query)
        # Apply dropout and add the residual connection for the feed-forward sub-layer
        query = query + self.dropout2(ffn_output)

        return query