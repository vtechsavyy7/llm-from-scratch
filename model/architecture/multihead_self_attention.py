from torch import nn
import torch


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout_prob):
        super().__init__()

        assert emb_dim % n_heads == 0

        self.n_heads = n_heads

        # Initialize the query, key and value linear layers
        # Note: The q,k and v dimensions for a single head are emb_dim // n_heads
        self.W_query = nn.Linear(emb_dim, emb_dim)
        self.W_key = nn.Linear(emb_dim, emb_dim)
        self.W_value = nn.Linear(emb_dim, emb_dim)

        # Final linear layer to combine the head outputs
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        # Dropout layer: 
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """Forward pass. 
        Note: Not adding a key_padding mask for this initial version

        Args: 
            x: torch.Tensor : Input tensor of shape [batch_size, seq_len, emb_dim]
            attn_mask: torch.Tensor : The causal attention mask of shape [seq_len, seq_len]. 
                                      Elements with value True indicate positions that should be masked.
        Returns:
            out_context: torch.Tensor : Output tensor of shape [batch_size, seq_len, emb_dim]
        """

        batch_size, seq_len, emb_dim = x.shape

        head_dim = emb_dim // self.n_heads

        # Project the input tensors onto the query, key and values space
        queries = self.W_query(x)    # [batch_size, seq_len, emb_dim]
        keys = self.W_key(x)
        values = self.W_value(x)

        # Separate out the individual heads 
        # [batch_size, seq_len, n_heads, head_dim]
        queries = queries.view(batch_size, seq_len, self.n_heads, -1)
        keys = keys.view(batch_size, seq_len, self.n_heads, -1)
        values = values.view(batch_size, seq_len, self.n_heads, -1)

        # Do a batched matrix multiplcation of queries and keys to get the raw attention weights

        # First transpose queries to be [batch_size, n_heads, seq_len, head_dim]
        queries = queries.transpose(1, 2)

        # Next permute keys to be [batch_size, n_heads, head_dim, seq_len]
        keys = keys.permute(0, 2, 3, 1)

        attn_scores = queries @ keys   # [batch_size, n_heads, seq_len(q), seq_len(k)]

        # Use the attn_mask to fill the attn_scores matrix with -torch.inf at the masking locations
        attn_scores.masked_fill_(attn_mask.bool(), -torch.inf)

        # Scale by the sqrt of head_dim and use the softmax to get normalized attention weights
        attn_weights = torch.softmax(attn_scores/ head_dim**0.5, dim=-1)  # [batch_size, n_heads, seq_len(q), seq_len(k)]

        # Apply dropout to the attention weights:
        attn_weights = self.dropout(attn_weights)

        # Multiply the attention weights with the values 
        # But first transpose the values to be of shape [batch_size, n_heads, seq_len(v), head_dim]
        values = values.transpose(1, 2)

        out_context = attn_weights @ values  # [batch_size, n_heads, seq_len(q), head_dim]

        out_context = out_context.transpose(1, 2)  # [batch_size, seq_len(q), n_heads, head_dim]

        # Reshape to combine all the attention heads
        out_context = out_context.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, emb_dim]

        # Apply the final linear projection: 
        out_context = self.out_proj(out_context)  # [batch_size, seq_len, emb_dim]

        return out_context



