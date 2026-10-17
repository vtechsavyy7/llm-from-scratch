from torch import nn
import torch


class MultiHeadAttention(nn.Module):
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

        # Register buffers for KV cache
        # Set persistent=False, since we don't want the kv cache to 
        # be saved in the model's state_dict()
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_kv_cache(self):
        """Reset the key-value cache."""
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_kv_cache: bool = False
    ):
        """Compute multi-head attention over separate query, key, and value inputs.

        Args: 
            queries: Query input of shape [batch_size, query_len, emb_dim].
            keys: Key input of shape [batch_size, key_len, emb_dim].
            values: Value input of shape [batch_size, key_len, emb_dim].
            attn_mask: Optional boolean mask broadcastable to
                       [batch_size, n_heads, query_len, key_len]. True positions
                       are excluded from attention.
            use_kv_cache: Whether to use the key-value cache for faster decoding
                        during inference
        Returns:
            Output tensor of shape [batch_size, query_len, emb_dim].
        """

        batch_size, query_len, emb_dim = queries.shape
        key_len = keys.size(1)

        head_dim = emb_dim // self.n_heads

        # Project the input tensors onto the query, key and values space
        queries = self.W_query(queries)        # [batch_size, query_len, emb_dim]
        # Keys and Values to need be checked for the KV cache usage
        keys_new = self.W_key(keys)            # [batch_size, key_len, emb_dim]
        values_new = self.W_value(values)      # [batch_size, key_len, emb_dim]

        # Separate out the individual heads 
        # [batch_size, sequence_len, n_heads, head_dim]
        queries = queries.view(batch_size, query_len, self.n_heads, head_dim)
        keys_new = keys_new.view(batch_size, key_len, self.n_heads, head_dim)
        values_new = values_new.view(batch_size, key_len, self.n_heads, head_dim)

        # Check if KV cache needs to be used: 
        if use_kv_cache: 
            # Concat new keys and values with the cached keys and values
            if self.cache_k is not None: 
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            else: 
                self.cache_k = keys_new
                self.cache_v = values_new
            # Assign the cached items to the keys and values
            keys, values = self.cache_k, self.cache_v  # [batch_size, key_len, n_heads, head_dim]
        else: 
            keys, values = keys_new, values_new    # [batch_size, key_len, n_heads, head_dim]

        ## Do a batched matrix multiplcation of queries and keys to get the raw attention weights

        # First transpose queries to be [batch_size, n_heads, query_len, head_dim]
        queries = queries.transpose(1, 2)

        # Next permute keys to be [batch_size, n_heads, head_dim, key_len]
        keys = keys.permute(0, 2, 3, 1)

        attn_scores = queries @ keys   # [batch_size, n_heads, query_len, key_len]

        # Use the attn_mask to fill the attn_scores matrix with -torch.inf at the masking locations
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask.bool(), -torch.inf)

        # Scale by the sqrt of head_dim and use the softmax to get normalized attention weights
        attn_weights = torch.softmax(attn_scores/ head_dim**0.5, dim=-1)  # [batch_size, n_heads, query_len, key_len]

        # Apply dropout to the attention weights:
        attn_weights = self.dropout(attn_weights)

        # Multiply the attention weights with the values 
        # But first transpose the values to be of shape [batch_size, n_heads, key_len, head_dim]
        values = values.transpose(1, 2)

        out_context = attn_weights @ values  # [batch_size, n_heads, query_len, head_dim]

        out_context = out_context.transpose(1, 2)  # [batch_size, query_len, n_heads, head_dim]

        # Reshape to combine all the attention heads
        out_context = out_context.reshape(batch_size, query_len, -1)  # [batch_size, query_len, emb_dim]

        # Apply the final linear projection: 
        out_context = self.out_proj(out_context)  # [batch_size, query_len, emb_dim]

        return out_context
