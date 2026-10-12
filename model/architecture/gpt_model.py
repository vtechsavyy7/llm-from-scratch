from torch import nn
import torch
from typing import Dict
from .transformer_block import TransformerBlock
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
            TransformerBlock(config["emb_dim"], config["n_heads"], config["hidden_dim"], config["dropout_prob"], config["qkv_bias"])
            for _ in range(config["n_layers"])
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(config["emb_dim"])

        # Output projection layer: 
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias = False)

    def load_hf_gpt2_weights(self, hf_model):
        """Copy the weights from a Hugging Face GPT-2 checkpoint into this custom GPT model."""
        sd_hf = hf_model.state_dict()
        sd_custom = self.state_dict()
        transpose_keys = {
            "trf_blocks.0.attn.in_proj_weight",
            "trf_blocks.0.attn.out_proj.weight",
            "trf_blocks.0.ffn.net.0.weight",
            "trf_blocks.0.ffn.net.2.weight",
        }

        for layer_idx in range(len(self.trf_blocks)):
            prefix = f"trf_blocks.{layer_idx}"
            transpose_keys.update({
                f"{prefix}.attn.in_proj_weight",
                f"{prefix}.attn.out_proj.weight",
                f"{prefix}.ffn.net.0.weight",
                f"{prefix}.ffn.net.2.weight",
            })

        mapping = {
            "tok_emb.weight": "transformer.wte.weight",
            "pos_emb.weight": "transformer.wpe.weight",
            "final_norm.weight": "transformer.ln_f.weight",
            "final_norm.bias": "transformer.ln_f.bias",
            "out_head.weight": "lm_head.weight",
        }

        for layer_idx in range(len(self.trf_blocks)):
            layer_prefix = f"trf_blocks.{layer_idx}"
            layering = {
                f"{layer_prefix}.norm1.weight": f"transformer.h.{layer_idx}.ln_1.weight",
                f"{layer_prefix}.norm1.bias": f"transformer.h.{layer_idx}.ln_1.bias",
                f"{layer_prefix}.attn.in_proj_weight": f"transformer.h.{layer_idx}.attn.c_attn.weight",
                f"{layer_prefix}.attn.in_proj_bias": f"transformer.h.{layer_idx}.attn.c_attn.bias",
                f"{layer_prefix}.attn.out_proj.weight": f"transformer.h.{layer_idx}.attn.c_proj.weight",
                f"{layer_prefix}.attn.out_proj.bias": f"transformer.h.{layer_idx}.attn.c_proj.bias",
                f"{layer_prefix}.norm2.weight": f"transformer.h.{layer_idx}.ln_2.weight",
                f"{layer_prefix}.norm2.bias": f"transformer.h.{layer_idx}.ln_2.bias",
                f"{layer_prefix}.ffn.net.0.weight": f"transformer.h.{layer_idx}.mlp.c_fc.weight",
                f"{layer_prefix}.ffn.net.0.bias": f"transformer.h.{layer_idx}.mlp.c_fc.bias",
                f"{layer_prefix}.ffn.net.2.weight": f"transformer.h.{layer_idx}.mlp.c_proj.weight",
                f"{layer_prefix}.ffn.net.2.bias": f"transformer.h.{layer_idx}.mlp.c_proj.bias",
            }
            mapping.update(layering)

        with torch.no_grad():
            for custom_key, hf_key in mapping.items():
                if custom_key not in sd_custom or hf_key not in sd_hf:
                    continue
                value = sd_hf[hf_key]
                if custom_key in transpose_keys:
                    value = value.t()
                sd_custom[custom_key].copy_(value)

        return self

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