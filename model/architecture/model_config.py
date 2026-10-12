from typing import  Dict

cfg_124m : Dict = {
    "vocab_size": 50257, 
    "context_length": 1024, 
    "emb_dim": 768,
    "hidden_dim": 3072,
    "n_layers": 12,
    "n_heads": 12,
    "dropout_prob": 0.1,
    "learning_rate": 0.001,
    "qkv_bias": True
}