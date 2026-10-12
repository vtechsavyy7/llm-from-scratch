from torch import nn

class FeedForwardLayer(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        """
        FeedForward network with one hidden layer. Uses a
        GELU activation function and dropout for regularization.

        Args:
            emb_dim (int): Dimension of the input embeddings.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        return self.net(x)