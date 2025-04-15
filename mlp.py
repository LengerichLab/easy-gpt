import torch.nn as nn

class FeedForward(nn.Module):
    """
    Two-layer feedforward network with configurable activation and dropout.
    Accepts a config object with attributes:
      - n_embd
      - dropout
      - bias
      - activation (optional: 'gelu' [default] or 'relu')
    """
    def __init__(self, config):
        super().__init__()
        act_fn = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
        }.get(getattr(config, 'activation', 'gelu').lower())

        if act_fn is None:
            raise ValueError(f"Unsupported activation: {getattr(config, 'activation', 'gelu')}")

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            act_fn,
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
