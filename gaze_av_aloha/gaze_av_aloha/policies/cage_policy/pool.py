import torch
from torch import nn, Tensor
from timm.layers import Mlp

def maybe_add_pos_embed(tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
    return tensor if pos_embed is None else tensor + pos_embed

class AttentionPoolingBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.xattn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=dropout)

    def forward(self, x, c, x_pos_emb=None, c_pos_emb=None):
        x_norm = self.norm1(x)
        q = k = maybe_add_pos_embed(x_norm, x_pos_emb)
        v = x_norm
        x = x + self.attn(q, k, v, need_weights=False)[0]

        q = maybe_add_pos_embed(self.norm2(x), x_pos_emb)
        k = maybe_add_pos_embed(c, c_pos_emb)
        v = c
        x = x + self.xattn(q, k, v, need_weights=False)[0]

        x = x + self.mlp(self.norm3(x))

        return x

class AttentionPooling(nn.Module):
    """
    Attention pooling layer that uses DiT as the backbone.
    """
    def __init__(self, 
        num_queries, 
        hidden_size=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.query_tokens = nn.Embedding(num_queries, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([
            AttentionPoolingBlock(
                hidden_size=hidden_size,
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(depth)
        ])
        self.norm2 = nn.LayerNorm(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        self.apply(_basic_init)

    def forward(self, c, c_pos_emb=None):
        """
        context: (B, N, D) - Input features, where B is batch size, N is sequence length, and D is feature dimension.
        """
        B = c.size(0)
        x = self.query_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        c = self.norm1(c)
        for block in self.blocks:
            x = block(x=x, c=c, c_pos_emb=c_pos_emb)
        x = self.norm2(x)
        return x 
    