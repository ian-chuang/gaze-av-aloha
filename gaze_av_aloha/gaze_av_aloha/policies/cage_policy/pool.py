import torch
import torch.nn as nn

from gaze_av_aloha.policies.cage_policy.blocks import TransformerBlock

class CAGEPooling(nn.Module):
    def __init__(self, dim, obs_dim, num_queries=1, layers=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim))
        self.obs_norm = nn.LayerNorm(obs_dim)
        self.x_attn = TransformerBlock(dim, cond_dim=obs_dim, dropout=dropout, cross_attn_only=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(layers)
        ])

    def forward(self, obs_emb):
        B, L, D = obs_emb.shape
        obs_emb = self.obs_norm(obs_emb)

        # mask = torch.ones(T, T, dtype=torch.bool).tril()
        # mask = mask.unsqueeze(0).expand(B, T, T).to(device=obs_emb.device)
        # cond_mask = mask.reshape(B, T, T, 1).repeat(1, 1, N, L).reshape(B, T, N*T*L)

        latents = self.latents.expand(B, -1, -1)
        latents = self.x_attn(latents, obs_emb) #cond_mask=cond_mask)
        
        for block in self.blocks:
            latents = block(latents) #mask=mask)

        return latents
    
class AttentionPooling(nn.Module):
    def __init__(self, dim, obs_dim, num_queries=1, layers=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim))
        self.obs_norm = nn.LayerNorm(obs_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, cond_dim=obs_dim, num_heads=num_heads, dropout=dropout) for _ in range(layers)
        ])

    def forward(self, obs_emb):
        B = obs_emb.shape[0]
        latents = self.latents.expand(B, -1, -1)

        obs_emb = self.obs_norm(obs_emb)
        for block in self.blocks:
            latents = block(latents, obs_emb) #mask=mask)

        return latents