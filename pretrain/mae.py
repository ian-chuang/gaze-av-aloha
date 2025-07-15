import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle():
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes
    
from gaze_av_aloha.policies.gaze_policy.vit import named_apply, init_weights_vit_timm, extend_valid_token_mask, get_activation_fn
from gaze_av_aloha.policies.gaze_policy.vit import Block as ViTBlock
from typing import Optional, Type, Callable
import torch

class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_registers: int,
        patch_size: int,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        act_layer: str,
        drop: float = 0.0,
        drop_path: float = 0.0,
        mask_ratio: float = 0.75
    ):
        super().__init__()

        assert num_registers == 1, "assume only using CLS token"

        act_layer = get_activation_fn(act_layer)

        self.patch_emb = torch.nn.Linear(patch_size * patch_size * 3, embedding_dim)

        self.pos_enc = torch.nn.Parameter(torch.randn(num_tokens, embedding_dim))

        self.reg_tokens = torch.nn.Parameter(torch.randn(num_registers, embedding_dim))

        self.shuffle = PatchShuffle(mask_ratio)

        self.blocks = torch.nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                ViTBlock(
                    dim=embedding_dim,
                    num_heads=num_heads,
                    act_layer=act_layer,
                    drop=drop,
                    drop_path=drop_path,
                )
            )

        self.norm = torch.nn.LayerNorm(embedding_dim)

        self.init_weights()

    @property
    def embed_dim(self) -> int:
        return self.patch_emb.out_features

    def init_weights(self):
        torch.nn.init.trunc_normal_(self.pos_enc, std=0.02)
        torch.nn.init.normal_(self.reg_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def forward(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters:
        - foveated_tokens (B, N, E): The input foveated image tokens.
        - valid_token_mask (B, N): An optional mask indicating which tokens are valid.

        Returns:
        - features (B, N, C): The output feature vectors, one per input token
        - register_features (B, N, C): The register feature vectors, one per register token
        """
        # normal
        tokens = self.patch_emb(
            tokens.flatten(2)
        ) + self.pos_enc.unsqueeze(0) # (B, S, D)

        # not normal
        tokens = tokens.transpose(0,1) # (S, B, D)
        tokens, forward_indexes, backward_indexes = self.shuffle.forward(tokens)
        tokens = tokens.transpose(0,1) # (B, S, D)

        # normal 
        num_registers = self.reg_tokens.shape[0]
        tokens = torch.cat(
            [
                tokens,
                self.reg_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1),
            ],
            dim=1,
        )
        valid_token_mask = extend_valid_token_mask(mask, num_registers)
        invalid_token_mask = (
            valid_token_mask.logical_not() if valid_token_mask is not None else None
        )
        for block in self.blocks:
            tokens = block(tokens, invalid_token_mask)
        # tokens = self.norm(tokens) 

        # not normal
        tokens = tokens.transpose(0,1) # (S, B, D)

        return tokens, backward_indexes
    
from gaze_av_aloha.policies.gaze_policy.tokenizer import BaseImageTokenizer, FoveatedImageTokenizer, ImageTokenizer

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)

        return patches, mask   


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 tokenizer: ImageTokenizer,
                 encoder: MAE_Encoder,
                 decoder: MAE_Decoder, 
                 ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img):
        tokens, mask = self.tokenizer.tokenize(img)
        features, backward_indexes = self.encoder(tokens, mask)
        pred_tokens, mask = self.decoder(features,  backward_indexes)
        return tokens.flatten(2).transpose(0,1), pred_tokens, mask 

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits
