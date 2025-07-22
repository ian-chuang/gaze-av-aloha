import os
import argparse
import torch
from torchvision import transforms
import torch
from mae import (
    MAE_ViT,
    MAE_Decoder,
    MAE_Encoder,
    BaseImageTokenizer,
    FoveatedImageTokenizer,
)
import kagglehub
from torchvision.datasets import ImageFolder
import einops
import imageio
from collections import OrderedDict


def load_checkpoint(path, model, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Check if keys are prefixed with "module."
    if list(state_dict.keys())[0].startswith("module."):
        # Remove "module." prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    epoch = checkpoint["epoch"]
    return epoch


if __name__ == "__main__":
    """

    VIEW TENSORBOARD:
    tensorboard --logdir logs/miniimagenet/mae-pretrain --port 6006

    python push_to_hub.py --type low_res_vit --checkpoint vit-b-mae_low_res_vit_1000.pth --repo_id iantc104/mae_vitb_low_res_vit
    python push_to_hub.py --type foveated_vit --checkpoint vit-b-mae_foveated_vit_1000.pth --repo_id iantc104/mae_vitb_foveated_vit_shift
    python push_to_hub.py --type foveated_vit --checkpoint vit-b-mae_no-noise_foveated_vit_1000.pth --repo_id iantc104/mae_vitb_foveated_vit

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--model_name", type=str, default="vit-b-mae")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--max_viz", type=int, default=3, help="Max number of images to visualize"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use for training (e.g., "cuda" or "cpu")',
    )
    parser.add_argument("--repo_id", type=str)
    args = parser.parse_args()

    if not args.checkpoint:
        model_path = f"{args.model_name}_{args.type}.pth"
    else:
        model_path = args.checkpoint

    if args.type == "foveated_vit":
        resize_shape = (288, 288)
        tokenizer = FoveatedImageTokenizer()
    elif args.type == "vit":
        resize_shape = (288, 288)
        tokenizer = BaseImageTokenizer()
    elif args.type == "low_res_vit":
        resize_shape = (216, 288)
        tokenizer = BaseImageTokenizer(token_size=64, height=256, width=320)
    else:
        raise ValueError(f"Unknown model type: {args.type}")

    transform_dataset = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                resize_shape, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    denormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    path = kagglehub.dataset_download("arjunashok33/miniimagenet")
    train_dataset = ImageFolder(path, transform=transform_dataset)
    dataloader = torch.utils.data.DataLoader(train_dataset, args.max_viz, shuffle=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    encoder = MAE_Encoder(
        num_tokens=tokenizer.get_num_tokens(),
        num_registers=1,
        patch_size=tokenizer.get_token_size(),
        depth=12,
        embedding_dim=768,
        num_heads=12,
        act_layer="gelu",
    )
    decoder = MAE_Decoder(
        num_tokens=tokenizer.get_num_tokens(),
        num_registers=1,
        patch_size=tokenizer.get_token_size(),
        depth=8,
        embedding_dim=384,
        num_heads=12,
    )
    model = MAE_ViT(
        tokenizer=tokenizer,
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to(device)
    model.eval()

    start_epoch = 0
    assert os.path.exists(model_path)
    print(f"Loading checkpoint from {model_path}")
    start_epoch = load_checkpoint(model_path, model, device=device)
    print(f"Loadded from epoch {start_epoch}")

    img, label = next(iter(dataloader))
    img = img.to(device)
    n = min(img.size(0), args.max_viz)
    img = img[:n]
    centers = torch.zeros(
        (img.size(0), 2), dtype=torch.float32, device=img.device
    )  # Center crop
    tokens, pred_tokens, mask = model(img[: args.max_viz], centers)
    mask = mask.to(torch.bool)
    viz = []
    token_size = tokenizer.get_token_size()
    viz_tokens = denormalize(
        einops.rearrange(
            tokens, "b n (c h w) -> b n c h w", c=3, h=token_size, w=token_size
        )
    ).clip(0, 1)
    viz_tokens_masked = denormalize(
        einops.rearrange(
            tokens * (~mask),
            "b n (c h w) -> b n c h w",
            c=3,
            h=token_size,
            w=token_size,
        )
    ).clip(0, 1)
    viz_pred_tokens = pred_tokens.detach()
    viz_pred_tokens[~mask] = tokens[~mask]  # Fill masked tokens with original tokens
    viz_pred_tokens = denormalize(
        einops.rearrange(
            viz_pred_tokens, "b n (c h w) -> b n c h w", c=3, h=token_size, w=token_size
        )
    ).clip(0, 1)
    for i in range(viz_tokens.size(0)):
        viz.append(
            torch.cat(
                [
                    tokenizer.generate_visualization(viz_tokens[i]),
                    tokenizer.generate_visualization(viz_tokens_masked[i]),
                    tokenizer.generate_visualization(viz_pred_tokens[i]),
                ],
                dim=2,
            )
        )
    viz = torch.cat(viz, dim=1)
    viz = viz.cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
    viz = (viz * 255).astype("uint8")
    imageio.imwrite(f"visualization_{args.type}.png", viz)
    print(f"Visualization saved to visualization_{args.type}.png")

    from gaze_av_aloha.policies.gaze_policy.vit import create_vit_b

    vit = create_vit_b(tokenizer.get_num_tokens(), tokenizer.get_token_size())
    vit = vit.to(device)
    vit.eval()

    def copy_matching_weights(src_module, tgt_module, startswith=()):
        src_state = dict(src_module.named_parameters())
        tgt_state = dict(tgt_module.named_parameters())

        for name, param in tgt_state.items():
            if any(name.startswith(prefix) for prefix in startswith):
                if name in src_state and src_state[name].shape == param.shape:
                    param.data.copy_(src_state[name].data)

    encoder = model.encoder
    copy_matching_weights(
        encoder, vit, startswith=["patch_emb", "pos_enc", "reg_tokens", "blocks"]
    )

    class DummyPatchShuffle:
        def forward(self, patches: torch.Tensor, masks: torch.Tensor = None):
            return patches, masks, None, None

    encoder.shuffle = DummyPatchShuffle()
    encoder.norm = torch.nn.Identity()

    x = torch.randn(
        32,
        tokenizer.get_num_tokens(),
        tokenizer.get_token_size() ** 2 * 3,
        device=img.device,
    )
    mask = torch.ones(
        (32, tokenizer.get_num_tokens()), dtype=torch.bool, device=img.device
    )
    encoder_feat, _ = encoder(x, mask)
    encoder_feat = encoder_feat.transpose(0, 1)  # (B, S, D)
    encoder_feat, encoder_reg = (
        encoder_feat[:, encoder.num_registers :],
        encoder_feat[:, : encoder.num_registers],
    )
    vit_feat, vit_reg = vit(x, mask)

    assert (
        encoder_feat.shape == vit_feat.shape
    ), f"Shape mismatch: {encoder_feat.shape} vs {vit_feat.shape}"
    assert (
        encoder_reg.shape == vit_reg.shape
    ), f"Shape mismatch: {encoder_reg.shape} vs {vit_reg.shape}"
    assert torch.allclose(
        encoder_feat, vit_feat, atol=1e-5
    ), "Encoder and ViT features do not match"
    assert torch.allclose(
        encoder_reg, vit_reg, atol=1e-5
    ), "Encoder and ViT registers do not match"

    vit.push_to_hub(
        repo_id=args.repo_id,
    )
