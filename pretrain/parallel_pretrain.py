import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision import transforms
from tqdm import tqdm
import random
import torch
import numpy as np
from mae import MAE_ViT, MAE_Decoder, MAE_Encoder, BaseImageTokenizer, FoveatedImageTokenizer
import kagglehub
from torchvision.datasets import ImageFolder
import einops
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size, port=12355):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def main_worker(rank, world_size, args):
    setup_seed(args.seed)
    setup_distributed(rank, world_size, args.port)
    print(f"[Rank {rank}] Using CUDA device: {torch.cuda.current_device()} / real GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # params
    device = torch.device(f'cuda:{rank}')
    model_path = f"{args.model_name}_{args.type}.pth"

    print(f"Using model path: {model_path}")

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # tokenizer
    if args.type == "foveated_vit":
        resize_shape = (288,288)
        tokenizer = FoveatedImageTokenizer()
    elif args.type == "vit":
        resize_shape = (288,288)
        tokenizer = BaseImageTokenizer()
    elif args.type == "low_res_vit":
        resize_shape = (216,288)
        tokenizer = BaseImageTokenizer(token_size=64, height=256, width=320)
    else:
        raise ValueError(f"Unknown model type: {args.type}")

    # dataset
    transform_dataset = transforms.Compose([
        transforms.RandomResizedCrop(resize_shape, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    denormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    path = kagglehub.dataset_download("arjunashok33/miniimagenet")
    train_dataset = ImageFolder(path, transform=transform_dataset)
    # dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=load_batch_size,
        shuffle=False,  # must be False when using DistributedSampler
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    # model
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
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # optimizer and scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    # logger
    if rank == 0:
        writer = SummaryWriter(os.path.join('logs', 'miniimagenet', model_path))

    # reload model if exists
    start_epoch = 0
    if os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        start_epoch = load_checkpoint(model_path, model, optim, lr_scheduler, device=device)
        print(f"Resuming from epoch {start_epoch}")

    step_count = 0
    optim.zero_grad()
    for e in range(start_epoch, args.total_epoch):
        train_sampler.set_epoch(e)
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            if args.add_noise:
                centers = torch.randn((img.size(0), 2), dtype=torch.float32, device=img.device) * 0.2
            else:
                centers = torch.zeros((img.size(0), 2), dtype=torch.float32, device=img.device)  # Center crop
            tokens, pred_tokens, mask = model(img, centers)
            loss = torch.mean((pred_tokens - tokens) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()

        if rank == 0:
            avg_loss = sum(losses) / len(losses)
            writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')

            model.eval()
            with torch.no_grad():
                img, label = next(iter(dataloader))
                img = img.to(device)
                n = min(img.size(0), args.max_viz)
                img = img[:n]
                centers = torch.zeros((img.size(0), 2), dtype=torch.float32, device=img.device)  # Center crop
                tokens, pred_tokens, mask = model(img[:args.max_viz], centers)
                mask = mask.to(torch.bool)
                viz = []
                token_size = tokenizer.get_token_size()
                viz_tokens = denormalize(einops.rearrange(tokens, "b n (c h w) -> b n c h w", c=3, h=token_size, w=token_size)).clip(0, 1)
                viz_tokens_masked = denormalize(einops.rearrange(tokens * (~mask), "b n (c h w) -> b n c h w", c=3, h=token_size, w=token_size)).clip(0, 1)
                viz_pred_tokens = pred_tokens.detach()
                viz_pred_tokens[~mask] = tokens[~mask]  # Fill masked tokens with original tokens
                viz_pred_tokens = denormalize(einops.rearrange(viz_pred_tokens, "b n (c h w) -> b n c h w", c=3, h=token_size, w=token_size)).clip(0, 1)
                for i in range(viz_tokens.size(0)):
                    viz.append(
                        torch.cat([
                            tokenizer.generate_visualization(viz_tokens[i]), 
                            tokenizer.generate_visualization(viz_tokens_masked[i]),
                            tokenizer.generate_visualization(viz_pred_tokens[i])
                        ], dim=2)
                    )
                viz = torch.cat(viz, dim=1)
                writer.add_image('mae_image', viz, global_step=e)
        
            save_checkpoint(model, optim, lr_scheduler, e + 1, model_path)
            if e+1 == 1000:
                save_checkpoint(model, optim, lr_scheduler, e + 1, model_path.replace('.pth', '_1000.pth'))


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--type', type=str)
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the input images')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_name', type=str, default='vit-b-mae')
    parser.add_argument('--max_viz' , type=int, default=3, help='Max number of images to visualize')
    parser.add_argument('--port', type=int, default=12355, help='Port for distributed training')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        # Single GPU — no multiprocessing needed
        main_worker(rank=0, world_size=1, args=args)

if __name__ == "__main__":
    main()

    