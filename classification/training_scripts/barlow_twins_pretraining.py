from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms

from timm.scheduler import CosineLRScheduler

from models import build_vmamba
# (other custom/model imports as needed...)

def parse_args():
    parser = argparse.ArgumentParser(description='Barlow Twins Pretraining with Vision Mamba')
    parser.add_argument('data', type=Path, metavar='DIR', help='Path to dataset')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to config file')
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='Number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='Total number of epochs to run')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='Mini-batch size')
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR', help='Base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR', help='Base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W', help='Weight decay')
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L', help='Weight on off-diagonal terms')
    parser.add_argument('--projector', default='2048-2048-2048', type=str, metavar='MLP', help='Projector MLP')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='Print frequency')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path, metavar='DIR', help='Path to checkpoint directory')
    return parser.parse_args()

def setup_distributed(args):
    """Configures distributed training, including SLURM handling if present."""
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        cmd = f'scontrol show hostnames {os.getenv("SLURM_JOB_NODELIST")}'
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node

def main():
    args = parse_args()
    print(f"Configuration:\n{vars(args)}")
    setup_distributed(args)
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
    else:
        stats_file = None

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=1e-6,
        warmup_t=10,
        warmup_lr_init=1e-6,
        t_in_epochs=True
    )

    start_epoch = 0
    ckpt_path = args.checkpoint_dir / 'checkpoint.pth'
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if args.rank == 0:
            print(f"Resuming from checkpoint at epoch {start_epoch}")
    dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    if args.rank == 0:
        print(f"Loaded dataset from {args.data / 'train'}, {len(dataset)} images found.")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0, "Batch size must be divisible by world size."
    per_device_batch_size = args.batch_size // args.world_size

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=False, sampler=sampler
    )

    scaler = torch.cuda.amp.GradScaler()
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(train_loader, start=epoch * len(train_loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + step / len(train_loader))
            if step % args.print_freq == 0 and args.rank == 0:
                stats = dict(epoch=epoch, step=step,
                             lr=optimizer.param_groups[0]['lr'],
                             loss=loss.item(),
                             time=int(time.time() - start_time))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # Save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict())
            torch.save(state, ckpt_path)

    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.checkpoint_dir / 'vmamba_final.pth')
        if stats_file:
            stats_file.close()

# Handlers
def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    sys.exit()

def handle_sigterm(signum, frame):
    pass

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# --- Model and Transform Classes ---

class BarlowTwins(nn.Module):
    """Barlow Twins implementation with Vision Mamba backbone."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = build_vmamba()
        self.backbone.fc = nn.Identity()
        sizes = [1024] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1], bias=False),
                nn.BatchNorm1d(sizes[i + 1]),
                nn.ReLU(inplace=True)
            ]
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.args.lambd * off_diag

class GaussianBlur:
    def __init__(self, p): self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img

class Solarization:
    def __init__(self, p): self.p = p
    def __call__(self, img):
        if random.random() < self.p: return ImageOps.solarize(img)
        return img

class Transform:
    """Data augmentation for self-supervised pretraining (two views)."""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(512, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.8
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8710, 0.6560, 0.7565], std=[0.1524, 0.2428, 0.1715])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(512, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.8
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8710, 0.6560, 0.7565], std=[0.1524, 0.2428, 0.1715])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform_prime(x)

if __name__ == '__main__':
    main()
