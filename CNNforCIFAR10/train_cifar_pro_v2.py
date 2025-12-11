#!/usr/bin/env python3
"""
train_cifar_pro_v2.py

Improved CIFAR-10 training (CPU ONLY version):
- ResNet-18 改造版，适配 CIFAR-10
- 数据增强：RandomCrop + RandomHorizontalFlip (+ 可选 AutoAugment) + RandomErasing
- MixUp, label smoothing
- SGD + Nesterov + OneCycleLR / CosineAnnealingLR
- early stopping, best-checkpoint saving
- profiles:
    quick: 小数据 + 少轮次，用于快速检查流程
    full:  全量训练；在 CPU 上会自动调成「约 3 小时内 + 尽量高精度」
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights  # 为了兼容接口，实际不使用预训练

# ------------------------
# Utilities
# ------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    return res


# MixUp helpers
def mixup_data(x, y, alpha=1.0, device='cpu'):
    if alpha > 0:
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ------------------------
# Model (resnet18 adapted to CIFAR)
# ------------------------
def get_resnet18_cifar(num_classes=10):
    # use weights=None to avoid deprecation warning
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------
# Train / Eval
# ------------------------
def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    device,
    epoch,
    mixup_alpha=0.0,
    label_smoothing=0.0,
    grad_clip=None,
    scheduler=None,
    use_onecycle=False,
    print_freq=100,
):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for i, (images, targets) in enumerate(train_loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if mixup_alpha and mixup_alpha > 0.0:
            inputs, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha, device=device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if use_onecycle and scheduler is not None:
            scheduler.step()

        bs = targets.size(0)
        running_loss += loss.item() * bs
        total += bs
        acc1 = accuracy(outputs.detach(), targets, topk=(1,))[0]
        running_acc += acc1 * bs / 100.0

        if i % print_freq == 0 or i == len(train_loader):
            print(f"Epoch [{epoch}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f} Acc@1: {acc1:.2f}%")
    epoch_loss = running_loss / total
    epoch_acc = running_acc / total * 100.0
    return epoch_loss, epoch_acc


def evaluate(test_loader, model, criterion, device):
    model.eval()
    total = 0
    running_loss = 0.0
    correct1 = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            bs = targets.size(0)
            running_loss += loss.item() * bs
            total += bs
            _, preds = outputs.topk(1, 1, True, True)
            correct1 += preds.eq(targets.view(-1, 1)).sum().item()
    return running_loss / total, 100.0 * correct1 / total


# ------------------------
# Argparse
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./dataset/CIFAR10', type=str)
    p.add_argument('--batch-size', default=64, type=int)
    p.add_argument('--epochs', default=100, type=int)
    p.add_argument('--lr', default=0.1, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--weight-decay', default=5e-4, type=float)
    p.add_argument('--workers', default=0, type=int)
    p.add_argument('--mixup-alpha', default=0.8, type=float)
    p.add_argument('--label-smoothing', default=0.1, type=float)
    p.add_argument('--grad-clip', default=5.0, type=float)
    p.add_argument('--checkpoint-dir', default='./checkpointsProV2', type=str)
    p.add_argument('--device', default=None, type=str)  # 为了兼容命令行参数，但会被强制成 cpu
    p.add_argument('--print-freq', default=100, type=int)
    p.add_argument('--optimizer', default='sgd', choices=['sgd', 'adamw'])
    p.add_argument('--seed', default=42, type=int)
    p.add_argument(
        '--profile',
        default='full',
        choices=['quick', 'full'],
        help='quick: 小 subset + 少轮次; full: 正常训练(在 CPU 上自动调成约 3 小时配置)',
    )
    p.add_argument('--early-stop-patience', default=10, type=int)
    return p.parse_args()


# ------------------------
# Main
# ------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- 强制 CPU，忽略 CUDA 本地环境暂不支持gpu ----
    if args.device is not None and args.device.lower() != 'cpu':
        print(f"Requested device '{args.device}', "
              "but current GPU is incompatible with this PyTorch build. Forcing CPU instead.")
    device = 'cpu'
    print('Using device: cpu (CUDA disabled for this environment)')

    # 如果是 CPU + full profile，自动调参数，目标：约 3 小时内训练完成
    if args.profile == 'full':
        print("CPU-only full profile detected -> adjusting hyperparameters for ~3 hours window.")
        # 限制在 40 epoch 加快训练时间
        if args.epochs > 40:
            args.epochs = 40
        # 提高 batch_size 减少 iteration 次数（如果内存扛得住）
        if args.batch_size < 128:
            args.batch_size = 128
        # 适中强度的 MixUp 和 label smoothing
        args.mixup_alpha = 0.4
        if args.label_smoothing < 0.1:
            args.label_smoothing = 0.1
        # early stop 容忍度稍微放大一点
        if args.early_stop_patience < 8:
            args.early_stop_patience = 8

    # Transforms: 只考虑 CPU 的情况，适中，不要太重又有一定泛化能力
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    if args.profile == 'quick':
        # quick 模式下也用轻量增强即可
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.10),
        ])
    else:
        # full 模式：适中增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.15),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = Path(args.data_dir)
    train_folder = data_dir / 'train'
    test_folder = data_dir / 'test'
    if not (train_folder.exists() and test_folder.exists()):
        raise RuntimeError(f'Local train/test folders not found at {data_dir}.')

    train_dataset = datasets.ImageFolder(str(train_folder), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(test_folder), transform=test_transform)

    # profiles
    if args.profile == 'quick':
        print("Profile=quick: using small subset and fewer epochs for fast testing.")
        train_dataset = Subset(train_dataset, list(range(min(2000, len(train_dataset)))))
        test_dataset = Subset(test_dataset, list(range(min(500, len(test_dataset)))))
        args.epochs = min(args.epochs, 5)
        args.batch_size = min(args.batch_size, 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,  # 纯 CPU，用不上 pinned memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    num_classes = (
        len(train_dataset.dataset.classes)
        if isinstance(train_dataset, Subset)
        else len(train_dataset.classes)
    )
    model = get_resnet18_cifar(num_classes=num_classes).to(device)

    # loss (label smoothing)
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    except TypeError:
        criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    # OneCycleLR 优先; 如果出错就退到 CosineAnnealingLR
    try:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * steps_per_epoch,
        )
        use_onecycle = True
    except Exception:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_onecycle = False

    # early stopping
    best_acc = 0.0
    no_improve = 0
    best_ckpt = Path(args.checkpoint_dir) / 'best.pth'
    recent_val_acc = deque(maxlen=5)

    print(
        f"Starting training: epochs={args.epochs} batch_size={args.batch_size} "
        f"lr={args.lr} optimizer={args.optimizer}"
    )
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
            epoch,
            mixup_alpha=args.mixup_alpha,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            scheduler=scheduler,
            use_onecycle=use_onecycle,
            print_freq=args.print_freq,
        )
        val_loss, val_acc = evaluate(test_loader, model, criterion, device)
        if not use_onecycle:
            scheduler.step()

        recent_val_acc.append(val_acc)
        mean_recent = sum(recent_val_acc) / len(recent_val_acc)

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_acc': val_acc,
                },
                best_ckpt,
            )
            no_improve = 0
        else:
            no_improve += 1

        t1 = time.time()
        print(
            f"Epoch {epoch}/{args.epochs} Time: {t1-t0:.1f}s "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% "
            f"Best Acc: {best_acc:.2f}% mean_recent: {mean_recent:.2f}"
        )

        # early stopping on plateau of validation acc
        if no_improve >= args.early_stop_patience and epoch >= 10:
            print(f"No improvement for {no_improve} epochs -> early stopping.")
            break

    print('Training finished. Best Acc: {:.2f}%'.format(best_acc))
    print(f"Best checkpoint saved at: {best_ckpt}")


if __name__ == '__main__':
    main()
