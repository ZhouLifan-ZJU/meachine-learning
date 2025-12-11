#!/usr/bin/env python3
"""
Resilient CIFAR-10 training script with updated ResNet18 initialization to remove deprecated warnings.

Features:
- Works on CPU/GPU
- MixUp support, gradient clipping, OneCycleLR/ CosineAnnealingLR scheduler
- Checkpoint saving
- Uses ResNet18 backbone modified for CIFAR
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import ResNet18_Weights  # 新导入

# ------------------------
# Utilities
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b,)

# ------------------------
# Model
# ------------------------
def get_resnet18_cifar(models_module, torch_module, num_classes=10):
    # 使用 weights=None 替代 deprecated pretrained 参数
    model = models_module.resnet18(weights=None)
    # 替换 conv1 和 maxpool 以适应 CIFAR
    model.conv1 = torch_module.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch_module.nn.Identity()
    model.fc = torch_module.nn.Linear(model.fc.in_features, num_classes)
    return model

# ------------------------
# Training / Evaluation
# ------------------------
def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, mixup_alpha=0.0, print_freq=100, grad_clip=None, scheduler=None, use_onecycle=False):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if mixup_alpha > 0:
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

        if (i + 1) % print_freq == 0 or (i+1) == len(train_loader):
            print(f"Epoch [{epoch}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc@1: {acc1:.2f}%")
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
            _, preds = outputs.topk(1,1,True,True)
            correct1 += preds.eq(targets.view(-1,1)).sum().item()
    return running_loss/total, 100.0 * correct1 / total

# ------------------------
# Main
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='./dataset/CIFAR10', type=str)
    p.add_argument('--batch-size', default=64, type=int)
    p.add_argument('--epochs', default=50, type=int)
    p.add_argument('--lr', default=0.01, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--weight-decay', default=5e-4, type=float)
    p.add_argument('--workers', default=0, type=int)
    p.add_argument('--mixup-alpha', default=0.2, type=float)
    p.add_argument('--checkpoint-dir', default='./checkpoints', type=str)
    p.add_argument('--device', default=None, type=str)
    p.add_argument('--print-freq', default=100, type=int)
    p.add_argument('--grad-clip', default=5.0, type=float)
    p.add_argument('--optimizer', default='sgd', choices=['sgd','adamw'])
    p.add_argument('--seed', default=42, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    set_seed(args.seed)

    # ------------------------
    # Data transforms
    # ------------------------
    normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = Path(args.data_dir)
    train_folder = data_dir / 'train'
    test_folder = data_dir / 'test'
    if train_folder.exists() and test_folder.exists():
        print(f'Found local train/test folders at {data_dir}. Using ImageFolder loader.')
        train_dataset = datasets.ImageFolder(str(train_folder), transform=train_transform)
        test_dataset = datasets.ImageFolder(str(test_folder), transform=test_transform)
        num_classes = len(train_dataset.classes)
    else:
        raise RuntimeError(f'Local train/test folders not found at {data_dir}.')

    pin_memory = True if (device == 'cuda') else False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=args.workers, pin_memory=pin_memory)

    # ------------------------
    # Model, loss, optimizer
    # ------------------------
    model = get_resnet18_cifar(models, torch, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------------------
    # Scheduler
    # ------------------------
    steps_per_epoch = max(1, len(train_loader))
    try:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs*steps_per_epoch)
        use_onecycle = True
    except Exception:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        use_onecycle = False

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer,
                                                device, epoch, mixup_alpha=args.mixup_alpha,
                                                print_freq=args.print_freq, grad_clip=args.grad_clip,
                                                scheduler=scheduler, use_onecycle=use_onecycle)
        val_loss, val_acc = evaluate(test_loader, model, criterion, device)
        if not use_onecycle:
            scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        # Save checkpoint
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(),
                    'best_acc': best_acc, 'optimizer': optimizer.state_dict()},
                   Path(args.checkpoint_dir)/f'ckpt_ep{epoch+1}.pth')

        t1 = time.time()
        print(f'Epoch {epoch}/{args.epochs} Time: {t1-t0:.1f}s Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% '
              f'Best Acc: {best_acc:.2f}%')

    print('Training finished. Best Acc: {:.2f}%'.format(best_acc))

if __name__ == '__main__':
    main()
