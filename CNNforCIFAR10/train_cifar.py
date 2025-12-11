#!/usr/bin/env python3
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

# ------------------------
# Utilities
# ------------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
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

def get_resnet18_cifar(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(train_loader, model, criterion, optimizer, device, torch_module, epoch, grad_clip=None, print_freq=100):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip:
            torch_module.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

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
    p.add_argument('--batch-size', default=32, type=int)
    p.add_argument('--epochs', default=3, type=int)
    p.add_argument('--lr', default=0.001, type=float)
    p.add_argument('--checkpoint-dir', default='./checkpoints', type=str)
    p.add_argument('--device', default=None, type=str)
    p.add_argument('--grad-clip', default=5.0, type=float)
    return p.parse_args()

def main():
    args = parse_args()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    set_seed(42)

    # Data transforms
    normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Model
    model = get_resnet18_cifar(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, device, torch, epoch, grad_clip=args.grad_clip)
        val_loss, val_acc = evaluate(test_loader, model, criterion, device)

        # Save checkpoint
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(),
                    'best_acc': val_acc, 'optimizer': optimizer.state_dict()},
                   Path(args.checkpoint_dir)/f'ckpt_ep{epoch+1}.pth')

        t1 = time.time()
        print(f'Epoch {epoch}/{args.epochs} Time: {t1-t0:.1f}s Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% '
              f'Best Acc: {best_acc:.2f}%')
        best_acc = max(best_acc, val_acc)

    print('Training finished. Best Acc: {:.2f}%'.format(best_acc))

if __name__ == '__main__':
    main()
