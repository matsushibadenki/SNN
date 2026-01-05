
from snn_research.models.transformer.spikformer import Spikformer
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Spikformer on CIFAR-100')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar10', 'imagenet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--T', type=int, default=4,
                        help='Simulation time steps')
    parser.add_argument('--img_size', type=int, default=32,
                        help='Input image size')  # CIFAR default
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str,
                        default='./results/spikformer_cifar100')
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()


def get_dataloader(args):
    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        raise NotImplementedError(
            "ImageNet not yet fully supported in this script due to path complexity.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, num_classes


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Args: {args}")

    train_loader, test_loader, num_classes = get_dataloader(args)

    model = Spikformer(
        img_size_h=args.img_size,
        img_size_w=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        T=args.T,
        num_classes=num_classes
    ).to(args.device)

    print(
        f"Model created. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # (B, num_classes)
            loss = criterion(outputs, targets)

            # SpikingJelly Functional Reset
            # Note: inside model.forward(), reset_net(self) is called.
            # But if we were running manually, we would need it.
            # Ideally, reset should happen BEFORE forward or AFTER.
            # Current model design resets INSIDE forward.

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        scheduler.step()

        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(
                    args.device), targets.to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total

        print(f"Epoch {epoch} Done. "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
              f"Time: {time.time() - start_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or test_acc > 70.0:  # Save if decent or periodic
            save_path = os.path.join(
                args.save_dir, f'spikformer_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_test_loss,
                'acc': test_acc
            }, save_path)
            print(f"Saved checkpoint to {save_path}")


if __name__ == '__main__':
    main()
