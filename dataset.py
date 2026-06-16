# dataset.py
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64, use_augmentation=False):
    """
    Builds data loaders for clean training, clean testing, and corrupted testing.
    """
    # 1. Base transform for clean images (Source Domain)
    clean_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales pixels to [-1, 1]
    ])

    # 2. Defended transform (Used only if use_augmentation=True during training)
    if use_augmentation:
        train_transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
            # Randomly apply a Gaussian Blur 50% of the time during training
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        train_transform = clean_transform

    # 3. Corrupted transform for the Stress Test (Target Domain)
    # We apply static Gaussian Blur and reduce brightness via ColorJitter
    corrupted_transform = T.Compose([
        T.ToTensor(),
        T.GaussianBlur(kernel_size=5, sigma=2.0),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4. Download and load the actual CIFAR-10 datasets
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_clean_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=clean_transform)
    test_corrupt_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=corrupted_transform)

    # 5. Create DataLoaders to yield 4D batches (Batch x Channels x Height x Width)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_clean_loader = DataLoader(test_clean_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_corrupt_loader = DataLoader(test_corrupt_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_clean_loader, test_corrupt_loader

if __name__ == '__main__':
    # Quick sanity check execution
    tl, tcl, tcorl = get_data_loaders(batch_size=64)
    print(f"Data Engine Ready! Train batches: {len(tl)}, Clean Test batches: {len(tcl)}, Corrupted Test batches: {len(tcorl)}")