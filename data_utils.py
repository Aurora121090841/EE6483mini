from tkinter import Image
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset  # Add Dataset
from config import DATA_CONFIG, TRAIN_CONFIG, IMBALANCE_CONFIG, MODEL_CONFIG
import os
# --------------------------
# New: Custom Dataset for Unlabeled Test Set
# --------------------------
class UnlabeledImageDataset(Dataset):
    """Load unlabeled images from a flat folder (no class subfolders)"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Get all image paths
        self.image_paths = [
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        if not self.image_paths:
            raise ValueError(f"No images found in test folder: {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Return (image, dummy_label=0) — dummy label for compatibility
        return image, 0  # Dummy label (ignored in prediction/visualization)

    def get_image_filenames(self):
        """Return filenames of test images (for prediction results)"""
        return [os.path.basename(path) for path in self.image_paths]

# --------------------------
# Modified: get_transforms (unchanged, just ensure consistency)
# --------------------------
def get_transforms(dataset_type, normalize_mean, normalize_std, is_train=True):
    img_size = MODEL_CONFIG['img_size']
    
    if is_train:
        if dataset_type == 'catdog':
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
        elif dataset_type == 'cifar10':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
    else:
        if dataset_type == 'catdog':
            return transforms.Compose([
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
        elif dataset_type == 'cifar10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])

# --------------------------
# Modified: get_dataloaders (update test set loading)
# --------------------------
def get_dataloaders(dataset_type):
    data_cfg = DATA_CONFIG[dataset_type]
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    normalize_mean = data_cfg['normalize']['mean']
    normalize_std = data_cfg['normalize']['std']
    
    # --------------------------
    # 1. Load Labeled Train/Val Sets (unchanged)
    # --------------------------
    train_dataset = datasets.ImageFolder(
        root=data_cfg['train_dir'],
        transform=get_transforms(dataset_type, normalize_mean, normalize_std, is_train=True)
    )
    val_dataset = datasets.ImageFolder(
        root=data_cfg['val_dir'],
        transform=get_transforms(dataset_type, normalize_mean, normalize_std, is_train=False)
    )

    # Handle class imbalance for training set
    train_loader = None
    if IMBALANCE_CONFIG['strategy'] == 'oversample':
        class_counts = np.bincount(train_dataset.targets)
        total_samples = len(train_dataset)
        class_weights = total_samples / class_counts
        sample_weights = [class_weights[label] for label in train_dataset.targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=total_samples,
            replacement=IMBALANCE_CONFIG['oversample_replace']
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=torch.cuda.is_available()  # Use pin_memory only if GPU exists
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------
    # 2. Load Unlabeled Test Set (Modified!)
    # --------------------------
    test_transform = get_transforms(
        dataset_type=dataset_type,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        is_train=False
    )
    # Use custom UnlabeledImageDataset instead of ImageFolder
    test_dataset = UnlabeledImageDataset(
        root=data_cfg['test_dir'],
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Print dataset info
    print(f"=== {dataset_type} Dataset Info ===")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    print(f"Num classes: {num_classes}, Class names: {class_names}")
    print(f"Test folder structure: Flat (unlabeled) — {len(test_dataset)} images found")

    return train_loader, val_loader, test_loader, class_names, test_dataset  # Add test_dataset return