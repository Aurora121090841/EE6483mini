from PIL import Image as PILImage  # 彻底替换tkinter的Image，避免冲突
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from datasets import load_dataset  # HuggingFace数据集加载
from config import DATA_CONFIG, TRAIN_CONFIG, IMBALANCE_CONFIG, MODEL_CONFIG
import os

# --------------------------
# 1. 通用HuggingFace数据集封装类
# --------------------------
class HuggingFaceDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 兼容不同数据集的图像存储格式（PIL Image 或 numpy array）
        if isinstance(item['image'], np.ndarray):
            image = PILImage.fromarray(item['image'])
        else:
            image = item['image'].convert("RGB")  # 统一转为RGB通道
        
        # 兼容不同数据集的标签字段（'label' 或 'labels'）
        label = item['labels'] if 'labels' in item else item['label']
        # 确保标签是整数（部分数据集可能是字符串，需映射）
        if isinstance(label, str):
            label = DATA_CONFIG[DATA_CONFIG['current_dataset']]['class_to_idx'][label]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------------------
# 2. 数据预处理转换函数
# --------------------------
def get_transforms(dataset_type, normalize_mean, normalize_std, is_train=True):
    img_size = MODEL_CONFIG['img_size']
    
    if is_train:
        if dataset_type == 'cifar10':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
        elif dataset_type == 'catdog':
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
    else:
        if dataset_type == 'cifar10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
        elif dataset_type == 'catdog':
            return transforms.Compose([
                transforms.Resize(int(img_size * 1.14)),  # 为中心裁剪预留空间
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])

# --------------------------
# 3. 在线数据集加载函数
# --------------------------
def get_online_datasets(dataset_type):
    if dataset_type == 'cifar10':
        # 从torchvision在线下载CIFAR10（自动缓存到cache_dir）
        train_dataset = datasets.CIFAR10(
            root=DATA_CONFIG['cache_dir'],
            train=True,
            download=True,
            transform=None
        )
        # CIFAR10无单独test集，用train=False作为val集，再拆分出test集
        val_test = datasets.CIFAR10(
            root=DATA_CONFIG['cache_dir'],
            train=False,
            download=True,
            transform=None
        )
        # 拆分val和test（8:2）
        val_size = int(len(val_test) * 0.8)
        val_dataset = torch.utils.data.Subset(val_test, range(val_size))
        test_dataset = torch.utils.data.Subset(val_test, range(val_size, len(val_test)))
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = 10
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
    elif dataset_type == 'catdog':
        # 从HuggingFace在线加载Aurora1609/cat_vs_dog
        dataset = load_dataset(DATA_CONFIG['catdog']['huggingface_name'])
        
        # 处理数据集分割
        if 'train' in dataset and 'validation' in dataset and 'test' in dataset:
            train_dataset = dataset['train']
            val_dataset = dataset['validation']
            test_dataset = dataset['test']
        elif 'train' in dataset and 'test' in dataset:
            # 从train拆分val（20%）
            train_val = dataset['train'].train_test_split(test_size=0.2, seed=42)
            train_dataset = train_val['train']
            val_dataset = train_val['test']
            test_dataset = dataset['test']
        else:
            # 只有train分割，拆分val（20%）和test（10%）
            full_train = dataset['train']
            train_val = full_train.train_test_split(test_size=0.3, seed=42)
            train_dataset = train_val['train']
            val_test = train_val['test'].train_test_split(test_size=1/3, seed=42)  # 0.3中的1/3作为test
            val_dataset = val_test['train']
            test_dataset = val_test['test']
        
        # 转换为自定义Dataset（统一接口）
        train_dataset = HuggingFaceDataset(train_dataset)
        val_dataset = HuggingFaceDataset(val_dataset)
        test_dataset = HuggingFaceDataset(test_dataset)
        
        class_names = ['cat', 'dog']
        num_classes = 2
        class_to_idx = {'cat': 0, 'dog': 1}
        
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    return train_dataset, val_dataset, test_dataset, class_names, num_classes, class_to_idx

# --------------------------
# 4. 类别权重计算函数
# --------------------------
def get_class_weights(dataset_type, train_dataset):
    """计算类别权重（用于加权损失）"""
    # 提取所有标签
    if dataset_type == 'cifar10':
        targets = train_dataset.targets
    else:  # catdog
        targets = [item[1] for item in train_dataset]
    
    class_counts = np.bincount(targets)
    total_samples = len(train_dataset)
    class_weights = total_samples / (len(class_counts) * class_counts)  # 均衡权重公式
    return torch.tensor(class_weights, dtype=torch.float32)

# --------------------------
# 5. 主数据加载函数（支持类别不平衡开关）
# --------------------------
def get_dataloaders(dataset_type):
    # 创建缓存目录（存储下载的数据集，避免重复下载）
    os.makedirs(DATA_CONFIG.get('cache_dir', './data_cache'), exist_ok=True)
    
    # 加载在线数据集（纯在线，无本地文件）
    train_dataset, val_dataset, test_dataset, class_names, num_classes, class_to_idx = get_online_datasets(dataset_type)
    
    # 更新配置（方便其他模块获取类别信息）
    DATA_CONFIG[dataset_type]['class_to_idx'] = class_to_idx
    DATA_CONFIG['current_dataset'] = dataset_type
    
    # 获取标准化参数（优先用配置，无则用默认）
    data_cfg = DATA_CONFIG[dataset_type] if dataset_type in DATA_CONFIG else {}
    normalize_mean = data_cfg.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    normalize_std = data_cfg.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
    
    # 获取数据转换 pipeline
    train_transform = get_transforms(dataset_type, normalize_mean, normalize_std, is_train=True)
    val_transform = get_transforms(dataset_type, normalize_mean, normalize_std, is_train=False)
    
    # 应用转换
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    test_dataset.transform = val_transform
    
    # --------------------------
    # 类别不平衡处理
    # --------------------------
    train_loader = None
    if IMBALANCE_CONFIG['enable']:  # 总开关：是否启用不平衡处理
        if IMBALANCE_CONFIG['strategy'] == 'oversample':
            # 提取标签计算权重
            if dataset_type == 'cifar10':
                targets = train_dataset.targets
            else:  # catdog
                targets = [item[1] for item in train_dataset]
            
            class_counts = np.bincount(targets)
            total_samples = len(train_dataset)
            class_weights = total_samples / class_counts
            sample_weights = [class_weights[label] for label in targets]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=total_samples,
                replacement=IMBALANCE_CONFIG['oversample_replace']
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAIN_CONFIG['batch_size'],
                sampler=sampler,
                num_workers=TRAIN_CONFIG.get('num_workers', 4),
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )
            print(f"启用类别不平衡处理（过采样策略），类别分布：{class_counts}")
        else:
            # 其他策略（如weighted_loss）：这里只创建普通DataLoader，损失函数在trainer中处理
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAIN_CONFIG['batch_size'],
                shuffle=True,
                num_workers=TRAIN_CONFIG.get('num_workers', 4),
                pin_memory=torch.cuda.is_available(),
                drop_last=True
            )
            print(f"启用类别不平衡处理（{IMBALANCE_CONFIG['strategy']}策略）")
    else:
        # 不启用类别不平衡处理：普通DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True,
            num_workers=TRAIN_CONFIG.get('num_workers', 4),
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        print("未启用类别不平衡处理")
    
    # 验证集/测试集加载器（固定配置）
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=TRAIN_CONFIG.get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=TRAIN_CONFIG.get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )

    # 打印数据集信息
    print(f"\n=== {dataset_type} 数据集信息 ===")
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")
    print(f"类别数: {num_classes}, 类别名称: {class_names}")
    print(f"标准化参数 - 均值: {normalize_mean}, 标准差: {normalize_std}")

    return train_loader, val_loader, test_loader, class_names, class_to_idx
