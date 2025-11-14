import torch
import argparse
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, IMBALANCE_CONFIG
from data_utils import get_dataloaders
from trainer import train_model, test_model
from models import get_model

# --------------------------
# 增强命令行参数（支持数据集、模型、不平衡处理的组合）
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='图像分类（支持数据集×模型×类别不平衡组合）')
    
    # 1. 数据集参数（二选一）
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'catdog'], 
                        default=DATA_CONFIG['current_dataset'] or 'catdog',
                        help='数据集类型（默认：catdog）')
    
    # 2. 模型参数（支持models.py中所有模型）
    parser.add_argument('--model_type', type=str, 
                        choices=MODEL_CONFIG['supported_models'],
                        default=MODEL_CONFIG['model_type'],
                        help=f'模型类型（默认：{MODEL_CONFIG["model_type"]}，可选：{MODEL_CONFIG["supported_models"]}）')
    
    # 3. 类别不平衡参数（新增，支持三种选择）
    parser.add_argument('--imbalance_strategy', type=str, 
                        choices=['none', 'oversample', 'weighted_loss'],
                        default='none',
                        help='类别不平衡处理策略（默认：none=不启用，可选：oversample/weighted_loss）')
    
    # 4. 可选：是否使用预训练权重（新增，灵活控制迁移学习）
    parser.add_argument('--pretrained', action='store_true', 
                        default=MODEL_CONFIG['pretrained'],
                        help='是否使用预训练权重（默认：%(default)s）')
    
    return parser.parse_args()

# --------------------------
# 动态配置更新（根据命令行参数覆盖默认配置）
# --------------------------
def update_config(args):
    # 更新数据集配置（自动适配输入尺寸）
    if args.dataset == 'cifar10':
        MODEL_CONFIG['img_size'] = 32  # CIFAR10是32x32
        MODEL_CONFIG['num_classes'] = DATA_CONFIG['cifar10']['num_classes']
    elif args.dataset == 'catdog':
        MODEL_CONFIG['img_size'] = 224  # CatDog适配224x224
        MODEL_CONFIG['num_classes'] = DATA_CONFIG['catdog']['num_classes']
    DATA_CONFIG['current_dataset'] = args.dataset
    
    # 更新模型配置
    MODEL_CONFIG['model_type'] = args.model_type
    MODEL_CONFIG['pretrained'] = args.pretrained
    
    # 更新类别不平衡配置
    if args.imbalance_strategy == 'none':
        IMBALANCE_CONFIG['enable'] = False
        IMBALANCE_CONFIG['strategy'] = 'none'
    else:
        IMBALANCE_CONFIG['enable'] = True
        IMBALANCE_CONFIG['strategy'] = args.imbalance_strategy

# --------------------------
# 主函数（保持原有流程，适配动态配置）
# --------------------------
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')

    # 1. 动态更新配置（命令行参数优先级最高）
    update_config(args)
    print(f'\n当前配置组合：')
    print(f'  数据集：{args.dataset}（输入尺寸：{MODEL_CONFIG["img_size"]}x{MODEL_CONFIG["img_size"]}）')
    print(f'  模型：{args.model_type}（预训练：{MODEL_CONFIG["pretrained"]}）')
    print(f'  类别不平衡处理：{"不启用" if args.imbalance_strategy == "none" else f"启用{args.imbalance_strategy}"}')

    # 2. 获取数据加载器（纯在线，无本地依赖）
    train_loader, val_loader, test_loader, class_names, class_to_idx = get_dataloaders(args.dataset)

    # 3. 获取模型（根据命令行参数选择）
    model = get_model(
        model_name=MODEL_CONFIG['model_type'],
        num_classes=MODEL_CONFIG['num_classes'],
        pretrained=MODEL_CONFIG['pretrained'],
        freeze_features=MODEL_CONFIG['freeze_features']
    ).to(device)
    print(f'\n模型初始化完成：{MODEL_CONFIG["model_type"]}（类别数：{MODEL_CONFIG["num_classes"]}）')

    # 4. 训练模型（自动适配不平衡处理策略）
    train_model(model, train_loader, val_loader, class_names, device)

    # 5. 测试模型（加载最佳权重）
    checkpoint = torch.load(TRAIN_CONFIG['model_save_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'\n加载最佳模型（验证准确率：{checkpoint["best_val_acc"]:.2f}%）')
    test_model(model, test_loader, class_names, device)

if __name__ == '__main__':
    main()
