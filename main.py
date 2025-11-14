import torch
import argparse
import random
import numpy as np
import os
from config import DATA_CONFIG, TRAIN_CONFIG, MODEL_CONFIG, ENSEMBLE_CONFIG, IMBALANCE_CONFIG, VIS_CONFIG
from models import get_model
from data_utils import get_dataloaders
from trainer import train_model
from visualization import run_visualization
from predict import Predictor, EnsemblePredictor

def set_seed(seed=TRAIN_CONFIG['seed']):
    """固定随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"已固定随机种子：{seed}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模型+多数据集+集成学习 图像分类框架')
    
    # 核心参数
    parser.add_argument('--dataset', type=str, choices=['catdog', 'cifar10'], default='catdog',
                        help='选择数据集（catdog/cifar10）')
    parser.add_argument('--model', type=str, choices=MODEL_CONFIG['supported_models'], default=MODEL_CONFIG['default_model'],
                        help='选择单模型训练（支持的模型：{}）'.format(','.join(MODEL_CONFIG['supported_models'])))
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'ensemble_train', 'ensemble_predict'], default='train',
                        help='运行模式：train（单模型训练）/ predict（单模型预测）/ ensemble_train（集成模型训练）/ ensemble_predict（集成预测）')
    
    # 可选参数
    parser.add_argument('--imbalance_strategy', type=str, choices=[None, 'weighted_loss', 'oversample'], default=IMBALANCE_CONFIG['strategy'],
                        help='数据不平衡处理策略（None/weighted_loss/oversample）')
    parser.add_argument('--vis', action='store_true', default=VIS_CONFIG['enable'],
                        help='训练后是否执行可视化分析')
    parser.add_argument('--vis_img', type=str, default=None,
                        help='用于特征图可视化的样本图像路径（如：dataset/test/cat.jpg）')
    parser.add_argument('--predict_path', type=str, default=None,
                        help='预测路径（单张图像路径或文件夹路径）')
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    # 固定随机种子
    set_seed()
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 更新配置（命令行参数优先级高于config.py）
    IMBALANCE_CONFIG['strategy'] = args.imbalance_strategy
    VIS_CONFIG['enable'] = args.vis
    
    # 1. 单模型训练模式
    if args.mode == 'train':
        # 获取数据加载器
        train_loader, val_loader, test_loader, class_names = get_dataloaders(args.dataset)
        # 获取模型
        model = get_model(
            model_name=args.model,
            num_classes=DATA_CONFIG[args.dataset]['num_classes'],
            pretrained=MODEL_CONFIG['pretrained'],
            freeze_features=MODEL_CONFIG['freeze_features']
        ).to(device)
        print(f"已初始化模型：{args.model}（类别数：{DATA_CONFIG[args.dataset]['num_classes']}）")
        
        # 训练模型
        model, train_history, val_history, best_model_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            dataset_type=args.dataset,
            train_dataset=train_loader.dataset,
            device=device,
            model_name=args.model
        )
        
        # 可视化分析（可选）
        if args.vis:
            run_visualization(
                model=model,
                train_history=train_history,
                val_history=val_history,
                test_loader=test_loader,
                class_names=class_names,
                dataset_type=args.dataset,
                model_name=args.model,
                device=device,
                feature_map_img_path=args.vis_img
            )
    
    # 2. 集成模型训练模式（训练所有集成模型）
    elif args.mode == 'ensemble_train':
        # 获取数据加载器（所有集成模型共用同一数据集）
        train_loader, val_loader, test_loader, class_names = get_dataloaders(args.dataset)
        num_classes = DATA_CONFIG[args.dataset]['num_classes']
        
        # 循环训练每个集成模型
        for model_name in ENSEMBLE_CONFIG['model_list']:
            print(f"\n" + "="*50)
            print(f"开始训练集成模型：{model_name}")
            print("="*50)
            
            # 初始化模型
            model = get_model(
                model_name=model_name,
                num_classes=num_classes,
                pretrained=MODEL_CONFIG['pretrained'],
                freeze_features=MODEL_CONFIG['freeze_features']
            ).to(device)
            
            # 训练模型
            model, train_history, val_history, best_model_path = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                dataset_type=args.dataset,
                train_dataset=train_loader.dataset,
                device=device,
                model_name=model_name
            )
            
            # 可视化分析（可选）
            if args.vis:
                run_visualization(
                    model=model,
                    train_history=train_history,
                    val_history=val_history,
                    test_loader=test_loader,
                    class_names=class_names,
                    dataset_type=args.dataset,
                    model_name=model_name,
                    device=device,
                    feature_map_img_path=args.vis_img
                )
    
    # 3. 单模型预测模式
    elif args.mode == 'predict':
        if not args.predict_path:
            raise ValueError("预测模式必须指定 --predict_path（单张图像或文件夹路径）")
        
        # 加载最佳模型路径
        model_path = os.path.join(DATA_CONFIG['model_save_dir'], f'best_{args.model}_{args.dataset}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重不存在：{model_path}（请先训练模型）")
        
        # 初始化预测器
        predictor = Predictor(model_path=model_path, dataset_type=args.dataset, device=device)
        
        # 执行预测
        if os.path.isfile(args.predict_path):
            # 单张图像预测
            result = predictor.predict_single(args.predict_path)
            print(f"\n预测结果：")
            print(f"图像：{result['filename']}")
            print(f"预测类别：{result['pred_class']}")
            print(f"置信度：{result['confidence']:.2f}%")
        elif os.path.isdir(args.predict_path):
            # 批量预测
            predictor.predict_batch(args.predict_path)
    
    # 4. 集成预测模式
    elif args.mode == 'ensemble_predict':
        if not args.predict_path:
            raise ValueError("集成预测模式必须指定 --predict_path（单张图像或文件夹路径）")
        
        # 初始化集成预测器
        ensemble_predictor = EnsemblePredictor(dataset_type=args.dataset, device=device)
        
        # 执行预测
        if os.path.isfile(args.predict_path):
            result = ensemble_predictor.predict_single(args.predict_path)
            print(f"\n集成预测结果：")
            print(f"图像：{result['filename']}")
            print(f"预测类别：{result['pred_class']}")
            print(f"置信度：{result['confidence']:.2f}%")
        elif os.path.isdir(args.predict_path):
            ensemble_predictor.predict_batch(args.predict_path)

if __name__ == '__main__':
    main()
