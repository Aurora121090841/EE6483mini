# --------------------------
# 数据集配置
# --------------------------
DATA_CONFIG = {
    'cache_dir': './data_cache',  # 缓存下载的数据集
    'current_dataset': '',  # 运行时自动更新为当前使用的数据集
    'cifar10': {
        'num_classes': 10,
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck'],
        'class_to_idx': {},  # 运行时自动填充
        'normalize': {
            'mean': [0.4914, 0.4822, 0.4465],  
            'std': [0.2470, 0.2435, 0.2616]    
        }
    },
    'catdog': {
        'num_classes': 2,
        'class_names': ['cat', 'dog'],
        'class_to_idx': {},  # 运行时自动填充
        'huggingface_name': 'Aurora1609/cat_vs_dog',  # HuggingFace数据集名称
        'normalize': {
            'mean': [0.485, 0.456, 0.406],  # ImageNet均值（适配预训练模型）
            'std': [0.229, 0.224, 0.225]    # ImageNet标准差
        }
    }
}

# --------------------------
# 训练配置
# --------------------------
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,  # 数据加载线程数（根据CPU核心数调整）
    'num_epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'optimizer': 'adamw',  # 支持 'adam' / 'adamw' / 'sgd'
    'scheduler': 'cosine',  # 支持 'cosine' / 'step' / 'none'
    'step_size': 10,  # step scheduler的衰减步长
    'gamma': 0.1,  # step scheduler的衰减系数
    'early_stop_patience': 5,  # 早停耐心值（无提升则停止）
    # 'model_save_path': 'best_model.pth',  # 最佳模型保存路径
    'log_dir': './logs',  # 训练日志目录
    'model_save_path': './saved_models/best_model.pth',  # 带目录的路径
}

# --------------------------
# 类别不平衡配置
# --------------------------
IMBALANCE_CONFIG = {
    'enable': False,  # 总开关：False=不启用，True=启用
    'strategy': 'oversample',  # 可选策略：'oversample'（过采样）/ 'weighted_loss'（加权损失）
    'oversample_replace': True,  # 过采样是否允许重复采样（True=允许，False=不允许）
    'weighted_loss_factor': 1.0  # 加权损失系数（调整权重强度）
}

# --------------------------
# 模型配置
# --------------------------

MODEL_CONFIG = {
    'img_size': 224,  # 输入图像尺寸（catdog用224，cifar10用32可调整）
    'num_classes': 2,  # 默认类别数（运行时会根据数据集自动更新）
    'model_type': 'resnet18',  # 默认模型
    'pretrained': True,  # 是否使用预训练权重
    'dropout_rate': 0.5,  # Dropout概率
    'freeze_backbone': False,  # 是否冻结主干网络
    'freeze_features': False,  # 补充的配置项
    'supported_models': [  # 新增：支持的模型列表（和models.py对应）
        'resnet18', 'resnet34', 'resnet50', 
        'efficientnet', 'convnext','custom_cnn','vgg16'
    ]
}
