'''
Author: Aurora
Date: 2025-11-14 00:48:26
Descripttion: 统一模型定义与获取接口
LastEditTime: 2025-11-14 12:02:03
'''
import torch
import torch.nn as nn
from torchvision import models
from config import MODEL_CONFIG  # 仅导入一次

# --------------------------
# 1. 自定义CNN模型
# --------------------------
class CustomCNN(nn.Module):
    """自定义轻量CNN（适配2/10分类，输入尺寸224x224）"""
    def __init__(self, num_classes=2, dropout_rate=MODEL_CONFIG['dropout_rate']):
        super().__init__()
        # 卷积特征提取器
        self.conv_layers = nn.Sequential(
            # Conv Block 1: 3→32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Conv Block 2: 32→64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Conv Block 3: 64→128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            # Conv Block 4: 128→256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # 全连接分类器（适配224输入：224/(2^4)=14 → 256*14*14）
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --------------------------
# 2. 迁移学习模型
# --------------------------
def get_model(model_name, num_classes=2, pretrained=MODEL_CONFIG['pretrained'], freeze_features=MODEL_CONFIG['freeze_features']):
    """
    统一模型获取接口：支持自定义CNN和多种迁移学习模型
    :param model_name: 模型名称（必须在 MODEL_CONFIG['supported_models'] 中）
    :param num_classes: 分类类别数（自动根据数据集调整）
    :param pretrained: 是否使用预训练权重
    :param freeze_features: 是否冻结特征层（仅迁移学习模型有效）
    :return: 适配后的模型
    """
    model = None
    model_name = model_name.lower()  # 统一小写，避免大小写冲突

    # 1. 自定义CNN（无预训练，不冻结）
    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes)

    # 2. ResNet系列
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        # 替换最后一层全连接层
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    # 3. VGG16
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    # 4. ConvNeXt（小型版本）
    elif model_name == 'convnext':
        model = models.convnext_small(pretrained=pretrained)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    # 5. EfficientNet B0
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # 不支持的模型
    else:
        supported = MODEL_CONFIG['supported_models']
        raise ValueError(f"不支持的模型：{model_name}，可选：{supported}")

    # 冻结特征层（仅迁移学习模型，自定义CNN不冻结）
    if freeze_features and model_name != 'custom_cnn':
        # 冻结主干网络所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后一层分类器（保证能训练）
        if model_name in ['resnet18', 'resnet50']:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == 'vgg16':
            for param in model.classifier[-1].parameters():
                param.requires_grad = True
        elif model_name == 'convnext':
            for param in model.classifier[2].parameters():
                param.requires_grad = True
        elif model_name == 'efficientnet':
            for param in model.classifier[1].parameters():
                param.requires_grad = True

    return model
