'''
Author: Aurora
Date: 2025-11-14 00:48:26
Descripttion: 
LastEditTime: 2025-11-14 11:29:23
'''
import torch
import torch.nn as nn
from torchvision import models

class CatDogCNN(nn.Module):
    """原自定义CNN模型"""
    def __init__(self, num_classes=2):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def get_model(model_name, num_classes=2, pretrained=False):
    """
    模型选择接口
    :param model_name: 模型名称，支持 'custom_cnn'、'vgg16'、'resnet50'
    :param num_classes: 分类类别数
    :param pretrained: 是否使用预训练权重
    :return: 构建好的模型
    """
    if model_name == 'custom_cnn':
        model = CatDogCNN(num_classes=num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        # 替换最后一层全连接层以适应目标类别数
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # 替换最后一层全连接层
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"不支持的模型: {model_name}，可选值：'custom_cnn'、'vgg16'、'resnet50'")
    
    return model

# 新增模型定义（可放在CatDogCNN类之后）
from torchvision import models

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ConvNeXtModel, self).__init__()
        # 加载预训练的ConvNeXt（小型版本）
        self.base_model = models.convnext_small(pretrained=pretrained)
        # 替换最后一层全连接层（适配二分类）
        in_features = self.base_model.classifier[2].in_features
        self.base_model.classifier[2] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetModel, self).__init__()
        # 加载预训练的EfficientNet（B0版本）
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        # 替换最后一层全连接层
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)
    


import torch
import torch.nn as nn
from torchvision import models
from config import MODEL_CONFIG

class CustomCNN(nn.Module):
    """自定义轻量CNN（适配2/10分类）"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        # 卷积特征提取器
        self.conv_layers = nn.Sequential(
            #  Conv Block 1: 3→32
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


def get_model(model_name, num_classes=2, pretrained=MODEL_CONFIG['pretrained'], freeze_features=MODEL_CONFIG['freeze_features']):
    """
    统一模型获取接口：根据名称返回适配后的模型
    :param model_name: 模型名称（参考MODEL_CONFIG['supported_models']）
    :param num_classes: 分类类别数（自动根据数据集调整）
    :param pretrained: 是否使用预训练权重
    :param freeze_features: 是否冻结特征层
    :return: 适配后的模型
    """
    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes)
    
    # VGG16（迁移学习）
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        # 替换最后一层全连接层
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    # ResNet50（迁移学习）
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    # ConvNeXt（迁移学习）
    elif model_name == 'convnext':
        model = models.convnext_small(pretrained=pretrained)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    
    # EfficientNet B0（迁移学习）
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"不支持的模型：{model_name}，可选：{MODEL_CONFIG['supported_models']}")
    
    # 冻结特征层（仅训练分类头）
    if freeze_features and model_name != 'custom_cnn':  # 自定义CNN不冻结
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后一层分类器
        if model_name in ['vgg16', 'resnet50']:
            for param in model.classifier.parameters() if model_name == 'vgg16' else [model.fc]:
                param.requires_grad = True
        elif model_name == 'convnext':
            for param in model.classifier[2].parameters():
                param.requires_grad = True
        elif model_name == 'efficientnet':
            for param in model.classifier[1].parameters():
                param.requires_grad = True
    
    return model