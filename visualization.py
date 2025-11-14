# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# import os
# from PIL import Image
# from config import DATA_CONFIG, VIS_CONFIG
# from data_utils import get_transforms

# # 全局可视化配置
# plt.rcParams['font.size'] = 12
# plt.rcParams['figure.figsize'] = (12, 8)
# plt.rcParams['axes.grid'] = True
# os.makedirs(DATA_CONFIG['vis_save_dir'], exist_ok=True)

# def plot_training_curves(train_history, val_history, dataset_type, model_name):
#     """绘制训练/验证损失和准确率曲线"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
#     # 损失曲线
#     ax1.plot(train_history['loss'], label='Train Loss', color='blue', linewidth=2)
#     ax1.plot(val_history['loss'], label='Val Loss', color='red', linewidth=2)
#     ax1.set_title(f'Training & Validation Loss ({model_name} - {dataset_type})')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()
    
#     # 准确率曲线
#     ax2.plot(train_history['acc'], label='Train Acc', color='blue', linewidth=2)
#     ax2.plot(val_history['acc'], label='Val Acc', color='red', linewidth=2)
#     ax2.set_title(f'Training & Validation Accuracy ({model_name} - {dataset_type})')
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('Accuracy (%)')
#     ax2.legend()
    
#     save_path = os.path.join(DATA_CONFIG['vis_save_dir'], f'training_curves_{model_name}_{dataset_type}.png')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"训练曲线已保存：{save_path}")
#     plt.close()

# def plot_confusion_matrix(model, test_loader, class_names, dataset_type, model_name, device):
#     """绘制混淆矩阵（测试集）"""
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device)
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.numpy())
    
#     # 生成混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
    
#     # 绘制热力图
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         cm, annot=True, fmt='d', cmap='Blues',
#         xticklabels=class_names,
#         yticklabels=class_names
#     )
#     plt.title(f'Confusion Matrix ({model_name} - {dataset_type})')
#     plt.xlabel('Predicted Class')
#     plt.ylabel('True Class')
    
#     save_path = os.path.join(DATA_CONFIG['vis_save_dir'], f'confusion_matrix_{model_name}_{dataset_type}.png')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"混淆矩阵已保存：{save_path}")
#     plt.close()

# def visualize_feature_maps(model, image_path, dataset_type, model_name, device):
#     """可视化中间卷积层特征图"""
#     # 1. 预处理图像
#     data_cfg = DATA_CONFIG[dataset_type]
#     transform = get_transforms(
#         dataset_type=dataset_type,
#         normalize_mean=data_cfg['normalize']['mean'],
#         normalize_std=data_cfg['normalize']['std'],
#         is_train=False
#     )
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0).to(device)
    
#     # 2. 注册钩子函数获取特征图
#     feature_maps = {}
#     def hook_fn(name):
#         def hook(module, input, output):
#             feature_maps[name] = output.detach()
#         return hook
    
#     # 为指定层注册钩子（仅支持自定义CNN的conv1/conv2）
#     hooks = []
#     if model_name == 'custom_cnn':
#         for layer_name in VIS_CONFIG['feature_layers']:
#             layer = getattr(model.conv_layers, layer_name, None)
#             if layer:
#                 hooks.append(layer.register_forward_hook(hook_fn(layer_name)))
    
#     # 3. 前向传播获取特征图
#     model.eval()
#     with torch.no_grad():
#         model(image_tensor)
    
#     # 4. 移除钩子
#     for hook in hooks:
#         hook.remove()
    
#     # 5. 可视化特征图（每个层显示前16个通道）
#     for layer_name, fm in feature_maps.items():
#         fm = fm.squeeze(0)  # (C, H, W)
#         num_channels = fm.size(0)
#         display_channels = min(16, num_channels)
        
#         fig, axes = plt.subplots(4, 4, figsize=(16, 16))
#         fig.suptitle(f'Feature Maps - {model_name} - {layer_name}', fontsize=16)
        
#         for i in range(display_channels):
#             ax = axes[i//4, i%4]
#             ax.imshow(fm[i].cpu().numpy(), cmap='viridis')
#             ax.axis('off')
#             ax.set_title(f'Channel {i+1}')
        
#         # 隐藏未使用的子图
#         for i in range(display_channels, 16):
#             axes[i//4, i%4].axis('off')
        
#         save_path = os.path.join(DATA_CONFIG['vis_save_dir'], f'feature_maps_{model_name}_{layer_name}_{dataset_type}.png')
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"特征图已保存：{save_path}")
#         plt.close()

# def analyze_test_samples(model, test_loader, class_names, dataset_type, model_name, device):
#     """分析测试集正确/错误分类样本"""
#     model.eval()
#     correct_samples = []
#     incorrect_samples = []
    
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images = images.to(device)
#             labels = labels.to(device)
            
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#             probs = torch.softmax(outputs, dim=1)
            
#             for img, label, pred, prob in zip(images, labels, preds, probs):
#                 if label == pred:
#                     correct_samples.append({
#                         'img': img.cpu(),
#                         'true_class': class_names[label.item()],
#                         'pred_class': class_names[pred.item()],
#                         'confidence': prob[pred.item()].item() * 100
#                     })
#                 else:
#                     incorrect_samples.append({
#                         'img': img.cpu(),
#                         'true_class': class_names[label.item()],
#                         'pred_class': class_names[pred.item()],
#                         'confidence': prob[pred.item()].item() * 100
#                     })
    
#     # 可视化样本（各显示5个）
#     num_samples = VIS_CONFIG['sample_analysis_num']
#     fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))
#     fig.suptitle(f'Test Sample Analysis ({model_name} - {dataset_type})', fontsize=16)
    
#     # 正确分类样本
#     axes[0, 0].set_ylabel('Correct Predictions', fontsize=14)
#     for i in range(min(num_samples, len(correct_samples))):
#         sample = correct_samples[i]
#         ax = axes[0, i]
#         # 反归一化：(C, H, W) → (H, W, C)
#         img_np = sample['img'].permute(1, 2, 0).numpy()
#         img_np = img_np * np.array(DATA_CONFIG[dataset_type]['normalize']['std']) + np.array(DATA_CONFIG[dataset_type]['normalize']['mean'])
#         img_np = np.clip(img_np, 0, 1)
        
#         ax.imshow(img_np)
#         ax.set_title(f"True: {sample['true_class']}\nPred: {sample['pred_class']} ({sample['confidence']:.1f}%)")
#         ax.axis('off')
    
#     # 错误分类样本
#     axes[1, 0].set_ylabel('Incorrect Predictions', fontsize=14)
#     for i in range(min(num_samples, len(incorrect_samples))):
#         sample = incorrect_samples[i]
#         ax = axes[1, i]
#         img_np = sample['img'].permute(1, 2, 0).numpy()
#         img_np = img_np * np.array(DATA_CONFIG[dataset_type]['normalize']['std']) + np.array(DATA_CONFIG[dataset_type]['normalize']['mean'])
#         img_np = np.clip(img_np, 0, 1)
        
#         ax.imshow(img_np)
#         ax.set_title(f"True: {sample['true_class']}\nPred: {sample['pred_class']} ({sample['confidence']:.1f}%)")
#         ax.axis('off')
    
#     save_path = os.path.join(DATA_CONFIG['vis_save_dir'], f'sample_analysis_{model_name}_{dataset_type}.png')
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"样本分析图已保存：{save_path}")
#     plt.close()

# def run_visualization(model, train_history, val_history, test_loader, class_names, dataset_type, model_name, device, feature_map_img_path=None):
#     """统一调用所有可视化功能"""
#     if not VIS_CONFIG['enable']:
#         print("可视化功能已禁用")
#         return
    
#     print("\n=== 开始可视化分析 ===")
#     # 1. 训练曲线
#     plot_training_curves(train_history, val_history, dataset_type, model_name)
#     # 2. 混淆矩阵
#     plot_confusion_matrix(model, test_loader, class_names, dataset_type, model_name, device)
#     # 3. 特征图（可选）
#     if feature_map_img_path and os.path.exists(feature_map_img_path):
#         visualize_feature_maps(model, feature_map_img_path, dataset_type, model_name, device)
#     # 4. 样本分析
#     analyze_test_samples(model, test_loader, class_names, dataset_type, model_name, device)
#     print("=== 可视化分析完成 ===")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from PIL import Image
from config import DATA_CONFIG, VIS_CONFIG

plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
os.makedirs(DATA_CONFIG['vis_save_dir'], exist_ok=True)

# --------------------------
# Unchanged: plot_training_curves (no labels needed)
# --------------------------
def plot_training_curves(train_history, val_history, dataset_type, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(train_history['loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(val_history['loss'], label='Val Loss', color='red', linewidth=2)
    ax1.set_title(f'Training & Validation Loss ({model_name} - {dataset_type})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_history['acc'], label='Train Acc', color='blue', linewidth=2)
    ax2.plot(val_history['acc'], label='Val Acc', color='red', linewidth=2)
    ax2.set_title(f'Training & Validation Accuracy ({model_name} - {dataset_type})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    save_path = os.path.join(DATA_CONFIG['vis_save_dir'], f'training_curves_{model_name}_{dataset_type}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Vis] Training curves saved: {save_path}")
    plt.close()

# --------------------------
# Modified: Skip confusion matrix (no true labels)
# --------------------------
def plot_confusion_matrix(model, test_loader, class_names, dataset_type, model_name, device):
    print("[Vis] Skipping confusion matrix — test set is unlabeled (no true labels)")
    return  # No true labels → can't compute confusion matrix

# --------------------------
# Unchanged: visualize_feature_maps (no labels needed)
# --------------------------
def visualize_feature_maps(model, image_path, dataset_type, model_name, device):
    data_cfg = DATA_CONFIG[dataset_type]
    transform = get_transforms(  # Need to import get_transforms
        dataset_type=dataset_type,
        normalize_mean=data_cfg['normalize']['mean'],
        normalize_std=data_cfg['normalize']['std'],
        is_train=False
    )
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    feature_maps = {}
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    hooks = []
    if model_name == 'custom_cnn':
        for layer_name in VIS_CONFIG['feature_layers']:
            # Adjust layer access for CustomCNN (conv_layers is a Sequential)
            for idx, module in enumerate(model.conv_layers):
                if isinstance(module, torch.nn.Conv2d) and f'conv{idx//6 + 1}' == layer_name:  # Group by Conv Block
                    hooks.append(module.register_forward_hook(hook_fn(layer_name)))
    
    model.eval()
    with torch.no_grad():
        model(image_tensor)
    
    for hook in hooks:
        hook.remove()
    
    for layer_name, fm in feature_maps.items():
        fm = fm.squeeze(0)
        num_channels = fm.size(0)
        display_channels = min(16, num_channels)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f'Feature Maps - {model_name} - {layer_name}', fontsize=16)
        
        for i in range(display_channels):
            ax = axes[i//4, i%4]
            ax.imshow(fm[i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Channel {i+1}')
        
        for i in range(display_channels, 16):
            axes[i//4, i%4].axis('off')
        
        save_path = os.path.join(DATA_CONFIG['vis_save_dir'], f'feature_maps_{model_name}_{layer_name}_{dataset_type}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Vis] Feature maps saved: {save_path}")
        plt.close()

# --------------------------
# Modified: Skip sample analysis (no true labels)
# --------------------------
def analyze_test_samples(model, test_loader, class_names, dataset_type, model_name, device):
    print("[Vis] Skipping sample analysis — test set is unlabeled (no true labels)")
    return  # No true labels → can't distinguish correct/incorrect predictions

# --------------------------
# Modified: Update run_visualization (pass test_dataset and skip label-dependent steps)
# --------------------------
def run_visualization(model, train_history, val_history, test_loader, class_names, dataset_type, model_name, device, feature_map_img_path=None, test_dataset=None):
    if not VIS_CONFIG['enable']:
        print("[Vis] Visualization disabled")
        return
    
    print("\n=== Starting Visualization ===")
    # 1. Training curves (safe for unlabeled test set)
    plot_training_curves(train_history, val_history, dataset_type, model_name)
    # 2. Confusion matrix (skipped — unlabeled)
    plot_confusion_matrix(model, test_loader, class_names, dataset_type, model_name, device)
    # 3. Feature maps (safe if image path is provided)
    if feature_map_img_path and os.path.exists(feature_map_img_path):
        visualize_feature_maps(model, feature_map_img_path, dataset_type, model_name, device)
    else:
        print("[Vis] Skipping feature maps — no valid image path provided")
    # 4. Sample analysis (skipped — unlabeled)
    analyze_test_samples(model, test_loader, class_names, dataset_type, model_name, device)
    print("=== Visualization Completed ===")

# Add missing import (critical!)
from data_utils import get_transforms