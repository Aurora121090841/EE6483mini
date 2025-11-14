# '''
# Author: Aurora
# Date: 2025-11-14 00:44:12
# Descripttion: 
# LastEditTime: 2025-11-14 00:53:29
# '''
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG
# import os

# NUM_EPOCHS = MODEL_CONFIG['num_epochs']
# LEARNING_RATE = MODEL_CONFIG['learning_rate']
# WEIGHT_DECAY = MODEL_CONFIG['weight_decay']
# LR_SCHEDULER_PATIENCE = TRAINING_CONFIG['lr_scheduler_patience']
# LR_SCHEDULER_FACTOR = TRAINING_CONFIG['lr_scheduler_factor']
# MODEL_SAVE_PATH = DATA_CONFIG['model_save_path']

# def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
#     """单轮训练"""
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]', ncols=100, ascii=True)
#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#         pbar.set_postfix({
#             'loss': f'{running_loss/total:.4f}',
#             'acc': f'{100*correct/total:.2f}%'
#         })
    
#     return running_loss / total, 100 * correct / total

# def validate(model, val_loader, criterion, device, epoch):
#     """验证模型"""
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]', ncols=100, ascii=True)
#     with torch.no_grad():
#         for images, labels in pbar:
#             images, labels = images.to(device), labels.to(device)
            
#             outputs = model(images)
#             loss = criterion(outputs, labels)
            
#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             pbar.set_postfix({
#                 'loss': f'{running_loss/total:.4f}',
#                 'acc': f'{100*correct/total:.2f}%'
#             })
    
#     return running_loss / total, 100 * correct / total

# def train_model(model, train_loader, val_loader, device):
#     """完整训练流程"""
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY
#     )
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=LR_SCHEDULER_FACTOR,
#         patience=LR_SCHEDULER_PATIENCE
#     )
    
#     best_val_acc = 0.0
#     model.to(device)
    
#     print('\n开始训练...')
#     print('=' * 60)
    
#     # 初始化历史记录
#     train_history = {'loss': [], 'acc': []}
#     val_history = {'loss': [], 'acc': []}
    

#     for epoch in range(NUM_EPOCHS):
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
#         val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
#         # 保存历史记录
#         train_history['loss'].append(train_loss)
#         train_history['acc'].append(train_acc)
#         val_history['loss'].append(val_loss)
#         val_history['acc'].append(val_acc)

#         # 学习率调整
#         scheduler.step(val_loss)
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f'  当前学习率: {current_lr:.6f}')
        
#         # 保存最佳模型
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc
#             }, MODEL_SAVE_PATH)
#             print(f'  ✓ 最佳模型已保存 (验证集准确率: {val_acc:.2f}%)')
        
#         print('-' * 60)
    
#     print('\n训练完成!')
#     print(f'最佳验证集准确率: {best_val_acc:.2f}%')
#     return model, train_history, val_history


# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
# 新增导入 ENSEMBLE_CONFIG（之前漏了）
from config import TRAIN_CONFIG, DATA_CONFIG, IMBALANCE_CONFIG, ENSEMBLE_CONFIG  
from data_utils import get_class_weights

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """单轮训练"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计指标
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / total
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """单轮验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', ncols=100)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / total
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, dataset_type, train_dataset, device, model_name):
    """完整训练流程（返回训练历史）"""
    # 损失函数（支持加权损失）
    criterion = None
    if IMBALANCE_CONFIG['strategy'] == 'weighted_loss':
        class_weights = get_class_weights(dataset_type, train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用加权损失函数，类别权重：{class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 优化器（Adam自适应优化）
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # 学习率调度器（根据验证损失调整）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=TRAIN_CONFIG['lr_scheduler']['patience'],
        factor=TRAIN_CONFIG['lr_scheduler']['factor'],
        verbose=True
    )
    
    # 训练历史记录（用于可视化）
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}
    
    # 修复：将 DATA_CONFIG['ensemble_config']['enable'] 改为 ENSEMBLE_CONFIG['enable']
    save_dir = DATA_CONFIG['ensemble_save_dir'] if ENSEMBLE_CONFIG['enable'] else DATA_CONFIG['model_save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f'best_{model_name}_{dataset_type}.pth')
    best_val_acc = 0.0
    

    # 训练循环
    print(f"\n=== 开始训练 {model_name} 模型 ===")
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, TRAIN_CONFIG['num_epochs'])
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, TRAIN_CONFIG['num_epochs'])
        
        # 记录历史
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'dataset_type': dataset_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_history': train_history,
                'val_history': val_history
            }, best_model_path)
            print(f"✓ 最佳模型已保存（验证准确率：{best_val_acc:.2f}%）")
        
        print("-" * 80)
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳验证准确率：{best_val_acc:.2f}%")
    print(f"模型保存路径：{best_model_path}")
    
    return model, train_history, val_history, best_model_path