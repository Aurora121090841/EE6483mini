import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import os
from config import TRAIN_CONFIG, IMBALANCE_CONFIG, MODEL_CONFIG, DATA_CONFIG
from data_utils import get_class_weights

def train_model(model, train_loader, val_loader, class_names, device):
    # --------------------------
    # 损失函数（支持加权损失开关）
    # --------------------------
    criterion = None
    if IMBALANCE_CONFIG['enable'] and IMBALANCE_CONFIG['strategy'] == 'weighted_loss':
        dataset_type = DATA_CONFIG['current_dataset']
        train_dataset = train_loader.dataset
        class_weights = get_class_weights(dataset_type, train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights * IMBALANCE_CONFIG['weighted_loss_factor'])
        print(f"使用加权交叉熵损失，类别权重：{class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用普通交叉熵损失")

    # --------------------------
    # 优化器
    # --------------------------
    if TRAIN_CONFIG['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])
    elif TRAIN_CONFIG['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])
    elif TRAIN_CONFIG['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], momentum=0.9, weight_decay=TRAIN_CONFIG['weight_decay'])
    else:
        raise ValueError(f"不支持的优化器：{TRAIN_CONFIG['optimizer']}")

    # --------------------------
    # 学习率调度器
    # --------------------------
    scheduler = None
    if TRAIN_CONFIG['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['num_epochs'], eta_min=1e-6)
    elif TRAIN_CONFIG['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=TRAIN_CONFIG['step_size'], gamma=TRAIN_CONFIG['gamma'])

    # --------------------------
    # 训练记录
    # --------------------------
    best_val_acc = 0.0
    early_stop_count = 0
    model_save_path = TRAIN_CONFIG['model_save_path']
    save_dir = os.path.dirname(model_save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # --------------------------
    # 训练循环（修复损失计算）
    # --------------------------
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{TRAIN_CONFIG["num_epochs"]} (Train)')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # 修复：补充 labels 参数（outputs=模型输出，labels=真实标签）
            loss = criterion(outputs, labels)  
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': train_loss/train_total, 'acc': 100.*train_correct/train_total})
        pbar.close()

        # --------------------------
        # 验证循环（同样修复损失计算）
        # --------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{TRAIN_CONFIG["num_epochs"]} (Val)')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # 修复：补充 labels 参数
                loss = criterion(outputs, labels)  

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({'loss': val_loss/val_total, 'acc': 100.*val_correct/val_total})
            pbar.close()

        # 计算指标
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss_avg = train_loss / train_total
        val_loss_avg = val_loss / val_total

        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')

        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            print(f'Current LR: {scheduler.get_last_lr()[0]:.6f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, model_save_path)
            print(f'✅ 保存最佳模型到：{model_save_path}（Val Acc: {best_val_acc:.2f}%）')
            early_stop_count = 0
        else:
            early_stop_count += 1
            print(f'❌ 验证集准确率未提升，早停计数：{early_stop_count}/{TRAIN_CONFIG["early_stop_patience"]}')
            if early_stop_count >= TRAIN_CONFIG['early_stop_patience']:
                print(f'⚠️  早停触发，停止训练')
                break

    print(f'\n训练完成！最佳验证准确率：{best_val_acc:.2f}%')
    return best_val_acc

def test_model(model, test_loader, class_names, device):
    """测试模型"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'acc': 100.*test_correct/test_total})
        pbar.close()

    test_acc = 100. * test_correct / test_total
    print(f'\n测试集准确率：{test_acc:.2f}%')
    return test_acc, all_preds, all_labels