# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from PIL import Image
# import os
# from tqdm import tqdm
# from config import MODEL_CONFIG, DATA_CONFIG, TRAINING_CONFIG, CLASS_CONFIG

# IMG_SIZE = MODEL_CONFIG['img_size']
# BATCH_SIZE = MODEL_CONFIG['batch_size']
# NUM_EPOCHS = MODEL_CONFIG['num_epochs']
# LEARNING_RATE = MODEL_CONFIG['learning_rate']
# WEIGHT_DECAY = MODEL_CONFIG['weight_decay']
# NUM_CLASSES = MODEL_CONFIG['num_classes']

# TRAIN_DIR = DATA_CONFIG['train_dir']
# VAL_DIR = DATA_CONFIG['val_dir']
# MODEL_SAVE_PATH = DATA_CONFIG['model_save_path']

# NUM_WORKERS = TRAINING_CONFIG['num_workers']
# PIN_MEMORY = TRAINING_CONFIG['pin_memory']
# LR_SCHEDULER_PATIENCE = TRAINING_CONFIG['lr_scheduler_patience']
# LR_SCHEDULER_FACTOR = TRAINING_CONFIG['lr_scheduler_factor']

# CLASS_NAMES = CLASS_CONFIG['class_names']

# class CatDogCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(CatDogCNN, self).__init__()
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(256 * 14 * 14, 512)
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, 256)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(256, num_classes)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x

# def train_model(model, train_loader, criterion, optimizer, device, epoch):
#     """Train one epoch with progress bar"""
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]', 
#                 ncols=100, ascii=True)
    
#     for images, labels in pbar:
#         images = images.to(device)
#         labels = labels.to(device)
        
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#         current_loss = running_loss / total
#         current_acc = 100 * correct / total
#         pbar.set_postfix({'loss': f'{current_loss:.4f}', 
#                           'acc': f'{current_acc:.2f}%'})
    
#     epoch_loss = running_loss / total
#     epoch_acc = 100 * correct / total
#     return epoch_loss, epoch_acc

# def validate_model(model, val_loader, criterion, device, epoch):
#     """Validate model with progress bar"""
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
    
#     pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]', 
#                 ncols=100, ascii=True)
    
#     with torch.no_grad():
#         for images, labels in pbar:
#             images = images.to(device)
#             labels = labels.to(device)
            
#             outputs = model(images)
#             loss = criterion(outputs, labels)
            
#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             current_loss = running_loss / total
#             current_acc = 100 * correct / total
#             pbar.set_postfix({'loss': f'{current_loss:.4f}', 
#                               'acc': f'{current_acc:.2f}%'})
    
#     epoch_loss = running_loss / total
#     epoch_acc = 100 * correct / total
#     return epoch_loss, epoch_acc

# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device: {device}')
    
#     train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
#     ])

#     val_transforms = transforms.Compose([
#         transforms.Resize(int(IMG_SIZE * 1.14)),
#         transforms.CenterCrop(IMG_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#     ])

    
#     train_dataset = datasets.ImageFolder(root = TRAIN_DIR,
#                                         transform = train_transforms)
#     val_dataset = datasets.ImageFolder(root = VAL_DIR,
#                                       transform = val_transforms)
    
#     train_loader = DataLoader(train_dataset,
#                              batch_size = BATCH_SIZE,
#                              shuffle = True,
#                              num_workers = NUM_WORKERS,
#                              pin_memory = PIN_MEMORY)
    
#     val_loader = DataLoader(val_dataset,
#                            batch_size = BATCH_SIZE,
#                            shuffle = False,
#                            num_workers = NUM_WORKERS,
#                            pin_memory = PIN_MEMORY)
    
#     print(f'Training set size: {len(train_dataset)}')
#     print(f'Validation set size: {len(val_dataset)}')
#     print(f'Class mapping: {train_dataset.class_to_idx}')
    
#     model = CatDogCNN(num_classes=NUM_CLASSES).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), 
#                           lr = LEARNING_RATE, 
#                           weight_decay = WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', 
#                                                      factor = LR_SCHEDULER_FACTOR, 
#                                                      patience = LR_SCHEDULER_PATIENCE)
    
#     best_val_acc = 0.0
#     train_history = {'loss': [], 'acc': []}
#     val_history = {'loss': [], 'acc': []}
    
#     print('\nStarting training...')
#     print('=' * 60)
    
#     for epoch in range(NUM_EPOCHS):
#         train_loss, train_acc = train_model(model, train_loader, criterion, 
#                                            optimizer, device, epoch)
        
#         val_loss, val_acc = validate_model(model, val_loader, criterion, 
#                                           device, epoch)
        
#         train_history['loss'].append(train_loss)
#         train_history['acc'].append(train_acc)
#         val_history['loss'].append(val_loss)
#         val_history['acc'].append(val_acc)
        
#         scheduler.step(val_loss)
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f'  Current learning rate: {current_lr:.6f}')
        
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#             }, 'best_model.pth')
#             print(f'  ✓ Best model saved (Val accuracy: {val_acc:.2f}%)')
        
#         print('-' * 60)
    
#     print('\nTraining completed!')
#     print(f'Best validation accuracy: {best_val_acc:.2f}%')

# def predict_image(image_path, model, device, class_names):
#     """Predict a single image"""
#     image = Image.open(image_path).convert('RGB')
    
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                            std=[0.229, 0.224, 0.225])
#     ])
    
#     image_tensor = transform(image).unsqueeze(0)
    
#     model.eval()
#     with torch.no_grad():
#         image_tensor = image_tensor.to(device)
#         outputs = model(image_tensor)
#         probabilities = torch.softmax(outputs, dim=1)
#         confidence, predicted = torch.max(probabilities, 1)
    
#     predicted_class = class_names[predicted.item()]
#     confidence_score = confidence.item() * 100
    
#     return predicted_class, confidence_score

# def predict_test_folder(test_folder, model, device, class_names):
#     """Batch predict images in test folder"""
#     model.eval()
#     results = []
    
#     image_files = [f for f in os.listdir(test_folder) 
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     print(f'\nStarting prediction for {len(image_files)} images...')
    
#     for filename in tqdm(image_files, desc='Prediction progress', ncols=100, ascii=True):
#         image_path = os.path.join(test_folder, filename)
#         predicted_class, confidence = predict_image(image_path, model, 
#                                                     device, class_names)
#         results.append({
#             'filename': filename,
#             'predicted_class': predicted_class,
#             'confidence': confidence
#         })
#         print(f'{filename}: {predicted_class} ({confidence:.2f}%)')
    
#     return results

# if __name__ == '__main__':
#     main()
import torch
import os
import csv
import numpy as np
from PIL import Image
from config import DATA_CONFIG, MODEL_CONFIG, ENSEMBLE_CONFIG
from models import get_model
from data_utils import get_transforms

class Predictor:
    """单模型预测器"""
    def __init__(self, model_path, dataset_type, device):
        self.device = device
        self.dataset_type = dataset_type
        self.data_cfg = DATA_CONFIG[dataset_type]
        self.class_names = self.data_cfg['class_names']
        self.num_classes = self.data_cfg['num_classes']
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        self.model_name = checkpoint['model_name']
        self.model = get_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 加载预处理管道
        self.transform = get_transforms(
            dataset_type=dataset_type,
            normalize_mean=self.data_cfg['normalize']['mean'],
            normalize_std=self.data_cfg['normalize']['std'],
            is_train=False
        )
    
    def predict_single(self, image_path):
        """预测单张图像"""
        # 预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        return {
            'filename': os.path.basename(image_path),
            'true_class': None,  # 测试集无真实标签
            'pred_class': self.class_names[pred_idx.item()],
            'confidence': confidence.item() * 100,
            'probabilities': probs.cpu().numpy()[0]
        }
    
    def predict_batch(self, folder_path):
        """批量预测文件夹图像"""
        results = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, filename)
                result = self.predict_single(image_path)
                results.append(result)
                print(f"{result['filename']} → {result['pred_class']} ({result['confidence']:.2f}%)")
        
        # 保存结果到CSV
        save_path = DATA_CONFIG['predictions_save_path']
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['filename', 'pred_class', 'confidence'] + [f'prob_{cls}' for cls in self.class_names]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                row = {
                    'filename': res['filename'],
                    'pred_class': res['pred_class'],
                    'confidence': res['confidence']
                }
                for i, cls in enumerate(self.class_names):
                    row[f'prob_{cls}'] = res['probabilities'][i]
                writer.writerow(row)
        print(f"\n批量预测结果已保存：{save_path}")
        return results

class EnsemblePredictor:
    """集成学习预测器（多模型融合）"""
    def __init__(self, dataset_type, device):
        self.device = device
        self.dataset_type = dataset_type
        self.data_cfg = DATA_CONFIG[dataset_type]
        self.class_names = self.data_cfg['class_names']
        self.num_classes = self.data_cfg['num_classes']
        self.model_list = ENSEMBLE_CONFIG['model_list']
        self.fusion_strategy = ENSEMBLE_CONFIG['fusion_strategy']
        self.weights = ENSEMBLE_CONFIG['weights']
        
        # 加载所有集成模型
        self.models = self._load_ensemble_models()
        # 预处理管道
        self.transform = get_transforms(
            dataset_type=dataset_type,
            normalize_mean=self.data_cfg['normalize']['mean'],
            normalize_std=self.data_cfg['normalize']['std'],
            is_train=False
        )
    
    def _load_ensemble_models(self):
        """加载所有集成模型的权重"""
        models = []
        save_dir = DATA_CONFIG['ensemble_save_dir']
        
        for model_name in self.model_list:
            model_path = os.path.join(save_dir, f'best_{model_name}_{self.dataset_type}.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"集成模型权重不存在：{model_path}（请先训练所有集成模型）")
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            model = get_model(
                model_name=model_name,
                num_classes=self.num_classes,
                pretrained=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            models.append(model)
            print(f"已加载集成模型：{model_name}")
        
        return models
    
    def _fusion(self, all_probs):
        """根据融合策略计算最终预测"""
        if self.fusion_strategy == 'hard_vote':
            # 硬投票：取多数模型预测的类别
            predictions = [np.argmax(prob) for prob in all_probs]
            final_pred_idx = max(set(predictions), key=predictions.count)
            # 置信度：多数类的平均概率
            confidences = [all_probs[i][final_pred_idx] for i, pred in enumerate(predictions) if pred == final_pred_idx]
            final_conf = np.mean(confidences) * 100
        
        elif self.fusion_strategy == 'soft_vote':
            # 软投票：平均所有模型的概率
            avg_probs = np.mean(all_probs, axis=0)
            final_pred_idx = np.argmax(avg_probs)
            final_conf = avg_probs[final_pred_idx] * 100
        
        elif self.fusion_strategy == 'weighted':
            # 加权融合：按模型权重加权概率和
            if len(self.weights) != len(self.models):
                raise ValueError(f"权重数量（{len(self.weights)}）与模型数量（{len(self.models)}）不匹配")
            weighted_probs = np.average(all_probs, axis=0, weights=self.weights)
            final_pred_idx = np.argmax(weighted_probs)
            final_conf = weighted_probs[final_pred_idx] * 100
        
        else:
            raise ValueError(f"不支持的融合策略：{self.fusion_strategy}")
        
        return final_pred_idx, final_conf
    
    def predict_single(self, image_path):
        """集成预测单张图像"""
        # 预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 收集所有模型的概率
        all_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
        
        # 融合结果
        final_pred_idx, final_conf = self._fusion(all_probs)
        
        return {
            'filename': os.path.basename(image_path),
            'pred_class': self.class_names[final_pred_idx],
            'confidence': final_conf,
            'model_probs': all_probs  # 各模型的原始概率
        }
    
    def predict_batch(self, folder_path):
        """集成批量预测"""
        results = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        print(f"\n=== 集成预测（{self.fusion_strategy}）===")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, filename)
                result = self.predict_single(image_path)
                results.append(result)
                print(f"{result['filename']} → {result['pred_class']} ({result['confidence']:.2f}%)")
        
        # 保存结果到CSV
        save_path = f"ensemble_{DATA_CONFIG['predictions_save_path']}"
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['filename', 'pred_class', 'confidence']
            for i, model_name in enumerate(self.model_list):
                for j, cls in enumerate(self.class_names):
                    fieldnames.append(f'{model_name}_prob_{cls}')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                row = {
                    'filename': res['filename'],
                    'pred_class': res['pred_class'],
                    'confidence': res['confidence']
                }
                for i, (model_name, model_prob) in enumerate(zip(self.model_list, res['model_probs'])):
                    for j, cls in enumerate(self.class_names):
                        row[f'{model_name}_prob_{cls}'] = model_prob[j]
                writer.writerow(row)
        print(f"\n集成预测结果已保存：{save_path}")
        return results