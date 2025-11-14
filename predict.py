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
