CIFAR10_TRAINING_CONFIG = {
    'num_workers': 4,
    'pin_memory': True,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5
}

CIFAR10_CLASS_CONFIG = {
    'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
}

# CIFAR-10专用配置
CIFAR10_DATA_CONFIG = {
    'data_root': './cifar10_data',
    'model_save_path': 'cifar10_best_model.pth',  # 原CNN模型路径
    'resnet_model_save_path': 'cifar10_resnet_best_model.pth',  # 新增ResNet路径
    'predictions_save_path': 'cifar10_predictions.csv'
}


CIFAR10_MODEL_CONFIG = {
    'img_size': 32,
    'num_classes': 10,
    'batch_size': 128,  # ResNet可适当调大batch_size（若GPU显存足够）
    'num_epochs': 50,
    'learning_rate': 0.001,  # ResNet可尝试0.0005~0.001
    'weight_decay': 0.0001,
    'dropout_rate': 0.5
}