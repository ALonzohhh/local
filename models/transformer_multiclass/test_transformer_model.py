#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer多分类模型测试脚本
用于加载已训练好的模型并在测试数据上评估性能
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import time
import sys

# PyTorch相关导入
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 导入评估工具
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义数据集类 (与原始模型相同)
class NetworkTrafficDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义Transformer模型类 (与原始模型相同)
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super(TransformerClassifier, self).__init__()
        
        # 特征嵌入层 (将输入特征映射到Transformer所需的维度)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码 (简化版本，直接使用可学习的位置嵌入)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # 输入形状: [batch_size, features]
        # 改变为Transformer需要的形状: [batch_size, seq_len=1, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, features]
        x = self.embedding(x)  # [batch_size, 1, d_model]
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # 应用Transformer编码器
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        
        # 提取序列的表示 (这里只有一个位置)
        x = x.squeeze(1)  # [batch_size, d_model]
        
        # 分类层
        x = self.classifier(x)
        return x

def load_and_preprocess_data(test_data_path):
    """加载并预处理测试数据"""
    print(f"加载测试数据: {test_data_path}")
    from fastparquet import ParquetFile
    
    # 加载测试数据
    pf_test = ParquetFile(test_data_path)
    df_test = pf_test.to_pandas()
    
    print(f"测试集大小: {df_test.shape}")
    
    # 提取特征和标签
    X_test = df_test.iloc[:, :35].copy()
    y_test_cat = df_test['attack_cat'].copy()
    
    # 显示测试集类别分布
    print("\n测试集类别分布:")
    test_class_counts = y_test_cat.value_counts()
    print(test_class_counts)
    
    # 加载训练好的标签编码器
    label_encoder_path = os.path.join('models', 'transformer_multiclass', 'label_encoder.joblib')
    print(f"加载标签编码器: {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)
    
    # 加载类别映射
    class_mapping_path = os.path.join('models', 'transformer_multiclass', 'class_mapping.txt')
    class_mapping = {}
    with open(class_mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            class_mapping[int(parts[0])] = parts[1]
    
    # 加载训练数据以获取特征统计信息进行标准化
    train_data_path = 'data/UNSW_NB15_training-set-new.parquet'
    pf_train = ParquetFile(train_data_path)
    df_train = pf_train.to_pandas()
    X_train = df_train.iloc[:, :35].copy()
    
    # 标准化特征
    print("\n正在标准化特征...")
    scaler = StandardScaler()
    
    # 使用训练数据拟合scaler
    numeric_cols = X_train.select_dtypes(include=['float32', 'int16', 'int32', 'int64', 'int8']).columns
    scaler.fit(X_train[numeric_cols])
    
    # 应用到测试数据
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 处理分类特征
    print("处理分类特征...")
    categorical_cols = X_train.select_dtypes(include=['category']).columns
    
    # 如果存在分类特征，进行独热编码
    if len(categorical_cols) > 0:
        X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
        X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
        
        # 确保测试集有与训练集相同的列
        for col in X_train_cat.columns:
            if col not in X_test_cat.columns:
                X_test_cat[col] = 0
        X_test_cat = X_test_cat[X_train_cat.columns]
        
        # 移除原始分类特征并添加独热编码的特征
        X_test = X_test.drop(categorical_cols, axis=1)
        X_test = pd.concat([X_test, X_test_cat], axis=1)
    
    # 编码标签
    y_test_encoded = label_encoder.transform(y_test_cat)
    
    # 转换为NumPy数组
    X_test_np = X_test.values.astype(np.float32)
    
    return X_test_np, y_test_encoded, y_test_cat, class_mapping

def load_model(model_path, input_dim, num_classes):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model

def test_model(model, X_test, y_test, class_mapping, batch_size=256):
    """测试模型性能"""
    # 创建数据集和数据加载器
    test_dataset = NetworkTrafficDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 开始测试
    print("\n开始测试模型...")
    start_time = time.time()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试进度"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    test_time = end_time - start_time
    print(f"测试完成，用时: {test_time:.2f}秒")
    
    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n测试结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 详细分类报告
    print("\n分类报告:")
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    
    # 保存测试报告
    with open('models/transformer_multiclass/test_report.txt', 'w') as f:
        f.write(f"测试时间: {test_time:.2f}秒\n\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(report)
    
    # 绘制混淆矩阵
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title('Transformer模型测试结果 - 混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/transformer_multiclass/test_confusion_matrix.png')
    
    # 计算每个类别的F1分数
    class_wise_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    categories = []
    f1_scores = []
    
    for category, metrics in class_wise_report.items():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            categories.append(category)
            f1_scores.append(metrics['f1-score'])
    
    # 可视化每个类别的F1分数
    categories = [class_mapping[int(cat)] for cat in categories]
    category_indices = np.argsort(f1_scores)
    sorted_categories = [categories[i] for i in category_indices]
    sorted_f1_scores = [f1_scores[i] for i in category_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_categories, sorted_f1_scores, color='skyblue')
    plt.xlabel('F1分数')
    plt.title('各攻击类别的F1分数 (测试结果)')
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/transformer_multiclass/test_category_f1_scores.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_time': test_time
    }

def main():
    """主函数"""
    # 设置路径
    test_data_path = 'data/UNSW_NB15_testing-set.parquet'
    model_path = 'models/transformer_multiclass/transformer_model_best.pth'
    
    # 如果提供了命令行参数，使用指定的测试数据路径
    if len(sys.argv) > 1:
        test_data_path = sys.argv[1]
    
    # 加载和预处理数据
    X_test, y_test_encoded, y_test_cat, class_mapping = load_and_preprocess_data(test_data_path)
    
    # 加载模型
    input_dim = X_test.shape[1]  # 特征数量
    num_classes = len(class_mapping)  # 类别数量
    model = load_model(model_path, input_dim, num_classes)
    
    # 测试模型
    metrics = test_model(model, X_test, y_test_encoded, class_mapping)
    
    print("\nTransformer多分类模型测试完成！")
    print(f"测试数据: {test_data_path}")
    print(f"测试样本数: {len(X_test)}")
    print(f"测试时间: {metrics['test_time']:.2f}秒")
    print(f"总体F1分数: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main() 