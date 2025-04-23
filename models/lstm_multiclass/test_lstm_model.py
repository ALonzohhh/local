#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM多分类模型测试脚本
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

# 定义LSTM模型类 (与原始模型相同)
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        
        # 特征嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2因为是双向LSTM
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 输入形状: [batch_size, features]
        # 改变为LSTM需要的形状: [batch_size, seq_len, hidden_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, features]
        x = self.embedding(x)  # [batch_size, 1, hidden_dim]
        
        # 应用LSTM
        output, (hidden, cell) = self.lstm(x)
        
        # 合并前向和后向的最后一个隐藏状态
        # hidden的形状: [num_layers * 2, batch_size, hidden_dim]
        # 获取最后一层的隐藏状态
        hidden_forward = hidden[-2, :, :]  # 前向LSTM的最后一层
        hidden_backward = hidden[-1, :, :]  # 后向LSTM的最后一层
        hidden_cat = torch.cat([hidden_forward, hidden_backward], dim=1)  # [batch_size, hidden_dim*2]
        
        # 分类层
        x = self.classifier(hidden_cat)
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
    label_encoder_path = os.path.join('models', 'lstm_multiclass', 'label_encoder.joblib')
    print(f"加载标签编码器: {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)
    
    # 加载类别映射
    class_mapping_path = os.path.join('models', 'lstm_multiclass', 'class_mapping.txt')
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
    model = LSTMClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=64,
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
    
    # 计算测试时间
    test_time = time.time() - start_time
    print(f"测试完成，用时: {test_time:.2f}秒")
    
    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n总体性能指标:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 保存评估指标到文件
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_time': test_time
    }
    
    with open('models/lstm_multiclass/test_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # 详细分类报告
    print("\n分类报告:")
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    
    with open('models/lstm_multiclass/test_report.txt', 'w') as f:
        f.write(report)
    
    # 混淆矩阵
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title('LSTM多分类模型测试混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/lstm_multiclass/test_confusion_matrix.png')
    
    # 可视化每个类别的F1分数
    print("\n生成每个类别的F1分数图...")
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    categories = []
    f1_scores = []
    
    for category in class_names:
        if category in report_dict:
            categories.append(category)
            f1_scores.append(report_dict[category]['f1-score'])
    
    # 排序以便更好的可视化
    sorted_indices = np.argsort(f1_scores)
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_categories, sorted_f1_scores, color='skyblue')
    plt.xlabel('F1分数')
    plt.title('各攻击类别的F1分数 (测试集)')
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/lstm_multiclass/test_category_f1_scores.png')
    
    return all_preds, all_labels, metrics

def main():
    """主函数"""
    try:
        # 参数处理
        test_data_path = 'data/UNSW_NB15_testing-set.parquet'
        model_path = 'models/lstm_multiclass/lstm_model_best.pth'
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 '{model_path}' 不存在。请先训练模型。")
            return
        
        # 加载和预处理数据
        X_test, y_test, y_test_cat, class_mapping = load_and_preprocess_data(test_data_path)
        
        # 加载模型
        input_dim = X_test.shape[1]
        num_classes = len(class_mapping)
        model = load_model(model_path, input_dim, num_classes)
        
        # 测试模型
        predictions, true_labels, metrics = test_model(model, X_test, y_test, class_mapping)
        
        print("\nLSTM多分类模型测试完成！")
        print(f"测试F1分数: {metrics['f1']:.4f}")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 