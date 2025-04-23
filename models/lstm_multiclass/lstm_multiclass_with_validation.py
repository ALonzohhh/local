#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于LSTM的多分类模型（使用验证集版本）
用于识别UNSW-NB15数据集中的9种攻击类型和1种正常流量
使用循环神经网络捕捉特征间的序列关系
"""

# ### 导入必要的库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from tqdm import tqdm

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# sklearn相关导入
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ### 创建目录
if not os.path.exists('models/lstm_multiclass'):
    os.makedirs('models/lstm_multiclass')

if not os.path.exists('visualizations/lstm_multiclass'):
    os.makedirs('visualizations/lstm_multiclass')

# ### 数据集类
class NetworkTrafficDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ### LSTM模型定义
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

# ### 数据加载和预处理
print("正在加载数据...")
from fastparquet import ParquetFile
train_set = 'data/UNSW_NB15_training-set-new.parquet'  # 使用新的训练集
val_set = 'data/UNSW_NB15_validation-set.parquet'      # 使用验证集
test_set = 'data/UNSW_NB15_testing-set.parquet'        # 测试集保持不变

pf_train_set = ParquetFile(train_set)
pf_val_set = ParquetFile(val_set)
pf_test_set = ParquetFile(test_set)

df_train = pf_train_set.to_pandas()
df_val = pf_val_set.to_pandas()
df_test = pf_test_set.to_pandas()

# 显示数据集大小
print(f"训练集大小: {df_train.shape}")
print(f"验证集大小: {df_val.shape}")
print(f"测试集大小: {df_test.shape}")

# 提取特征和标签
print("\n正在准备数据...")
X_train = df_train.iloc[:, :35].copy()
y_train_cat = df_train['attack_cat'].copy()
X_val = df_val.iloc[:, :35].copy()
y_val_cat = df_val['attack_cat'].copy()
X_test = df_test.iloc[:, :35].copy()
y_test_cat = df_test['attack_cat'].copy()

print("\n训练集类别分布:")
train_class_counts = y_train_cat.value_counts()
print(train_class_counts)
print("\n验证集类别分布:")
val_class_counts = y_val_cat.value_counts()
print(val_class_counts)
print("\n测试集类别分布:")
test_class_counts = y_test_cat.value_counts()
print(test_class_counts)

# ### 特征预处理
print("\n正在处理特征...")

# 标准化数值特征
with tqdm(total=100, desc="标准化数值特征") as pbar:
    scaler = StandardScaler()
    
    # 处理训练集并拟合scaler
    numeric_cols = X_train.select_dtypes(include=['float32', 'int16', 'int32', 'int64', 'int8']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    pbar.update(30)
    
    # 处理验证集
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    pbar.update(30)
    
    # 处理测试集
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    pbar.update(40)

# 处理分类特征 - 独热编码
with tqdm(total=100, desc="处理分类特征") as pbar:
    categorical_cols = X_train.select_dtypes(include=['category']).columns
    
    # 如果存在分类特征，进行独热编码
    if len(categorical_cols) > 0:
        X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
        X_val_cat = pd.get_dummies(X_val[categorical_cols], drop_first=True)
        X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
        
        # 确保验证集和测试集与训练集有相同的列
        for col in X_train_cat.columns:
            if col not in X_val_cat.columns:
                X_val_cat[col] = 0
            if col not in X_test_cat.columns:
                X_test_cat[col] = 0
        
        X_val_cat = X_val_cat[X_train_cat.columns]
        X_test_cat = X_test_cat[X_train_cat.columns]
        
        # 移除原始分类特征并添加独热编码的特征
        X_train = X_train.drop(categorical_cols, axis=1)
        X_val = X_val.drop(categorical_cols, axis=1)
        X_test = X_test.drop(categorical_cols, axis=1)
        
        X_train = pd.concat([X_train, X_train_cat], axis=1)
        X_val = pd.concat([X_val, X_val_cat], axis=1)
        X_test = pd.concat([X_test, X_test_cat], axis=1)
    pbar.update(100)

# 编码标签
with tqdm(total=100, desc="编码标签") as pbar:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_cat)
    y_val_encoded = label_encoder.transform(y_val_cat)
    y_test_encoded = label_encoder.transform(y_test_cat)
    
    # 保存标签编码器
    joblib.dump(label_encoder, 'models/lstm_multiclass/label_encoder.joblib')
    
    # 保存类别映射关系
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open('models/lstm_multiclass/class_mapping.txt', 'w') as f:
        for idx, label in class_mapping.items():
            f.write(f"{idx}: {label}\n")
    
    print(f"类别数量: {len(label_encoder.classes_)}")
    print("类别映射:", class_mapping)
    pbar.update(100)

# 转换为numpy数组
X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# ### 创建数据加载器
BATCH_SIZE = 256  # 可调整以适应内存限制
print(f"\n使用批次大小: {BATCH_SIZE}")

train_dataset = NetworkTrafficDataset(X_train_np, y_train_encoded)
val_dataset = NetworkTrafficDataset(X_val_np, y_val_encoded)
test_dataset = NetworkTrafficDataset(X_test_np, y_test_encoded)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 验证集不需要打乱
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ### 模型训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix(
                loss=running_loss/len(progress_bar), 
                accuracy=100.*correct/total
            )
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100. * val_correct / val_total
        val_losses.append(val_epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.2f}%')
        
        # 保存最佳模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), 'models/lstm_multiclass/lstm_model_best.pth')
            print(f'模型已保存: 验证损失从 {best_val_loss:.4f} 改善到 {val_epoch_loss:.4f}')
    
    return train_losses, val_losses

# ### 评估函数
def evaluate_model(model, data_loader, criterion, set_name="Validation"):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f'评估 {set_name} 集'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集所有预测和标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    print(f'{set_name} 损失: {avg_loss:.4f}, 准确率: {100*accuracy:.2f}%')
    
    return all_preds, all_labels, avg_loss, accuracy

# ### 模型训练和评估
def train_and_evaluate():
    # 模型参数
    input_dim = X_train_np.shape[1]  # 特征数量
    num_classes = len(label_encoder.classes_)  # 类别数量
    
    # 初始化模型
    model = LSTMClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 训练参数
    epochs = 10
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    print(f"\n开始训练LSTM模型，将运行{epochs}个epoch...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    
    # 记录结束时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f"模型训练完成，用时: {training_time:.2f}秒")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'models/lstm_multiclass/lstm_model_final.pth')
    print("最终模型已保存")
    
    # 绘制训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', label='训练损失')
    plt.plot(val_losses, marker='o', label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/lstm_multiclass/training_validation_loss.png')
    
    # 加载最佳模型进行评估
    print("\n加载最佳模型进行评估...")
    model.load_state_dict(torch.load('models/lstm_multiclass/lstm_model_best.pth'))
    
    # 在验证集上评估模型
    print("\n开始在验证集上评估模型...")
    val_predictions, val_true_labels, val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, "验证")
    
    # 计算验证集评估指标
    val_precision = precision_score(val_true_labels, val_predictions, average='weighted', zero_division=0)
    val_recall = recall_score(val_true_labels, val_predictions, average='weighted')
    val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
    
    print("\n验证集性能指标:")
    print(f"准确率: {val_accuracy:.4f}")
    print(f"精确率: {val_precision:.4f}")
    print(f"召回率: {val_recall:.4f}")
    print(f"F1分数: {val_f1:.4f}")
    
    # 保存验证集评估指标
    val_metrics = {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'loss': val_loss
    }
    
    with open('models/lstm_multiclass/validation_metrics.txt', 'w') as f:
        for metric, value in val_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # 验证集详细分类报告
    print("\n验证集分类报告:")
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    val_class_report = classification_report(val_true_labels, val_predictions, target_names=class_names, zero_division=0)
    print(val_class_report)
    
    with open('models/lstm_multiclass/validation_classification_report.txt', 'w') as f:
        f.write(val_class_report)
    
    # 验证集混淆矩阵
    print("\n正在生成验证集混淆矩阵...")
    val_cm = confusion_matrix(val_true_labels, val_predictions)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title('LSTM多分类模型验证集混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/lstm_multiclass/validation_confusion_matrix.png')
    
    # 可视化验证集每个类别的F1分数
    print("\n正在生成验证集每个类别的F1分数图...")
    val_report = classification_report(val_true_labels, val_predictions, output_dict=True, zero_division=0)
    categories = []
    f1_scores = []
    
    for i, category in enumerate(class_names):
        if category in val_report:
            categories.append(category)
            f1_scores.append(val_report[category]['f1-score'])
    
    # 排序以便更好的可视化
    sorted_indices = np.argsort(f1_scores)
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_categories, sorted_f1_scores, color='skyblue')
    plt.xlabel('F1分数')
    plt.title('各攻击类别的F1分数 (验证集)')
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/lstm_multiclass/validation_category_f1_scores.png')
    
    print("\nLSTM多分类模型训练和评估完成！")
    print("可视化结果已保存到 'visualizations/lstm_multiclass/' 目录")
    print("\n总结:")
    print(f"- 使用训练集: {len(X_train)}条记录")
    print(f"- 验证集: {len(X_val)}条记录")
    print(f"- 识别所有{len(class_mapping)}个攻击类别")
    print(f"- 训练时间: {training_time:.2f}秒")
    print(f"- 验证集F1分数: {val_f1:.4f}")
    
    return model, val_metrics

# ### 主程序
if __name__ == "__main__":
    try:
        train_and_evaluate()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc() 