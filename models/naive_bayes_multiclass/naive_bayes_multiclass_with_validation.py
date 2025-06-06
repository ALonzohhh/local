#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于朴素贝叶斯的多分类模型（使用验证集版本）
用于识别UNSW-NB15数据集中的9种攻击类型和1种正常流量
使用高斯朴素贝叶斯算法进行分类决策
"""

# ### 导入必要的库
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import pickle
from tqdm import tqdm

# sklearn相关导入
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ### 创建目录
if not os.path.exists('models/naive_bayes_multiclass'):
    os.makedirs('models/naive_bayes_multiclass')

if not os.path.exists('visualizations/naive_bayes_multiclass'):
    os.makedirs('visualizations/naive_bayes_multiclass')

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
    joblib.dump(label_encoder, 'models/naive_bayes_multiclass/label_encoder.joblib')
    
    # 保存类别映射关系
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open('models/naive_bayes_multiclass/class_mapping.txt', 'w') as f:
        for idx, label in class_mapping.items():
            f.write(f"{idx}: {label}\n")
    
    print(f"类别数量: {len(label_encoder.classes_)}")
    print("类别映射:", class_mapping)
    pbar.update(100)

# 转换为numpy数组
X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# 查看最终数据形状
print(f"\n最终训练特征形状: {X_train_np.shape}")
print(f"最终验证特征形状: {X_val_np.shape}")
print(f"最终测试特征形状: {X_test_np.shape}")

# ### 朴素贝叶斯模型训练和评估
def train_and_evaluate():
    # 初始化模型
    print("\n初始化朴素贝叶斯模型...")
    model = GaussianNB()
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    print("\n开始训练朴素贝叶斯模型...")
    model.fit(X_train_np, y_train_encoded)
    
    # 记录结束时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f"模型训练完成，用时: {training_time:.2f}秒")
    
    # 保存模型
    joblib.dump(model, 'models/naive_bayes_multiclass/naive_bayes_model_with_validation.joblib')
    print("模型已保存")
    
    # 获取并分析特征权重 (朴素贝叶斯中的类条件概率)
    feature_importance = np.abs(model.theta_).mean(axis=0)
    feature_names = X_train.columns
    
    # 只显示前20个重要特征
    indices = np.argsort(feature_importance)[-20:]
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(indices)), feature_importance[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('朴素贝叶斯模型 - 前20个重要特征')
    plt.tight_layout()
    plt.savefig('visualizations/naive_bayes_multiclass/feature_importance_with_validation.png')
    
    # 在验证集上评估模型
    print("\n开始在验证集上评估模型...")
    val_pred = model.predict(X_val_np)
    
    # 计算验证集评估指标
    val_accuracy = accuracy_score(y_val_encoded, val_pred)
    val_precision = precision_score(y_val_encoded, val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(y_val_encoded, val_pred, average='weighted')
    val_f1 = f1_score(y_val_encoded, val_pred, average='weighted')
    
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
        'training_time': training_time
    }
    
    with open('models/naive_bayes_multiclass/validation_metrics.txt', 'w') as f:
        for metric, value in val_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # 验证集详细分类报告
    print("\n验证集分类报告:")
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    val_class_report = classification_report(y_val_encoded, val_pred, target_names=class_names, zero_division=0)
    print(val_class_report)
    
    with open('models/naive_bayes_multiclass/validation_classification_report.txt', 'w') as f:
        f.write(val_class_report)
    
    # 验证集混淆矩阵
    print("\n正在生成验证集混淆矩阵...")
    val_cm = confusion_matrix(y_val_encoded, val_pred)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title('朴素贝叶斯多分类模型验证集混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/naive_bayes_multiclass/validation_confusion_matrix.png')
    
    # 可视化验证集每个类别的F1分数
    print("\n正在生成验证集每个类别的F1分数图...")
    val_report = classification_report(y_val_encoded, val_pred, output_dict=True, zero_division=0)
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
    plt.savefig('visualizations/naive_bayes_multiclass/validation_category_f1_scores.png')
    
    # 如果模型支持概率输出，生成ROC曲线
    if hasattr(model, "predict_proba"):
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        # 计算预测概率
        y_score = model.predict_proba(X_val_np)
        
        # 计算每个类的ROC曲线和AUC
        n_classes = len(class_mapping)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # 计算每个类的ROC
        for i in range(n_classes):
            # 将问题看作一个二分类问题: 当前类vs其他类
            y_true_binary = (y_val_encoded == i).astype(int)
            if i < y_score.shape[1]:  # 确保类别索引有效
                fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制所有ROC曲线
        plt.figure(figsize=(12, 10))
        colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'cyan', 'magenta', 'black', 'orange', 'brown'])
        
        # 首先绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # 绘制每个类的ROC曲线
        for i, color in zip(range(min(n_classes, y_score.shape[1])), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{class_mapping[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('朴素贝叶斯多分类模型 - ROC曲线 (验证集)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('visualizations/naive_bayes_multiclass/validation_roc_curve.png')
    
    print("\n朴素贝叶斯多分类模型训练和评估完成！")
    print("可视化结果已保存到 'visualizations/naive_bayes_multiclass/' 目录")
    print("\n总结:")
    print(f"- 使用训练集: {len(X_train_np)}条记录")
    print(f"- 验证集: {len(X_val_np)}条记录")
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