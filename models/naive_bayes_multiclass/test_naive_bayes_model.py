#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
朴素贝叶斯多分类模型测试脚本
用于加载已训练好的模型并在测试数据上评估性能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import time
import sys

# sklearn相关导入
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

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
    label_encoder_path = os.path.join('models', 'naive_bayes_multiclass', 'label_encoder.joblib')
    print(f"加载标签编码器: {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)
    
    # 加载类别映射
    class_mapping_path = os.path.join('models', 'naive_bayes_multiclass', 'class_mapping.txt')
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

def load_model(model_path):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    model = joblib.load(model_path)
    return model

def test_model(model, X_test, y_test, class_mapping):
    """测试模型性能"""
    # 开始测试
    print("\n开始测试模型...")
    start_time = time.time()
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算测试时间
    test_time = time.time() - start_time
    print(f"测试完成，用时: {test_time:.2f}秒")
    
    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
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
    
    with open('models/naive_bayes_multiclass/test_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # 详细分类报告
    print("\n分类报告:")
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    with open('models/naive_bayes_multiclass/test_report.txt', 'w') as f:
        f.write(report)
    
    # 混淆矩阵
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title('朴素贝叶斯多分类模型测试混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/naive_bayes_multiclass/test_confusion_matrix.png')
    
    # 可视化每个类别的F1分数
    print("\n生成每个类别的F1分数图...")
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
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
    plt.title('测试集上各攻击类别的F1分数')
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/naive_bayes_multiclass/test_category_f1_scores.png')
    
    # 如果模型支持概率输出，生成ROC曲线
    if hasattr(model, "predict_proba"):
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        # 计算预测概率
        y_score = model.predict_proba(X_test)
        
        # 计算每个类的ROC曲线和AUC
        n_classes = len(class_mapping)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # 计算每个类的ROC
        for i in range(n_classes):
            # 将问题看作一个二分类问题: 当前类vs其他类
            y_true_binary = (y_test == i).astype(int)
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
        plt.title('朴素贝叶斯多分类模型 - ROC曲线 (测试集)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('visualizations/naive_bayes_multiclass/test_roc_curve.png')
    
    return y_pred, metrics

def main():
    """主函数"""
    # 确保输出目录存在
    if not os.path.exists('visualizations/naive_bayes_multiclass'):
        os.makedirs('visualizations/naive_bayes_multiclass')
    
    # 数据路径
    test_data_path = 'data/UNSW_NB15_testing-set.parquet'
    model_path = 'models/naive_bayes_multiclass/naive_bayes_model_with_validation.joblib'
    
    try:
        # 加载和预处理测试数据
        X_test, y_test, y_test_cat, class_mapping = load_and_preprocess_data(test_data_path)
        
        # 加载模型
        model = load_model(model_path)
        
        # 测试模型
        y_pred, metrics = test_model(model, X_test, y_test, class_mapping)
        
        print("\n朴素贝叶斯多分类模型测试完成！")
        print(f"测试集大小: {len(X_test)} 样本")
        print(f"整体F1分数: {metrics['f1']:.4f}")
        print(f"测试时间: {metrics['test_time']:.2f} 秒")
        
    except FileNotFoundError as e:
        print(f"错误: 找不到所需文件 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 