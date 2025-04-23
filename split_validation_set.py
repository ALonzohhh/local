#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练集分割脚本
从训练数据集中分割出验证集，确保它不会在训练数据中出现
"""

import os
import numpy as np
import pandas as pd
from fastparquet import ParquetFile, write
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("正在加载训练数据...")
train_data_path = 'data/UNSW_NB15_training-set.parquet'
pf_train = ParquetFile(train_data_path)
df_train = pf_train.to_pandas()

print(f"原始训练集大小: {df_train.shape}")

# 检查训练集中类别分布
print("\n原始训练集类别分布:")
train_class_counts = df_train['attack_cat'].value_counts()
print(train_class_counts)

# 确保每个类别都被分到训练集和验证集
# 使用分层抽样，保持类别比例
train_ratio = 0.6  # 60%用于训练，40%用于验证

print(f"\n正在分割数据集，训练集比例: {train_ratio}, 验证集比例: {1-train_ratio}")
df_train_new, df_val = train_test_split(
    df_train, 
    test_size=1-train_ratio,
    stratify=df_train['attack_cat'],  # 保持类别比例
    random_state=42
)

print(f"新训练集大小: {df_train_new.shape}")
print(f"验证集大小: {df_val.shape}")

# 检查分割后的类别分布
print("\n新训练集类别分布:")
new_train_class_counts = df_train_new['attack_cat'].value_counts()
print(new_train_class_counts)

print("\n验证集类别分布:")
val_class_counts = df_val['attack_cat'].value_counts()
print(val_class_counts)

# 确保数据分布正确
print("\n验证各类别分布比例是否保持一致:")
for cat in train_class_counts.index:
    original_ratio = train_class_counts[cat] / len(df_train)
    new_train_ratio = new_train_class_counts[cat] / len(df_train_new)
    val_ratio = val_class_counts[cat] / len(df_val)
    print(f"{cat}: 原始数据 {original_ratio:.4f}, 新训练集 {new_train_ratio:.4f}, 验证集 {val_ratio:.4f}")

# 保存新的训练集和验证集
if not os.path.exists('data'):
    os.makedirs('data')

print("\n正在保存新的训练集和验证集...")
write('data/UNSW_NB15_training-set-new.parquet', df_train_new)
write('data/UNSW_NB15_validation-set.parquet', df_val)

print("数据集分割完成！")
print(f"新的训练集保存至: data/UNSW_NB15_training-set-new.parquet, 样本数: {len(df_train_new)}")
print(f"验证集保存至: data/UNSW_NB15_validation-set.parquet, 样本数: {len(df_val)}") 