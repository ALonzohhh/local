import pandas as pd
import os

print("正在检查数据集中的空值...")

# 基本数据集路径
data_paths = [
    "data/UNSW_NB15_training-set.parquet",
    "data/UNSW_NB15_training-set-new.parquet",
    "data/UNSW_NB15_validation-set.parquet",
    "data/UNSW_NB15_testing-set.parquet",
    "data/UNSW_NB15_training-set-with-nulls.parquet",
    "data/UNSW_NB15_training-set-filled.parquet"
]

# 检查文件是否存在
for path in data_paths:
    if os.path.exists(path):
        print(f"文件存在: {path}")
        try:
            # 读取数据
            df = pd.read_parquet(path)
            print(f"成功读取: {path}，形状: {df.shape}")
            
            # 检查空值
            null_count = df.isnull().sum().sum()
            print(f"总空值数量: {null_count}")
            
            # 如果有空值，打印有空值的列
            if null_count > 0:
                print("有空值的列:")
                null_columns = df.columns[df.isnull().any()].tolist()
                for col in null_columns:
                    col_null_count = df[col].isnull().sum()
                    print(f"  {col}: {col_null_count} 空值 ({col_null_count/len(df)*100:.2f}%)")
            else:
                print("该数据集没有空值")
                
            print("-" * 50)
        except Exception as e:
            print(f"读取 {path} 时出错: {e}")
    else:
        print(f"文件不存在: {path}")

print("检查完成！") 