import pandas as pd

# 读取数据集
train = pd.read_parquet('data/UNSW_NB15_training-set.parquet')
test = pd.read_parquet('data/UNSW_NB15_testing-set.parquet')

# 打印数据集形状
print(f'训练集形状: {train.shape}')
print(f'测试集形状: {test.shape}')

# 打印数据集中的特征类型
print('\n数据集特征类型:')
print(train.dtypes)

# 打印各攻击类别与标签的对应关系
print('\n攻击类别与标签的对应关系:')
attack_label_relation = pd.crosstab(train['attack_cat'], train['label'])
print(attack_label_relation)

# 计算各攻击类别占比
print('\n训练集攻击类别占比:')
total = len(train)
for cat in train['attack_cat'].unique():
    count = len(train[train['attack_cat'] == cat])
    print(f'{cat}: {count} ({count/total*100:.2f}%)')

# 查看是否有缺失值
print('\n缺失值检查:')
print(train.isnull().sum().any())

# 查看分类特征的唯一值
print('\n分类特征的唯一值:')
categorical_cols = train.select_dtypes(include=['category']).columns
for col in categorical_cols:
    print(f'{col}: {len(train[col].unique())}个唯一值')
    print(train[col].value_counts().head(10))
    print()

# 数值特征统计
print('\n数值特征基本统计:')
print(train.describe().T[['mean', 'std', 'min', 'max']])

# 打印特征列表
print('\n特征列:')
for col in train.columns:
    print(col)

# 打印训练集中攻击类别分布
print('\n训练集攻击类别分布:')
print(train['attack_cat'].value_counts())

# 比较训练集和测试集的分布
print('\n测试集攻击类别分布:')
print(test['attack_cat'].value_counts())
print('\n测试集标签分布:')
print(test['label'].value_counts()) 