import sys

import pandas as pd

# 读取 CSV 文件
csv_name = sys.argv[1]
file = f'./Data/CsvMeasurements/{csv_name}.csv'
df = pd.read_csv(file)

# 确定要正规化的列（前 n-1 列）
n = df.shape[1]  # 总列数
columns_to_normalize = df.columns[:n-1]  # 前 n-1 列

# 进行 Min-Max 归一化
df_normalized = df.copy()
df_normalized[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

# 将结果保存到新的 CSV 文件
df_normalized.to_csv(file, index=False)

print(f"正规化完成！结果已保存到 '{file}'。")
