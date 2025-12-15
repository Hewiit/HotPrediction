import pandas as pd

# 读取CSV文件
df = pd.read_csv('popularity.csv', encoding='utf-8')

# 找到热度列的最大值
max_popularity = df['热度'].max()

# 计算标准化热度：归一化到0-100
df['标准化热度'] = (df['热度'] / max_popularity) * 100

# 保留两位小数
df['标准化热度'] = df['标准化热度'].round(2)

# 保存结果到新文件（或覆盖原文件）
df.to_csv('popularity.csv', index=False, encoding='utf-8-sig')

print(f"处理完成！")
print(f"最大热度值: {max_popularity}")
print(f"\n前5行数据预览:")
print(df.head())
print(f"\n数据已保存到 popularity.csv")

