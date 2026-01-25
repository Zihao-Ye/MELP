import pandas as pd
from collections import Counter

# 读取CSV文件
train_df = pd.read_csv('/root/MELP/src/melp/data_split/mimic-iv-ecg/train.csv')
val_df = pd.read_csv('/root/MELP/src/melp/data_split/mimic-iv-ecg/val.csv')

# 定义report列
report_cols = [f'report_{i}' for i in range(18)]

# 收集所有诊断（转为小写）
all_diagnoses = []

for df in [train_df, val_df]:
    for col in report_cols:
        if col in df.columns:
            # 获取非空值，转为小写
            values = df[col].dropna().astype(str).str.lower().str.strip()
            all_diagnoses.extend(values.tolist())

# 过滤空字符串
all_diagnoses = [d for d in all_diagnoses if d and d != 'nan']

# 统计每个诊断的数量
diagnosis_counts = Counter(all_diagnoses)

# 计算总数和百分比
total = sum(diagnosis_counts.values())

# 创建DataFrame
result_df = pd.DataFrame([
    {'diagnosis': diag, 'count': count, 'percentage': round(count / total * 100, 2)}
    for diag, count in diagnosis_counts.items()
])

# 按count降序排序
result_df = result_df.sort_values('count', ascending=False).reset_index(drop=True)

# 保存结果
output_path = '/root/MELP/src/melp/data_split/mimic-iv-ecg/unique_diagnoses_train_val.csv'
result_df.to_csv(output_path, index=False)

print(f"共发现 {len(diagnosis_counts)} 种不同的诊断（按小写判断）")
print(f"总诊断数量: {total}")
print(f"结果已保存到: {output_path}")
print("\n前20个最常见的诊断:")
print(result_df.head(20).to_string(index=False))
