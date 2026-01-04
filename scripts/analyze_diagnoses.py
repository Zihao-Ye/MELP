"""
统计MIMIC-IV-ECG数据集中独立的诊断描述

功能：
1. 提取所有report_0到report_17的值
2. 去重并统计频率
3. 输出到CSV文件供后续分类使用

使用方法：
python scripts/analyze_diagnoses.py
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import argparse


def analyze_diagnoses(csv_path: str, output_path: str = None):
    """
    分析诊断描述

    Args:
        csv_path: 输入CSV文件路径
        output_path: 输出文件路径（可选）
    """
    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"总样本数: {len(df)}")

    # 提取所有report列
    report_cols = [f"report_{i}" for i in range(18)]
    existing_cols = [col for col in report_cols if col in df.columns]
    print(f"找到 {len(existing_cols)} 个report列: {existing_cols}")

    # 统计所有诊断描述
    all_diagnoses = []
    for col in existing_cols:
        values = df[col].dropna().astype(str).tolist()
        # 过滤空字符串
        values = [v.strip() for v in values if v.strip() and v.strip().lower() != 'nan']
        all_diagnoses.extend(values)

    print(f"总诊断描述数（含重复）: {len(all_diagnoses)}")

    # 统计频率
    counter = Counter(all_diagnoses)
    print(f"独立诊断描述数（去重）: {len(counter)}")

    # 转换为DataFrame
    result_df = pd.DataFrame([
        {"diagnosis": k, "count": v}
        for k, v in counter.most_common()
    ])

    # 添加频率百分比
    total = result_df["count"].sum()
    result_df["percentage"] = (result_df["count"] / total * 100).round(2)

    # 输出结果
    if output_path is None:
        output_path = Path(csv_path).parent / "unique_diagnoses.csv"

    result_df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    # 打印统计摘要
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    print(f"独立诊断描述数: {len(result_df)}")
    print(f"总出现次数: {total}")
    print(f"\nTop 30 最常见的诊断描述:")
    print("-"*60)
    for i, row in result_df.head(30).iterrows():
        print(f"{i+1:3d}. [{row['count']:6d}] ({row['percentage']:5.2f}%) {row['diagnosis']}")

    # 按类型初步分析
    print("\n" + "="*60)
    print("初步分类分析（基于关键词）")
    print("="*60)

    # 定义关键词模式
    patterns = {
        "rhythm（节律）": ["rhythm", "tachycardia", "bradycardia", "fibrillation", "flutter", "arrhythmia"],
        "wave（波形）": ["wave", "ST", "QRS", "QT", "PR", "voltage", "axis", "progression"],
        "beat（心拍/传导）": ["PVC", "PAC", "block", "conduction", "escape", "ectopic", "extrasystole"],
        "structure（结构）": ["hypertrophy", "enlargement", "abnormality", "dilation"],
        "infarct（梗死/缺血）": ["infarct", "ischemia", "injury", "necrosis"],
        "conclusion（结论）": ["Normal ECG", "Borderline ECG", "Abnormal ECG", "Otherwise normal"]
    }

    categorized = {k: [] for k in patterns}
    uncategorized = []

    for diag in result_df["diagnosis"].tolist():
        matched = False
        for category, keywords in patterns.items():
            if any(kw.lower() in diag.lower() for kw in keywords):
                categorized[category].append(diag)
                matched = True
                break
        if not matched:
            uncategorized.append(diag)

    for category, items in categorized.items():
        print(f"\n{category}: {len(items)} 种")
        for item in items[:5]:
            print(f"  - {item}")
        if len(items) > 5:
            print(f"  ... 等 {len(items)} 种")

    print(f"\n未分类: {len(uncategorized)} 种")
    for item in uncategorized[:10]:
        print(f"  - {item}")
    if len(uncategorized) > 10:
        print(f"  ... 等 {len(uncategorized)} 种")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计MIMIC-IV-ECG诊断描述")
    parser.add_argument("--input", type=str,
                        default="/root/MELP/src/melp/data_split/mimic-iv-ecg/train.csv",
                        help="输入CSV文件路径")
    parser.add_argument("--output", type=str,
                        default="/root/MELP/src/melp/data_split/mimic-iv-ecg/unique_diagnoses.csv",
                        help="输出CSV文件路径")

    args = parser.parse_args()

    analyze_diagnoses(args.input, args.output)
