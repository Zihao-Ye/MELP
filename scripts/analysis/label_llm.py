import os

from openai import OpenAI
from tqdm import tqdm

import pandas as pd
import json
import time

MODEL = "gpt-5.1"
# MODEL = "deepseek-v3"
DATA_PATH = "./HUAXI_V1_18_info.csv"
TEST_DATA_PATH = "./test.csv"
LABEL_DATA_PATH = "./HUAXI_V1_18_info_label.csv"
ERROR_LOG_PATH = "./label_llm_errors.log"
client = OpenAI(
    api_key="sk-inhKk4zcAMk2FGRVoK0z0zOyupHJti9jXZpYgGAv5At7xMui",
    base_url="https://api.chatanywhere.tech/v1"
)


# ==================== 从文件加载 Prompt 模板 ====================

def load_prompt_template(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


# 预加载模板（启动时读一次）
LABEL_SYSTEM_PROMPT_TEMPLATE = load_prompt_template("./prompt_templates/label_system_prompt.txt")


def load_diagnosis(path: str, column: str = "diagnosis") -> list:
    """
        从 CSV 文件加载诊断列数据
        :param path: CSV 文件路径
        :param column: 列名，默认 'diagnosis'
        :return: 诊断文本列表（str），保留原始顺序，空值转为空字符串
        """
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"CSV 中未找到列 '{column}'，可用列: {list(df.columns)}")

    # 处理 NaN → 空字符串，并转为 str
    diagnoses = df[column].fillna("").astype(str).tolist()
    return diagnoses


# ===== 函数 2：单次调用 LLM =====
def invoke_llm(diagnosis_text: str, row_idx: int = None) -> str:
    """
    对单条诊断文本发起一次 LLM 请求，返回原始响应字符串
    :param diagnosis_text: 诊断文本
    :param row_idx: 行编号（用于记录异常，从0开始）
    """
    system_prompt = LABEL_SYSTEM_PROMPT_TEMPLATE
    user_prompt = diagnosis_text

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=30,
            response_format={"type": "json_object"}  # 强制 JSON（若模型支持）
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_reason = f"LLM调用失败: {str(e)}"
        print(f"调用失败 (输入: {diagnosis_text[:50]}...): {e}")
        if row_idx is not None:
            log_error(row_idx, error_reason, ERROR_LOG_PATH)
        return ""


# ===== 函数 3：解析 LLM 响应 =====
def parse_llm(response_content: str, row_idx: int = None) -> list:
    """
    从 LLM 返回的字符串中提取结构化结果
    返回: {
        "labels": [1, 3]
    }
    :param response_content: LLM 返回的响应内容
    :param row_idx: 行编号（用于记录异常，从0开始）
    """
    if not response_content:
        # 如果内容为空，说明 invoke_llm 已经处理并记录了错误，这里不重复记录
        return []

    try:
        data = json.loads(response_content)
        labels = data.get("labels", [])
        return labels
    except json.JSONDecodeError as e:
        error_reason = f"JSON解析失败: {str(e)}"
        print(f"JSON 解析失败: {response_content[:100]}...")
        if row_idx is not None:
            log_error(row_idx, error_reason, ERROR_LOG_PATH)
        return []
    except Exception as e:
        error_reason = f"解析异常: {str(e)}"
        print(f"解析异常: {e}")
        if row_idx is not None:
            log_error(row_idx, error_reason, ERROR_LOG_PATH)
        return []


def convert_to_one_hot(labels_list, total_classes=20):
    """
    将标签列表转为 one-hot 编码（list of int）
    规则：
      - labels_list: 如 [1, 12]
      - 若为空列表，则返回 [0,0,...,1]（第20位为1）
    """
    one_hot = [0] * total_classes
    if not labels_list:
        # 空 → 归为第20类（索引19）
        one_hot[19] = 1
    else:
        for label in labels_list:
            if isinstance(label, int) and 1 <= label <= total_classes:
                one_hot[label - 1] = 1
            elif isinstance(label, str) and label.isdigit():
                num = int(label)
                if 1 <= num <= total_classes:
                    one_hot[num - 1] = 1
    return one_hot


def log_error(row_idx: int, error_reason: str, log_path: str = ERROR_LOG_PATH):
    """
    记录异常到日志文件
    :param row_idx: 行编号（从0开始，记录时会+1显示为从1开始）
    :param error_reason: 异常原因
    :param log_path: 日志文件路径
    """
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"行号: {row_idx + 1}, 原因: {error_reason}\n")
    except Exception as e:
        print(f"写入异常日志失败: {e}")


# ===== 主流程：逐行处理（支持断点续传） =====
def main():
    csv_path = DATA_PATH
    output_path = LABEL_DATA_PATH

    # 1. 加载原始数据
    df = pd.read_csv(csv_path)
    diagnosis_list = load_diagnosis(csv_path, column="diagnosis")
    print(f"成功加载 {len(diagnosis_list)} 行数据")
    
    # 初始化异常日志文件（追加模式，保留历史记录）
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总行数: {len(diagnosis_list)}\n")
        f.write(f"{'='*50}\n")

    # 2. 检查输出文件是否存在，确定起始位置
    start_idx = 0
    results = []
    
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path, encoding="utf-8-sig")
            # 由于只有处理完的行才会写入，文件中有多少行就表示处理了多少行
            start_idx = len(existing_df)
            # 加载已处理的结果（用于后续保存）
            label_columns = [str(i) for i in range(1, 21)]
            if all(col in existing_df.columns for col in label_columns):
                results = existing_df[label_columns].values.tolist()
            print(f"检测到已处理 {start_idx} 行，将从第 {start_idx + 1} 行继续处理")
        except Exception as e:
            print(f"读取已存在文件时出错: {e}，将从头开始处理")

    # 3. 逐行处理：每行 = 1 次 LLM 调用
    total = len(diagnosis_list)
    for idx in tqdm(range(start_idx, total), desc="处理进度", initial=start_idx, total=total):
        diag_text = diagnosis_list[idx]

        try:
            # 跳过空行（可选）
            if not diag_text.strip():
                one_hot = [0] * 20
                one_hot[19] = 1  # 空行归为第20类
            else:
                # 调用 LLM（一次请求），传入行号用于异常记录
                raw_response = invoke_llm(diag_text, row_idx=idx)

                # 解析结果，传入行号用于异常记录
                parsed = parse_llm(raw_response, row_idx=idx)

                # 转换为 one-hot 编码（如果解析失败返回空列表，convert_to_one_hot 会将其归为第20类）
                one_hot = convert_to_one_hot(parsed)
            
            results.append(one_hot)

        except Exception as e:
            # 捕获其他未预期的异常，记录并继续处理
            error_reason = f"未预期的异常: {str(e)}"
            log_error(idx, error_reason, ERROR_LOG_PATH)
            # 使用默认值（全0，第20位为1）继续处理
            default_one_hot = [0] * 20
            default_one_hot[19] = 1
            results.append(default_one_hot)
            print(f"第 {idx + 1} 行处理异常: {error_reason}")

        # 4. 每处理完一条记录后立即保存（防止中断丢失数据）
        # 只保存已处理的行，不预先填充未处理的行
        try:
            one_hot_df = pd.DataFrame(
                results,
                columns=[str(i) for i in range(1, 21)]  # 列名: "1", "2", ..., "20"
            )
            # 只取已处理的行对应的原始数据
            processed_df = df.iloc[:len(results)].copy()
            result_df = pd.concat([processed_df, one_hot_df], axis=1)
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")  # utf-8-sig 支持 Excel 中文
        except Exception as e:
            # 保存失败也记录异常
            log_error(idx, f"保存失败: {str(e)}", ERROR_LOG_PATH)
            print(f"第 {idx + 1} 行保存异常: {e}")

        # # 控制请求频率（避免限流）
        # time.sleep(0.8)
    
    print(f"处理完成！共处理 {len(results)} 行，结果已保存到 {output_path}")
    
    # 记录完成信息到异常日志
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"完成处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"共处理 {len(results)} 行\n")
        f.write(f"{'='*50}\n\n")


def test_single():
    diag_text = "窦性心律;电轴不偏;不完全性右束支阻滞？"
    idx = 0
    raw_response = invoke_llm(diag_text, row_idx=idx)

    # 解析结果，传入行号用于异常记录
    parsed = parse_llm(raw_response, row_idx=idx)

    # 转换为 one-hot 编码（如果解析失败返回空列表，convert_to_one_hot 会将其归为第20类）
    one_hot = convert_to_one_hot(parsed)

    print(','.join(map(str,one_hot)))


if __name__ == "__main__":
    test_single()
    # main()
