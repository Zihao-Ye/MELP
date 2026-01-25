"""
ECG 诊断尺度分类脚本

使用 LLM 将诊断文本分类到不同的 ECG 尺度类别：
- Local: 波形与心拍级别（ST段变化、T波倒置、Q波、束支阻滞等）
- Global: 节律与患者级别（窦性心律、房颤、心肌梗死、心室肥大等）
- Irrelevant: 无关信息（噪声、伪影、免责声明等）

支持断点续传，每处理一条记录后立即保存。
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ==================== 配置 ====================

# 默认模型
# DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MODEL = "gpt-5.2"

# 数据路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_PATH = PROJECT_ROOT / "src/melp/data_split/mimic-iv-ecg/unique_diagnoses_train_val.csv"
OUTPUT_PATH = SCRIPT_DIR / "output/diagnoses_scale_labeled.csv"
ERROR_LOG_PATH = SCRIPT_DIR / "output/label_scale_errors.log"
PROMPT_PATH = SCRIPT_DIR / "prompt_templates/scale_category_prompt.txt"

# 有效的尺度类别
VALID_CATEGORIES = {"local", "global", "irrelevant"}


# ==================== 工具函数 ====================

def load_prompt_template(filepath: str) -> str:
    """从文件加载 Prompt 模板"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_diagnoses(path: str, column: str = "diagnosis") -> pd.DataFrame:
    """
    从 CSV 文件加载诊断数据

    Args:
        path: CSV 文件路径
        column: 诊断列名

    Returns:
        DataFrame，包含原始数据
    """
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"CSV 中未找到列 '{column}'，可用列: {list(df.columns)}")

    # 处理 NaN
    df[column] = df[column].fillna("").astype(str)
    return df


def log_error(row_idx: int, error_reason: str, log_path: Path):
    """记录异常到日志文件"""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"行号: {row_idx + 1}, 原因: {error_reason}\n")
    except Exception as e:
        print(f"写入异常日志失败: {e}")


# ==================== LLM 调用 ====================

def create_client(api_key: str, base_url: str = None) -> OpenAI:
    """创建 OpenAI 客户端"""
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def invoke_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    diagnosis_text: str,
    row_idx: int = None,
    log_path: Path = None
) -> str:
    """
    对单条诊断文本发起 LLM 请求

    Args:
        client: OpenAI 客户端
        model: 模型名称
        system_prompt: 系统提示词
        diagnosis_text: 诊断文本
        row_idx: 行编号（用于记录异常）
        log_path: 错误日志路径

    Returns:
        LLM 响应字符串
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": diagnosis_text}
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=30,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_reason = f"LLM调用失败: {str(e)}"
        print(f"调用失败 (输入: {diagnosis_text[:50]}...): {e}")
        if row_idx is not None and log_path:
            log_error(row_idx, error_reason, log_path)
        return ""


def parse_llm_response(
    response_content: str,
    row_idx: int = None,
    log_path: Path = None
) -> list:
    """
    解析 LLM 响应，提取 categories 字段

    期望的 LLM 输出格式:
    {
        "categories": ["waveform", "beat"]
    }

    Args:
        response_content: LLM 响应内容
        row_idx: 行编号
        log_path: 错误日志路径

    Returns:
        类别列表，如 ["waveform", "beat"]
    """
    if not response_content:
        return []

    try:
        data = json.loads(response_content)
        categories = data.get("categories", [])

        # 验证类别有效性
        valid_cats = []
        for cat in categories:
            cat_lower = cat.lower().strip()
            if cat_lower in VALID_CATEGORIES:
                valid_cats.append(cat_lower)
            else:
                print(f"警告: 无效类别 '{cat}'，已忽略")

        return valid_cats

    except json.JSONDecodeError as e:
        error_reason = f"JSON解析失败: {str(e)}"
        print(f"JSON 解析失败: {response_content[:100]}...")
        if row_idx is not None and log_path:
            log_error(row_idx, error_reason, log_path)
        return []
    except Exception as e:
        error_reason = f"解析异常: {str(e)}"
        print(f"解析异常: {e}")
        if row_idx is not None and log_path:
            log_error(row_idx, error_reason, log_path)
        return []


def categories_to_columns(categories: list) -> dict:
    """
    将类别列表转换为列值

    Args:
        categories: 类别列表，如 ["waveform", "beat"]

    Returns:
        字典，如 {"waveform": 1, "beat": 1, "rhythm": 0, "ignore": 0}
    """
    result = {cat: 0 for cat in VALID_CATEGORIES}
    for cat in categories:
        if cat in result:
            result[cat] = 1
    return result


# ==================== 主流程 ====================

def process_diagnoses(
    client: OpenAI,
    model: str,
    system_prompt: str,
    df: pd.DataFrame,
    output_path: Path,
    log_path: Path,
    diagnosis_column: str = "diagnosis",
    save_interval: int = 1
):
    """
    处理所有诊断，支持断点续传

    Args:
        client: OpenAI 客户端
        model: 模型名称
        system_prompt: 系统提示词
        df: 原始数据 DataFrame
        output_path: 输出文件路径
        log_path: 错误日志路径
        diagnosis_column: 诊断列名
        save_interval: 保存间隔（每处理多少条保存一次）
    """
    total = len(df)

    # 检查是否有已处理的结果（断点续传）
    start_idx = 0
    result_df = df.copy()

    # 初始化结果列
    for cat in VALID_CATEGORIES:
        if cat not in result_df.columns:
            result_df[cat] = None

    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, encoding="utf-8-sig")
            # 找到最后一个已处理的行
            for cat in VALID_CATEGORIES:
                if cat in existing_df.columns:
                    # 找到第一个 NaN 的位置
                    null_mask = existing_df[cat].isna()
                    if null_mask.any():
                        start_idx = null_mask.idxmax()
                    else:
                        start_idx = len(existing_df)
                    break

            # 加载已处理的结果
            result_df = existing_df.copy()
            print(f"检测到已处理 {start_idx} 行，将从第 {start_idx + 1} 行继续处理")
        except Exception as e:
            print(f"读取已存在文件时出错: {e}，将从头开始处理")

    # 初始化日志
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总行数: {total}, 起始行: {start_idx + 1}\n")
        f.write(f"{'='*50}\n")

    # 逐行处理
    for idx in tqdm(range(start_idx, total), desc="处理进度", initial=start_idx, total=total):
        diag_text = df.iloc[idx][diagnosis_column]

        try:
            if not diag_text.strip():
                # 空行标记为 irrelevant
                categories = ["irrelevant"]
            else:
                # 调用 LLM
                raw_response = invoke_llm(
                    client, model, system_prompt, diag_text,
                    row_idx=idx, log_path=log_path
                )

                # 解析响应
                categories = parse_llm_response(raw_response, row_idx=idx, log_path=log_path)

                # 如果解析失败，标记为 irrelevant
                if not categories:
                    categories = ["irrelevant"]

            # 更新结果
            col_values = categories_to_columns(categories)
            for cat, val in col_values.items():
                result_df.at[idx, cat] = val

        except Exception as e:
            error_reason = f"未预期的异常: {str(e)}"
            log_error(idx, error_reason, log_path)
            print(f"第 {idx + 1} 行处理异常: {error_reason}")

            # 标记为 irrelevant
            for cat in VALID_CATEGORIES:
                result_df.at[idx, cat] = 1 if cat == "irrelevant" else 0

        # 定期保存
        if (idx + 1) % save_interval == 0 or idx == total - 1:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            except Exception as e:
                log_error(idx, f"保存失败: {str(e)}", log_path)
                print(f"第 {idx + 1} 行保存异常: {e}")

    # 记录完成信息
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"完成处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"共处理 {total - start_idx} 行\n")
        f.write(f"{'='*50}\n\n")

    print(f"处理完成！结果已保存到 {output_path}")

    # 打印统计信息
    print_statistics(result_df)


def print_statistics(df: pd.DataFrame):
    """打印分类统计信息"""
    print("\n" + "="*50)
    print("分类统计:")
    print("="*50)

    for cat in VALID_CATEGORIES:
        if cat in df.columns:
            count = df[cat].sum()
            pct = count / len(df) * 100
            print(f"  {cat}: {int(count)} ({pct:.2f}%)")

    # 多尺度统计
    if all(cat in df.columns for cat in ["local", "global"]):
        multi_scale = ((df["local"] == 1) & (df["global"] == 1)).sum()
        print(f"\n  多尺度诊断 (Local+Global): {int(multi_scale)} ({multi_scale/len(df)*100:.2f}%)")

    print("="*50)


def test_single(client: OpenAI, model: str, system_prompt: str, diagnosis: str):
    """测试单条诊断"""
    print(f"诊断文本: {diagnosis}")
    print("-" * 40)

    raw_response = invoke_llm(client, model, system_prompt, diagnosis)
    print(f"LLM 响应: {raw_response}")

    categories = parse_llm_response(raw_response)
    print(f"解析结果: {categories}")

    col_values = categories_to_columns(categories)
    print(f"列值: {col_values}")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="ECG 诊断尺度分类")
    parser.add_argument("--api-key", type=str, default="sk-inhKk4zcAMk2FGRVoK0z0zOyupHJti9jXZpYgGAv5At7xMui", help="OpenAI API Key")
    parser.add_argument("--base-url", type=str, default="https://api.chatanywhere.tech/v1", help="API Base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="输入数据路径")
    parser.add_argument("--output-path", type=str, default=str(OUTPUT_PATH), help="输出路径")
    parser.add_argument("--prompt-path", type=str, default=str(PROMPT_PATH), help="Prompt 模板路径")
    parser.add_argument("--save-interval", type=int, default=1, help="保存间隔")
    parser.add_argument("--test", type=str, default=None, help="测试单条诊断")

    args = parser.parse_args()

    # 创建客户端
    client = create_client(args.api_key, args.base_url)

    # 加载 prompt
    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        print(f"错误: Prompt 文件不存在: {prompt_path}")
        print("请先创建 prompt 模板文件")
        sys.exit(1)

    system_prompt = load_prompt_template(str(prompt_path))

    # 测试模式
    if args.test:
        test_single(client, args.model, system_prompt, args.test)
        return

    # 加载数据
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)

    df = load_diagnoses(str(data_path))
    print(f"成功加载 {len(df)} 行数据")

    # 处理
    process_diagnoses(
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        df=df,
        output_path=Path(args.output_path),
        log_path=ERROR_LOG_PATH,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
