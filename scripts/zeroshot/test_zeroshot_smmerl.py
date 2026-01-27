"""
SMMERL Zero-shot 评估脚本

评估预训练的 SMMERL 模型在各个数据集上的 zero-shot 分类性能。

使用方法:
CUDA_VISIBLE_DEVICES=0 python test_zeroshot_smmerl.py \
    --ckpt_path /path/to/checkpoint.ckpt \
    --test_sets ptbxl_super_class ptbxl_sub_class \
    --scale all \
    --save_results
"""

import os
import json
import torch
import random
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from dateutil import tz
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score

from melp.models.smmerl_model import SMMERLModel
from melp.datasets.finetune_datamodule import ECGDataModule
from melp.paths import RAW_DATA_PATH, PROMPT_PATH, RESULTS_PATH

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: str):
    """加载预训练的 SMMERL 模型"""
    print(f"Loading model from: {ckpt_path}")
    model = SMMERLModel.load_from_checkpoint(ckpt_path)
    model = model.to(device)
    model.eval()

    # 打印模型信息
    print(f"  - ECG Encoder: {model.ecg_encoder_name}")
    print(f"  - Lead Group Strategy: {model.lead_group_strategy}")
    print(f"  - Pool Type: {model.pool_type}")

    return model


def get_dataloader(dataset_name: str, batch_size: int, num_workers: int):
    """获取测试数据加载器"""
    print(f"\nLoading dataset: {dataset_name}")

    dm = ECGDataModule(
        dataset_dir=str(RAW_DATA_PATH),
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        train_data_pct=1
    )
    test_loader = dm.test_dataloader()
    print(f"  - Test samples: {len(test_loader.dataset)}")
    print(f"  - Classes: {test_loader.dataset.labels_name}")

    return test_loader


@torch.no_grad()
def get_class_embeddings(model, class_names: List[str], mode: str = 'concat'):
    """
    计算类别的文本嵌入

    Args:
        model: SMMERL 模型
        class_names: 类别名称列表
        mode: 多尺度融合模式 ('concat', 'mean', 'rhythm')

    Returns:
        zeroshot_weights: (embed_dim, num_classes) 文本嵌入矩阵
    """
    model.eval()
    zeroshot_weights = []

    for text in tqdm(class_names, desc='Computing class embeddings'):
        text = text.lower()
        texts = [text]

        # 获取文本嵌入
        class_embedding = model.get_text_emb(texts, mode=mode)

        # 归一化
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        class_embedding = class_embedding.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()

        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


@torch.no_grad()
def get_ecg_predictions(model, loader, zeroshot_weights, mode: str = 'concat'):
    """
    计算 ECG 的预测结果

    Args:
        model: SMMERL 模型
        loader: 数据加载器
        zeroshot_weights: 类别文本嵌入
        mode: 多尺度融合模式 ('concat', 'mean', 'rhythm')

    Returns:
        predictions: (N, num_classes) 预测分数
    """
    model.eval()
    y_pred = []

    for batch in tqdm(loader, desc='Computing ECG embeddings'):
        ecg = batch['ecg'].to(device)

        # 获取 ECG 嵌入
        ecg_emb = model.ext_ecg_emb(ecg, mode=mode)
        ecg_emb = ecg_emb / ecg_emb.norm(dim=-1, keepdim=True)

        # 计算相似度
        logits = ecg_emb @ zeroshot_weights
        y_pred.append(logits.cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    return y_pred


def compute_AUCs(gt, pred, n_class):
    """计算每个类别的 AUROC"""
    AUROCs = []
    for i in range(n_class):
        if len(np.unique(gt[:, i])) == 1:
            # 如果只有一个类别，跳过
            AUROCs.append(np.nan)
            continue
        AUROCs.append(roc_auc_score(gt[:, i], pred[:, i], average='macro', multi_class='ovo'))
    return AUROCs


def compute_metrics(gt, pred, class_names):
    """
    计算 AUROC, F1, ACC 指标

    Returns:
        res_dict: 包含每个类别指标的字典
    """
    n_class = len(class_names)

    # AUROC
    AUROCs = compute_AUCs(gt, pred, n_class)
    AUROCs = [i * 100 if not np.isnan(i) else np.nan for i in AUROCs]

    # F1 和 ACC
    max_f1s = []
    accs = []
    for i in range(n_class):
        gt_i = gt[:, i]
        pred_i = pred[:, i]

        if len(np.unique(gt_i)) == 1:
            max_f1s.append(np.nan)
            accs.append(np.nan)
            continue

        precision, recall, thresholds = precision_recall_curve(gt_i, pred_i)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5

        max_f1s.append(max_f1 * 100)
        accs.append(accuracy_score(gt_i, pred_i > max_f1_thresh) * 100)

    # 构建结果字典
    res_dict = {}
    for i, name in enumerate(class_names):
        res_dict[f'AUROC_{name}'] = AUROCs[i]
        res_dict[f'F1_{name}'] = max_f1s[i]
        res_dict[f'ACC_{name}'] = accs[i]

    return res_dict


def compute_summary_metrics(res_dict, class_names, metrics=['AUROC', 'F1', 'ACC']):
    """计算汇总指标（宏平均）"""
    summary = {}
    for metric in metrics:
        values = [res_dict[f'{metric}_{name}'] for name in class_names]
        values = [v for v in values if not np.isnan(v)]
        summary[metric] = np.mean(values) if values else np.nan
    return summary


def bootstrap_ci(gt, pred, class_names, n_bootstrap=1000, confidence_level=0.95):
    """Bootstrap 置信区间估计"""
    num_samples = len(gt)
    all_results = []

    for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
        # 有放回采样
        indices = np.random.choice(num_samples, num_samples, replace=True)
        gt_sample = gt[indices]
        pred_sample = pred[indices]

        try:
            res_dict = compute_metrics(gt_sample, pred_sample, class_names)
            summary = compute_summary_metrics(res_dict, class_names)
            all_results.append(summary)
        except Exception as e:
            continue

    if not all_results:
        return None

    boot_df = pd.DataFrame(all_results)
    alpha = 1 - confidence_level

    ci_results = {}
    for metric in boot_df.columns:
        values = boot_df[metric].dropna()
        if len(values) > 0:
            ci_results[metric] = {
                'mean': values.mean(),
                'lower': values.quantile(alpha / 2),
                'upper': values.quantile(1 - alpha / 2)
            }

    return ci_results


@torch.no_grad()
def zeroshot_eval(
    model,
    test_loader,
    results_dir: str,
    mode: str = 'concat',
    compute_ci: bool = False,
    save_results: bool = False
):
    """
    Zero-shot 评估

    Args:
        model: SMMERL 模型
        test_loader: 测试数据加载器
        results_dir: 结果保存目录
        mode: 多尺度融合模式 ('concat', 'mean', 'rhythm')
        compute_ci: 是否计算置信区间
        save_results: 是否保存结果
    """
    # 加载 prompt
    with open(PROMPT_PATH, 'r') as f:
        prompt_dict = json.load(f)

    class_names = test_loader.dataset.labels_name
    target_prompts = [prompt_dict[name] for name in class_names]

    # 获取真实标签
    gt = test_loader.dataset.labels

    # 计算类别嵌入
    zeroshot_weights = get_class_embeddings(model, target_prompts, mode)

    # 计算预测
    pred = get_ecg_predictions(model, test_loader, zeroshot_weights, mode)

    # 计算指标
    res_dict = compute_metrics(gt, pred, class_names)
    summary = compute_summary_metrics(res_dict, class_names)

    # 打印结果
    print("\n" + "=" * 60)
    print("Zero-shot Evaluation Results")
    print("=" * 60)
    print(f"  AUROC: {summary['AUROC']:.2f}%")
    print(f"  F1:    {summary['F1']:.2f}%")
    print(f"  ACC:   {summary['ACC']:.2f}%")

    # 每个类别的结果
    print("\nPer-class AUROC:")
    for name in class_names:
        auroc = res_dict[f'AUROC_{name}']
        if not np.isnan(auroc):
            print(f"  {name}: {auroc:.2f}%")

    # 保存结果
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

        # 保存详细结果
        results_df = pd.DataFrame([res_dict])
        results_df['AUROC_mean'] = summary['AUROC']
        results_df['F1_mean'] = summary['F1']
        results_df['ACC_mean'] = summary['ACC']
        results_df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
        print(f"\nResults saved to: {results_dir}/results.csv")

    # Bootstrap 置信区间
    if compute_ci:
        print("\nComputing 95% confidence intervals...")
        ci_results = bootstrap_ci(gt, pred, class_names)

        if ci_results:
            print("\nMetrics with 95% CI:")
            for metric, values in ci_results.items():
                print(f"  {metric}: {values['mean']:.2f}% [{values['lower']:.2f}%, {values['upper']:.2f}%]")

            if save_results:
                ci_df = pd.DataFrame(ci_results).T
                ci_df.to_csv(os.path.join(results_dir, 'results_ci.csv'))

    return summary


def main(args):
    # 加载模型
    model = load_model(args.ckpt_path)

    print(f"\nMode: {args.mode}")

    # 生成带时间戳的实验目录名
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    exp_name = f"zeroshot_smmerl_{model.ecg_encoder_name}_{extension}"
    exp_dir = RESULTS_PATH / exp_name
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\nExperiment directory: {exp_dir}")

    # 汇总结果
    all_results = []

    # 对每个数据集进行评估
    for dataset_name in args.test_sets:
        print("\n" + "=" * 60)
        print(f"Evaluating on: {dataset_name}")
        print("=" * 60)

        # 获取数据加载器
        test_loader = get_dataloader(dataset_name, args.batch_size, args.num_workers)

        # 结果目录（每个数据集一个子目录）
        results_dir = str(exp_dir / dataset_name)

        # 评估
        summary = zeroshot_eval(
            model=model,
            test_loader=test_loader,
            results_dir=results_dir,
            mode=args.mode,
            compute_ci=args.compute_ci,
            save_results=args.save_results
        )

        summary['dataset'] = dataset_name
        all_results.append(summary)

    # 打印汇总
    print("\n" + "=" * 60)
    print("Summary Across All Datasets")
    print("=" * 60)

    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))

    # 计算平均
    print(f"\nOverall Mean AUROC: {summary_df['AUROC'].mean():.2f}%")
    print(f"Overall Mean F1:    {summary_df['F1'].mean():.2f}%")
    print(f"Overall Mean ACC:   {summary_df['ACC'].mean():.2f}%")

    # 保存汇总结果
    if args.save_results:
        summary_path = str(exp_dir / "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SMMERL Zero-shot Evaluation")

    # 模型参数
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to pretrained checkpoint')

    # 数据参数
    parser.add_argument('--test_sets', type=str, nargs='+',
                        default=["ptbxl_super_class", "ptbxl_sub_class",
                                 "ptbxl_form", "ptbxl_rhythm",
                                 "icbeb", "chapman"],
                        help='List of datasets to evaluate')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    # 评估参数
    parser.add_argument('--mode', type=str, default='concat',
                        choices=['concat', 'mean', 'rhythm'],
                        help='Multi-scale fusion mode')
    parser.add_argument('--compute_ci', action='store_true',
                        help='Compute 95%% confidence interval via bootstrap')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to CSV files')

    args = parser.parse_args()
    main(args)




