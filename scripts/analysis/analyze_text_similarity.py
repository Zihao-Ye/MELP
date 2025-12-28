"""
Analyze text similarity distribution across the MIMIC-IV-ECG training dataset.

This script computes pairwise text similarity using medical LM embeddings
and provides statistics to help tune similarity_threshold and soft_positive_weight.

Usage:
    # Full analysis (all training samples)
    python analyze_text_similarity.py \
        --dataset_dir /disk1/*/ECG/raw \
        --text_encoder_name ncbi/MedCPT-Query-Encoder

    # Quick test (1000 samples)
    python analyze_text_similarity.py \
        --dataset_dir /disk1/*/ECG/raw \
        --num_samples 1000 \
        --batch_size 32

    # Use different text encoder
    python analyze_text_similarity.py \
        --dataset_dir /disk1/*/ECG/raw \
        --text_encoder_name fuyingw/heart_bert
"""

import argparse
import sys
import os
import random
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightning import seed_everything

from melp.datasets.pretrain_datamodule import ECGTextDataModule
from transformers import AutoTokenizer, AutoModel
from melp.paths import RAW_DATA_PATH

# Global settings (same as main_pretrain.py)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class TextSimilarityAnalyzer:
    """Analyzer for text similarity distribution."""

    def __init__(self, text_encoder_name, device='cuda'):
        self.text_encoder_name = text_encoder_name
        self.device = device

        print(f"Loading text encoder: {text_encoder_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.lm_model = AutoModel.from_pretrained(text_encoder_name).to(device)
        self.lm_model.eval()
        print(f"✓ Text encoder loaded on {device}\n")

    def compute_text_similarity(self, texts):
        """
        Compute pairwise text similarity matrix (same logic as MERL model).

        Args:
            texts: List of text strings [batch_size]

        Returns:
            similarity_matrix: [batch_size, batch_size] cosine similarity
        """
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)

        # Get embeddings
        with torch.no_grad():
            if self.text_encoder_name == "ncbi/MedCPT-Query-Encoder":
                text_embeddings = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).pooler_output
            elif self.text_encoder_name in ["google/flan-t5-small", "google/flan-t5-base"]:
                sequence_output = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state
                eos_mask = input_ids.eq(self.lm_model.config.eos_token_id).bool()
                batch_size, _, hidden_size = sequence_output.shape
                text_embeddings = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
            elif "bert" in self.text_encoder_name.lower():
                text_embeddings = self.lm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).pooler_output
            else:
                raise NotImplementedError(f"Encoder {self.text_encoder_name} not supported")

        # Normalize and compute similarity
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        similarity_matrix = text_embeddings @ text_embeddings.T

        return similarity_matrix.cpu()

    def analyze_dataset(self, dataloader, num_samples=None):
        """
        Analyze text similarity across the dataset.

        Args:
            dataloader: DataLoader for the dataset
            num_samples: Maximum number of samples to process (None for all)

        Returns:
            statistics: Dictionary with similarity statistics
        """
        all_similarities = []
        all_off_diagonal = []
        soft_positive_counts = {}
        batch_soft_counts = {}  # For variance analysis

        total_samples = 0
        if num_samples is None:
            total_batches = len(dataloader)
            print(f"Analyzing ALL batches ({total_batches} total)...")
        else:
            total_batches = min(len(dataloader), (num_samples + dataloader.batch_size - 1) // dataloader.batch_size)
            print(f"Analyzing {total_batches} batches (up to {num_samples} samples)...")

        pbar = tqdm(total=total_batches, desc="Processing", unit="batch")

        for batch_idx, batch in enumerate(dataloader):
            if num_samples and total_samples >= num_samples:
                break

            texts = batch['report']
            batch_size = len(texts)

            # Compute similarity matrix for this batch
            try:
                sim_matrix = self.compute_text_similarity(texts)  # [B, B]
            except Exception as e:
                print(f"\n⚠️  Warning: Skipping batch {batch_idx} due to error: {e}")
                pbar.update(1)
                continue

            # Extract all pairwise similarities
            all_similarities.append(sim_matrix.flatten())

            # Extract off-diagonal (non-self) similarities
            mask = ~torch.eye(batch_size, dtype=torch.bool)
            off_diag = sim_matrix[mask]
            all_off_diagonal.append(off_diag)

            # Count soft positives at different thresholds
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                # Number of soft positives per sample (excluding self)
                off_diag_matrix = off_diag.view(batch_size, batch_size - 1)
                soft_pos_per_sample = (off_diag_matrix > threshold).sum(dim=1).float()
                soft_pos_count = soft_pos_per_sample.mean().item()

                if threshold not in soft_positive_counts:
                    soft_positive_counts[threshold] = []
                    batch_soft_counts[threshold] = []

                soft_positive_counts[threshold].append(soft_pos_count)
                batch_soft_counts[threshold].append(soft_pos_count)

            total_samples += batch_size
            pbar.update(1)

        pbar.close()

        # Concatenate all similarities
        all_similarities = torch.cat(all_similarities).numpy()
        all_off_diagonal = torch.cat(all_off_diagonal).numpy()

        # Compute statistics
        stats = {
            'all': {
                'mean': float(np.mean(all_similarities)),
                'std': float(np.std(all_similarities)),
                'min': float(np.min(all_similarities)),
                'max': float(np.max(all_similarities)),
                'percentiles': {
                    '10%': float(np.percentile(all_similarities, 10)),
                    '25%': float(np.percentile(all_similarities, 25)),
                    '50%': float(np.percentile(all_similarities, 50)),
                    '75%': float(np.percentile(all_similarities, 75)),
                    '90%': float(np.percentile(all_similarities, 90)),
                    '95%': float(np.percentile(all_similarities, 95)),
                    '99%': float(np.percentile(all_similarities, 99)),
                }
            },
            'off_diagonal': {
                'mean': float(np.mean(all_off_diagonal)),
                'std': float(np.std(all_off_diagonal)),
                'min': float(np.min(all_off_diagonal)),
                'max': float(np.max(all_off_diagonal)),
                'percentiles': {
                    '10%': float(np.percentile(all_off_diagonal, 10)),
                    '25%': float(np.percentile(all_off_diagonal, 25)),
                    '50%': float(np.percentile(all_off_diagonal, 50)),
                    '75%': float(np.percentile(all_off_diagonal, 75)),
                    '90%': float(np.percentile(all_off_diagonal, 90)),
                    '95%': float(np.percentile(all_off_diagonal, 95)),
                    '99%': float(np.percentile(all_off_diagonal, 99)),
                }
            },
            'soft_positives_per_sample': {
                f'threshold_{t}': {
                    'mean': float(np.mean(counts)),
                    'std': float(np.std(batch_soft_counts[t])),
                    'min': float(np.min(batch_soft_counts[t])),
                    'max': float(np.max(batch_soft_counts[t])),
                }
                for t, counts in soft_positive_counts.items()
            },
            'total_samples': total_samples,
            # Raw data for plotting (use different keys to avoid overwriting stats)
            'all_similarities_raw': all_similarities,
            'off_diagonal_raw': all_off_diagonal,
            'batch_soft_counts_raw': batch_soft_counts,
        }

        return stats

    def print_statistics(self, stats):
        """Print statistics in a readable format."""
        print("\n" + "="*80)
        print("TEXT SIMILARITY ANALYSIS RESULTS")
        print("="*80)

        print(f"\nTotal samples analyzed: {stats['total_samples']}")

        print("\n" + "-"*80)
        print("ALL PAIRWISE SIMILARITIES (including diagonal/self)")
        print("-"*80)
        print(f"Mean:  {stats['all']['mean']:.4f}")
        print(f"Std:   {stats['all']['std']:.4f}")
        print(f"Range: [{stats['all']['min']:.4f}, {stats['all']['max']:.4f}]")
        print("\nPercentiles:")
        for p, v in stats['all']['percentiles'].items():
            print(f"  {p:>4s}: {v:.4f}")

        print("\n" + "-"*80)
        print("OFF-DIAGONAL SIMILARITIES (excluding diagonal/self) ← KEY METRIC")
        print("-"*80)
        print(f"Mean:  {stats['off_diagonal']['mean']:.4f}")
        print(f"Std:   {stats['off_diagonal']['std']:.4f}")
        print(f"Range: [{stats['off_diagonal']['min']:.4f}, {stats['off_diagonal']['max']:.4f}]")
        print("\nPercentiles:")
        for p, v in stats['off_diagonal']['percentiles'].items():
            print(f"  {p:>4s}: {v:.4f}")

        print("\n" + "-"*80)
        print("SOFT POSITIVES PER SAMPLE (at different thresholds)")
        print("-"*80)
        print(f"{'Threshold':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Variation':<10}")
        print("-"*80)

        for threshold_key in sorted(stats['soft_positives_per_sample'].keys()):
            threshold = float(threshold_key.split('_')[1])
            data = stats['soft_positives_per_sample'][threshold_key]
            variation = "HIGH" if data['std'] > data['mean'] * 0.5 else "MEDIUM" if data['std'] > data['mean'] * 0.3 else "LOW"
            print(f"{threshold:<12.1f} {data['mean']:<8.2f} {data['std']:<8.2f} {data['min']:<8.2f} {data['max']:<8.2f} {variation:<10}")

        print("\n" + "="*80)
        print("RECOMMENDATIONS FOR HYPERPARAMETERS")
        print("="*80)

        # Recommendations based on statistics
        p75_off_diag = stats['off_diagonal']['percentiles']['75%']
        p90_off_diag = stats['off_diagonal']['percentiles']['90%']
        p95_off_diag = stats['off_diagonal']['percentiles']['95%']

        print(f"\n1. SIMILARITY THRESHOLD (controls how many soft positives):")
        print(f"   Current setting in code: 0.30")
        print(f"   Options:")
        print(f"   - Conservative (few soft positives):  threshold = {p90_off_diag:.2f}  (90th percentile)")
        print(f"   - Moderate (balanced):                threshold = {p75_off_diag:.2f}  (75th percentile)")
        print(f"   - Aggressive (many soft positives):   threshold = 0.30  (current)")

        avg_soft_pos_at_03 = stats['soft_positives_per_sample']['threshold_0.3']['mean']
        std_soft_pos_at_03 = stats['soft_positives_per_sample']['threshold_0.3']['std']
        max_soft_pos_at_03 = stats['soft_positives_per_sample']['threshold_0.3']['max']

        print(f"\n2. SOFT POSITIVE WEIGHT (controls impact of soft positives):")
        print(f"   Current setting in code: 0.10")
        print(f"   At threshold=0.3: avg={avg_soft_pos_at_03:.1f}, std={std_soft_pos_at_03:.1f}, max={max_soft_pos_at_03:.1f} soft positives")

        if avg_soft_pos_at_03 > 15:
            print(f"   ⚠️  MANY soft positives! Suggest weight = 0.05 - 0.10")
        elif avg_soft_pos_at_03 > 8:
            print(f"   →  MEDIUM soft positives. Suggest weight = 0.10 - 0.20")
        else:
            print(f"   ✓  FEW soft positives. Weight = 0.20 - 0.50 is acceptable")

        print(f"\n3. EXPECTED LOSS VARIANCE:")
        print(f"   Soft positive std = {std_soft_pos_at_03:.1f} (batch-to-batch variation)")
        if std_soft_pos_at_03 > avg_soft_pos_at_03 * 0.5:
            print(f"   ⚠️  HIGH variance! Loss will fluctuate significantly across batches")
            print(f"   →  Consider: increase threshold OR decrease soft_positive_weight")
        elif std_soft_pos_at_03 > avg_soft_pos_at_03 * 0.3:
            print(f"   →  MEDIUM variance. Some loss fluctuation expected (normal)")
        else:
            print(f"   ✓  LOW variance. Loss should be relatively stable")

        # Loss magnitude estimation
        total_weight_mean = 1.0 + avg_soft_pos_at_03 * 0.1  # assuming weight=0.1
        total_weight_max = 1.0 + max_soft_pos_at_03 * 0.1
        print(f"\n4. EXPECTED SOFT LABEL SUM (indicator of loss magnitude):")
        print(f"   With soft_positive_weight=0.1 and threshold=0.3:")
        print(f"   - Mean total weight per sample: {total_weight_mean:.2f}")
        print(f"   - Max total weight per sample:  {total_weight_max:.2f}")
        print(f"   - Fluctuation ratio: {total_weight_max / total_weight_mean:.2f}x")

        print("\n" + "="*80)
        print("SUGGESTED CONFIGURATIONS (choose one)")
        print("="*80)

        # Configuration 1: Conservative
        print("\nConfig 1 - CONSERVATIVE (stable, fewer soft positives):")
        print(f"  similarity_threshold = {p90_off_diag:.2f}")
        print(f"  soft_positive_weight = 0.20")

        # Configuration 2: Balanced
        print("\nConfig 2 - BALANCED (recommended):")
        print(f"  similarity_threshold = {p75_off_diag:.2f}")
        print(f"  soft_positive_weight = 0.15")

        # Configuration 3: Aggressive
        if avg_soft_pos_at_03 > 10:
            weight = 0.05
        elif avg_soft_pos_at_03 > 5:
            weight = 0.10
        else:
            weight = 0.20
        print("\nConfig 3 - AGGRESSIVE (max soft supervision):")
        print(f"  similarity_threshold = 0.30")
        print(f"  soft_positive_weight = {weight:.2f}")

        print("\n" + "="*80 + "\n")

    def plot_distribution(self, stats, save_path='text_similarity_distribution.png'):
        """Plot similarity distribution with 4 subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle('Text Similarity Distribution Analysis (MIMIC-IV-ECG)',
                     fontsize=16, fontweight='bold')

        # Plot 1: Histogram of all similarities
        ax = axes[0, 0]
        ax.hist(stats['all_similarities_raw'], bins=100, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(stats['all']['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats['all']['mean']:.3f}")
        ax.axvline(1.0, color='green', linestyle=':', linewidth=2, alpha=0.5,
                   label="Self-similarity")
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('All Pairwise Similarities (including self)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Histogram of off-diagonal similarities
        ax = axes[0, 1]
        ax.hist(stats['off_diagonal_raw'], bins=100, alpha=0.7, edgecolor='black', color='coral')
        ax.axvline(stats['off_diagonal']['mean'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {stats['off_diagonal']['mean']:.3f}")
        for threshold, color in [(0.3, 'green'), (0.5, 'blue'), (0.7, 'purple')]:
            pct = np.mean(stats['off_diagonal_raw'] > threshold) * 100
            ax.axvline(threshold, color=color, linestyle=':', linewidth=1.5, alpha=0.6,
                      label=f"Thr={threshold}: {pct:.1f}% above")
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Off-Diagonal Similarities (KEY METRIC)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 3: CDF
        ax = axes[1, 0]
        sorted_sim = np.sort(stats['off_diagonal_raw'])
        cdf = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
        ax.plot(sorted_sim, cdf, linewidth=2.5, color='navy')

        for threshold, color in [(0.3, 'green'), (0.5, 'blue'), (0.7, 'purple')]:
            percentile = np.mean(stats['off_diagonal_raw'] <= threshold) * 100
            ax.axvline(threshold, color=color, linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f"Thr={threshold}: {percentile:.1f}% below")

        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title('CDF of Off-Diagonal Similarities', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 4: Soft positives vs threshold (with variance)
        ax = axes[1, 1]
        thresholds = sorted([float(k.split('_')[1]) for k in stats['soft_positives_per_sample'].keys()])
        means = [stats['soft_positives_per_sample'][f'threshold_{t}']['mean'] for t in thresholds]
        stds = [stats['soft_positives_per_sample'][f'threshold_{t}']['std'] for t in thresholds]

        ax.plot(thresholds, means, marker='o', linewidth=2.5, markersize=8, color='darkgreen',
                label='Mean soft positives')
        ax.fill_between(thresholds,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3, color='lightgreen', label='±1 std (variance)')

        ax.axhline(10, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='10 soft pos')
        ax.axhline(5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='5 soft pos')
        ax.axvline(0.3, color='purple', linestyle=':', alpha=0.5, linewidth=1.5, label='Current thr=0.3')

        ax.set_xlabel('Similarity Threshold', fontsize=11)
        ax.set_ylabel('Avg Soft Positives per Sample', fontsize=11)
        ax.set_title('Soft Positives vs Threshold', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {save_path}")
        plt.close()


def main(args):
    # Set random seed for reproducibility (same as main_pretrain.py)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    seed_everything(args.seed)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*80)
    print("TEXT SIMILARITY ANALYSIS FOR SOFTCLIPLOSS HYPERPARAMETER TUNING")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Dataset: MIMIC-IV-ECG (train split)")
    print(f"Text encoder: {args.text_encoder_name}\n")

    # Initialize analyzer
    analyzer = TextSimilarityAnalyzer(args.text_encoder_name, device=device)

    # Load datamodule
    datamodule = ECGTextDataModule(
        dataset_dir=str(RAW_DATA_PATH),
        dataset_list=['mimic-iv-ecg'],  # Fixed to MIMIC-IV-ECG
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_data_pct=1.0,  # Use full training set
        use_cmsc=False,
        use_rlm=False,
        transforms=None,
        n_views=1,
    )

    dataloader = datamodule.train_dataloader()
    print(f"✓ Dataset loaded: {len(dataloader)} batches\n")

    # Analyze
    stats = analyzer.analyze_dataset(dataloader, num_samples=args.num_samples)

    # Print results
    analyzer.print_statistics(stats)

    # Plot
    analyzer.plot_distribution(stats, save_path=args.output_path)

    # Save raw statistics
    if args.save_stats:
        import json
        # Exclude raw data arrays (only save statistics)
        stats_to_save = {k: v for k, v in stats.items()
                        if not k.endswith('_raw')}
        json_path = args.output_path.replace('.png', '_stats.json')
        with open(json_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        print(f"✓ Statistics saved to: {json_path}")

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze text similarity distribution for SoftClipLoss hyperparameter tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 1000 samples
  python analyze_text_similarity.py --dataset_dir /disk1/*/ECG/raw --num_samples 1000

  # Full analysis (all training samples)
  python analyze_text_similarity.py --dataset_dir /disk1/*/ECG/raw

  # Use different text encoder
  python analyze_text_similarity.py --dataset_dir /disk1/*/ECG/raw --text_encoder_name fuyingw/heart_bert
        """
    )

    # Optional arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--text_encoder_name', type=str,
                       default='ncbi/MedCPT-Query-Encoder',
                       help='Text encoder model name (default: ncbi/MedCPT-Query-Encoder)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for processing (default: 256)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Max samples to analyze (default: None, meaning all). Use 1000-10000 for quick test.')
    parser.add_argument('--output_path', type=str,
                       default='text_similarity_distribution.png',
                       help='Path to save output plot (default: text_similarity_distribution.png)')
    parser.add_argument('--save_stats', action='store_true', default=True,
                       help='Save statistics to JSON (default: True)')

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = Path(args.output_path).parent
    if output_dir != Path('.'):
        output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
