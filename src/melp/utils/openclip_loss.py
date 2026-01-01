'''
Borrow from open_clip: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/main.py
'''
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=True,
            gather_with_grad=True,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, return_gather_features=False):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
            
            if return_gather_features:
                return logits_per_image, logits_per_text, all_image_features, all_text_features
            else:
                return logits_per_image, logits_per_text

        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
            return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class SoftClipLoss(nn.Module):
    """
    Soft-label contrastive learning loss for ECG-text pairs.

    Uses pre-computed text similarity matrix (from medical LM embeddings like MedCPT)
    to create soft labels, allowing partial positives for samples with overlapping diagnoses.

    This design choice is deliberate:
    - ClipLoss uses projected features (proj_text_emb) for alignment learning
    - Soft labels use medical LM embeddings for stable semantic similarity
    - This separation provides stable supervision while allowing feature learning

    Args:
        similarity_threshold: Threshold for considering samples as soft positives (default: 0.3)
        soft_positive_weight: Weight multiplier for soft positives (default: 0.5)
        local_loss: Whether to use local loss (only compute loss on local ECG features)
        gather_with_grad: Whether to gather features with gradient
        cache_labels: Not used (kept for API compatibility)
        rank: Current GPU rank
        world_size: Total number of GPUs
        use_horovod: Whether to use Horovod
    """

    def __init__(
            self,
            similarity_threshold=0.3,
            soft_positive_weight=0.5,
            local_loss=True,
            gather_with_grad=True,
            cache_labels=False,  # unused but kept for compatibility
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.soft_positive_weight = soft_positive_weight
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def build_soft_labels(self, text_sim_matrix, device, num_logits):
        """
        Build soft labels for contrastive loss.

        Args:
            text_sim_matrix: [num_queries, num_classes]
                - For image->text: [local_batch, global_batch]
                - For text->image: [local_batch, global_batch] (after transpose logic handled outside)
            device: target device
            num_logits: local batch size (used to compute hard positive offset)

        Returns:
            soft_labels: same shape as text_sim_matrix
                - Hard positive: 1.0
                - Soft positive (sim > threshold): soft_positive_weight * sim
                - Others: 0.0
        """
        num_queries, num_classes = text_sim_matrix.shape

        # Initialize soft labels
        soft_labels = torch.zeros_like(text_sim_matrix, device=device)

        # Compute hard positive indices (same logic as ClipLoss.get_ground_truth)
        row_indices = torch.arange(num_queries, device=device)
        if self.world_size > 1 and self.local_loss:
            col_indices = row_indices + num_logits * self.rank
        else:
            col_indices = row_indices  # global mode or single GPU

        # Clamp col_indices to valid range (in case of last batch mismatch, though rare)
        col_indices = torch.clamp(col_indices, 0, num_classes - 1)

        # Set hard positives
        soft_labels[row_indices, col_indices] = 1.0

        # Create mask for non-hard positions
        diag_mask = torch.zeros_like(soft_labels, dtype=torch.bool)
        diag_mask[row_indices, col_indices] = True

        # Soft positives: similarity >= threshold AND not hard positive
        soft_mask = (text_sim_matrix >= self.similarity_threshold) & (~diag_mask)
        soft_labels[soft_mask] = self.soft_positive_weight * text_sim_matrix[soft_mask]

        return soft_labels

    def gather_text_similarity(self, text_sim_matrix):
        """
        Gather text similarity matrix from all GPUs.

        Args:
            text_sim_matrix: Local similarity matrix [local_batch, local_batch]

        Returns:
            Gathered similarity matrix:
                - For local_loss: [local_batch, global_batch]
                - For global_loss: [global_batch, global_batch]
        """
        if self.world_size == 1:
            return text_sim_matrix

        assert has_distributed, 'torch.distributed did not import correctly'

        # Gather along column (text) dimension
        gathered_sim = [torch.zeros_like(text_sim_matrix) for _ in range(self.world_size)]
        dist.all_gather(gathered_sim, text_sim_matrix)

        # Concatenate along column dimension: [local_batch, global_batch]
        all_sim = torch.cat(gathered_sim, dim=1)

        # If global loss, also gather along row dimension
        if not self.local_loss:
            gathered_rows = [torch.zeros_like(all_sim) for _ in range(self.world_size)]
            dist.all_gather(gathered_rows, all_sim)
            all_sim = torch.cat(gathered_rows, dim=0)  # [global_batch, global_batch]

        return all_sim

    def get_logits(self, image_features, text_features, logit_scale):
        """Compute logits with multi-GPU gathering."""
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def soft_cross_entropy(self, logits, soft_labels):
        """
        Compute soft cross-entropy loss WITHOUT normalizing soft_labels.

        This preserves the relative importance of hard positives (weight=1.0)
        vs soft positives (weight=soft_positive_weight * sim), providing stronger
        supervision signal compared to normalized version.
        """
        log_probs = F.log_softmax(logits, dim=1)
        soft_targets = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss

    def forward(self, image_features, text_features, logit_scale, text_sim_matrix, output_dict=False):
        """
        Forward pass for soft-label contrastive learning.

        Args:
            image_features: ECG features [local_batch, embed_dim]
            text_features: Text features [local_batch, embed_dim]
            logit_scale: Temperature parameter
            text_sim_matrix: Pre-computed text similarity [local_batch, local_batch]
                (computed from medical LM embeddings like MedCPT, NOT from proj_text_emb)
            output_dict: Whether to return dict or scalar

        Returns:
            loss or {"contrastive_loss": loss}
        """
        device = image_features.device

        # Get logits (handles multi-GPU gathering)
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        num_logits = logits_per_image.shape[0]  # local batch size

        # Gather text similarity matrix across GPUs
        if self.world_size > 1:
            gathered_text_sim = self.gather_text_similarity(text_sim_matrix)
        else:
            gathered_text_sim = text_sim_matrix

        # Build soft labels from gathered similarity matrix
        # Both image->text and text->image use the same text similarity matrix
        # because image_i corresponds to text_i (text_sim serves as proxy for image_sim)
        soft_labels_image = self.build_soft_labels(gathered_text_sim, device, num_logits)
        soft_labels_text = self.build_soft_labels(gathered_text_sim, device, num_logits)

        # Compute soft cross-entropy loss
        loss_image = self.soft_cross_entropy(logits_per_image, soft_labels_image)
        loss_text = self.soft_cross_entropy(logits_per_text, soft_labels_text)

        total_loss = (loss_image + loss_text) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class LearnableSoftClipLoss(nn.Module):
    """
    可学习软标签对比学习损失

    与 SoftClipLoss 的区别：
    1. 阈值和权重不再是固定超参数，而是从外部传入的可学习参数
    2. 相似度矩阵从外部传入，支持混合可学习相似度

    Args:
        local_loss: Whether to use local loss
        gather_with_grad: Whether to gather features with gradient
        rank: Current GPU rank
        world_size: Total number of GPUs
        use_horovod: Whether to use Horovod
    """

    def __init__(
            self,
            local_loss=True,
            gather_with_grad=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def build_soft_labels(self, sim_matrix, threshold, soft_weight, device, num_logits):
        """
        使用外部传入的可学习参数构建软标签（可微分版本）

        Args:
            sim_matrix: [num_queries, num_classes] 相似度矩阵（可以是混合后的）
            threshold: 可学习阈值参数 (nn.Parameter)
            soft_weight: 可学习权重参数 (nn.Parameter)
            device: target device
            num_logits: local batch size

        Returns:
            soft_labels: same shape as sim_matrix (可微分)
        """
        num_queries, num_classes = sim_matrix.shape

        # 对参数应用 sigmoid 确保在合理范围
        threshold_val = torch.sigmoid(threshold)  # [0, 1]
        soft_weight_val = torch.sigmoid(soft_weight)  # [0, 1]

        # 创建硬正样本的 one-hot 标签
        row_indices = torch.arange(num_queries, device=device)
        if self.world_size > 1 and self.local_loss:
            col_indices = row_indices + num_logits * self.rank
        else:
            col_indices = row_indices

        # Clamp col_indices to valid range
        col_indices = torch.clamp(col_indices, 0, num_classes - 1)

        # 创建硬正样本掩码 (不可微，但只用于标记位置)
        hard_positive_mask = torch.zeros(num_queries, num_classes, device=device)
        hard_positive_mask[row_indices, col_indices] = 1.0

        # 使用可微的方式计算软正样本权重
        # soft_gate: 当 sim > threshold 时趋近于 1，否则趋近于 0
        # 使用 sigmoid 平滑过渡，temperature 控制平滑程度
        temperature = 10.0  # 控制 sigmoid 的陡峭程度
        soft_gate = torch.sigmoid(temperature * (sim_matrix - threshold_val))

        # 软正样本的权重 = soft_gate * soft_weight_val * sim_matrix
        # 排除硬正样本位置 (1 - hard_positive_mask)
        soft_positive_weights = soft_gate * soft_weight_val * sim_matrix * (1 - hard_positive_mask)

        # 最终软标签 = 硬正样本(权重1.0) + 软正样本(可微权重)
        soft_labels = hard_positive_mask + soft_positive_weights

        return soft_labels

    def gather_sim_matrix(self, sim_matrix):
        """
        Gather similarity matrix from all GPUs.
        """
        if self.world_size == 1:
            return sim_matrix

        assert has_distributed, 'torch.distributed did not import correctly'

        # Gather along column dimension
        gathered_sim = [torch.zeros_like(sim_matrix) for _ in range(self.world_size)]
        dist.all_gather(gathered_sim, sim_matrix)

        # Concatenate along column dimension: [local_batch, global_batch]
        all_sim = torch.cat(gathered_sim, dim=1)

        # If global loss, also gather along row dimension
        if not self.local_loss:
            gathered_rows = [torch.zeros_like(all_sim) for _ in range(self.world_size)]
            dist.all_gather(gathered_rows, all_sim)
            all_sim = torch.cat(gathered_rows, dim=0)

        return all_sim

    def get_logits(self, image_features, text_features, logit_scale):
        """Compute logits with multi-GPU gathering."""
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def soft_cross_entropy(self, logits, soft_labels):
        """
        Compute soft cross-entropy loss.
        """
        log_probs = F.log_softmax(logits, dim=1)
        soft_targets = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss

    def forward(self, image_features, text_features, logit_scale,
                sim_matrix, threshold, soft_weight, output_dict=False):
        """
        Forward pass for learnable soft-label contrastive learning.

        Args:
            image_features: ECG features [local_batch, embed_dim]
            text_features: Text features [local_batch, embed_dim]
            logit_scale: Temperature parameter
            sim_matrix: 混合相似度矩阵，已经是 [local_batch, global_batch] 形状
                        (在 compute_hybrid_similarity 中已完成 gather)
            threshold: 可学习阈值参数 (nn.Parameter)
            soft_weight: 可学习软正样本权重 (nn.Parameter)
            output_dict: Whether to return dict or scalar

        Returns:
            loss or {"contrastive_loss": loss}
        """
        device = image_features.device

        # Get logits
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        num_logits = logits_per_image.shape[0]

        # sim_matrix 已经在 compute_hybrid_similarity 中完成了 gather
        # 形状为 [local_batch, global_batch]，可以直接使用

        # Build soft labels using learnable parameters
        soft_labels_image = self.build_soft_labels(sim_matrix, threshold, soft_weight, device, num_logits)
        soft_labels_text = self.build_soft_labels(sim_matrix, threshold, soft_weight, device, num_logits)

        # Compute soft cross-entropy loss
        loss_image = self.soft_cross_entropy(logits_per_image, soft_labels_image)
        loss_text = self.soft_cross_entropy(logits_per_text, soft_labels_text)

        total_loss = (loss_image + loss_text) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class MultiScaleClipLoss(nn.Module):
    """
    多尺度对比学习损失

    对ECG的三个层级(wave/beat/rhythm)分别和文本对齐，
    每个层级有独立的投影空间，避免相互干扰。

    Args:
        embed_dim: ECG/Text encoder输出维度
        proj_dim: 投影后的对齐空间维度
        local_loss: 是否使用local loss
        gather_with_grad: 是否在gather时保留梯度
        rank: 当前GPU rank
        world_size: GPU总数
    """

    def __init__(
            self,
            embed_dim: int = 256,
            proj_dim: int = 256,
            local_loss: bool = True,
            gather_with_grad: bool = True,
            cache_labels: bool = True,
            rank: int = 0,
            world_size: int = 1,
            use_horovod: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        # ECG侧投影头：每个尺度独立
        self.wave_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.beat_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.rhythm_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        # Text侧投影头：对应三个尺度
        self.text_wave_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.text_beat_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.text_rhythm_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        # 可学习的temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 复用ClipLoss的底层实现
        self.clip_loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )

    def forward(
            self,
            wave_feat: torch.Tensor,
            beat_feat: torch.Tensor,
            rhythm_feat: torch.Tensor,
            text_feat: torch.Tensor,
            output_dict: bool = False
    ):
        """
        Args:
            wave_feat: (B, D) 波段级特征 (已池化)
            beat_feat: (B, D) 心拍级特征 (已池化)
            rhythm_feat: (B, D) 节律级特征 (已池化)
            text_feat: (B, D) 文本特征

        Returns:
            total_loss 或 dict包含各层级loss
        """
        logit_scale = self.logit_scale.exp()

        # ECG投影 + L2归一化
        wave_z = F.normalize(self.wave_proj(wave_feat), dim=-1)
        beat_z = F.normalize(self.beat_proj(beat_feat), dim=-1)
        rhythm_z = F.normalize(self.rhythm_proj(rhythm_feat), dim=-1)

        # Text投影 + L2归一化
        text_wave_z = F.normalize(self.text_wave_proj(text_feat), dim=-1)
        text_beat_z = F.normalize(self.text_beat_proj(text_feat), dim=-1)
        text_rhythm_z = F.normalize(self.text_rhythm_proj(text_feat), dim=-1)

        # 三个独立的对比loss
        loss_wave = self.clip_loss(wave_z, text_wave_z, logit_scale)
        loss_beat = self.clip_loss(beat_z, text_beat_z, logit_scale)
        loss_rhythm = self.clip_loss(rhythm_z, text_rhythm_z, logit_scale)

        total_loss = loss_wave + loss_beat + loss_rhythm

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "loss_wave": loss_wave,
                "loss_beat": loss_beat,
                "loss_rhythm": loss_rhythm,
            }
        return total_loss

    def get_ecg_embedding(
            self,
            wave_feat: torch.Tensor,
            beat_feat: torch.Tensor,
            rhythm_feat: torch.Tensor,
            mode: str = 'concat'
    ) -> torch.Tensor:
        """
        推理时获取ECG的最终表示

        Args:
            wave_feat, beat_feat, rhythm_feat: (B, D) 各层级特征
            mode: 'concat' 拼接, 'mean' 平均, 'rhythm' 只用rhythm

        Returns:
            (B, proj_dim) 或 (B, 3*proj_dim) 取决于mode
        """
        wave_z = F.normalize(self.wave_proj(wave_feat), dim=-1)
        beat_z = F.normalize(self.beat_proj(beat_feat), dim=-1)
        rhythm_z = F.normalize(self.rhythm_proj(rhythm_feat), dim=-1)

        if mode == 'concat':
            return torch.cat([wave_z, beat_z, rhythm_z], dim=-1)  # (B, 3D)
        elif mode == 'mean':
            return (wave_z + beat_z + rhythm_z) / 3  # (B, D)
        elif mode == 'rhythm':
            return rhythm_z  # (B, D)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_text_embedding(
            self,
            text_feat: torch.Tensor,
            mode: str = 'concat'
    ) -> torch.Tensor:
        """
        推理时获取Text的最终表示（需要和get_ecg_embedding的mode对应）
        """
        text_wave_z = F.normalize(self.text_wave_proj(text_feat), dim=-1)
        text_beat_z = F.normalize(self.text_beat_proj(text_feat), dim=-1)
        text_rhythm_z = F.normalize(self.text_rhythm_proj(text_feat), dim=-1)

        if mode == 'concat':
            return torch.cat([text_wave_z, text_beat_z, text_rhythm_z], dim=-1)
        elif mode == 'mean':
            return (text_wave_z + text_beat_z + text_rhythm_z) / 3
        elif mode == 'rhythm':
            return text_rhythm_z
        else:
            raise ValueError(f"Unknown mode: {mode}")


class SemanticEnhancedClipLoss(nn.Module):
    """
    语义增强的对比学习损失

    特点：
    1. 使用detach后的文本嵌入计算batch内语义相似度
    2. 基于语义相似度构建软标签，降低高相似负样本的惩罚
    3. 支持可学习的阈值和权重参数
    4. 软标签构建使用可微分的sigmoid gate

    软标签策略：
    - 硬正样本（对角线）：权重 = 1.0
    - 软正样本（sim > threshold）：权重 = soft_weight * sim
    - 负样本：权重 = 0.0
    """

    def __init__(
        self,
        # 可学习参数初始值
        init_threshold: float = 3.0,       # sigmoid后 ≈ 0.95
        init_soft_weight: float = -3.0,    # sigmoid后 ≈ 0.05
        # 分布式参数
        local_loss: bool = True,
        gather_with_grad: bool = True,
        cache_labels: bool = True,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # 可学习参数
        self.learnable_threshold = nn.Parameter(torch.tensor(init_threshold))
        self.learnable_soft_weight = nn.Parameter(torch.tensor(init_soft_weight))

        # 可学习的temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_semantic_similarity(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        计算batch内语义相似度矩阵（使用detach保证稳定性）

        Args:
            text_emb: (B, D) 文本嵌入（来自文本编码器，投影前）

        Returns:
            sim_matrix: (B, B) 相似度矩阵，值域[-1, 1]
        """
        with torch.no_grad():
            text_emb_detached = text_emb.detach()
            text_emb_norm = F.normalize(text_emb_detached, dim=-1)
            sim_matrix = text_emb_norm @ text_emb_norm.T
        return sim_matrix

    def gather_sim_matrix(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """
        Gather语义相似度矩阵

        Args:
            sim_matrix: (local_B, local_B) 本地相似度矩阵

        Returns:
            gathered_sim: (local_B, global_B) for local_loss
        """
        if self.world_size == 1:
            return sim_matrix

        assert has_distributed, 'torch.distributed did not import correctly'

        # Gather along column dimension
        gathered_sim = [torch.zeros_like(sim_matrix) for _ in range(self.world_size)]
        dist.all_gather(gathered_sim, sim_matrix)

        # Concatenate: [local_B, global_B]
        all_sim = torch.cat(gathered_sim, dim=1)

        if not self.local_loss:
            # Global loss: also gather along row dimension
            gathered_rows = [torch.zeros_like(all_sim) for _ in range(self.world_size)]
            dist.all_gather(gathered_rows, all_sim)
            all_sim = torch.cat(gathered_rows, dim=0)

        return all_sim

    def build_soft_labels(
        self,
        sim_matrix: torch.Tensor,
        num_logits: int,
        threshold: torch.Tensor,
        soft_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        基于语义相似度构建软标签（可微分版本）

        Args:
            sim_matrix: (local_B, global_B) 语义相似度矩阵
            num_logits: local batch size
            threshold: 可学习阈值（已sigmoid）
            soft_weight: 可学习软权重（已sigmoid）

        Returns:
            soft_labels: (local_B, global_B)
        """
        device = sim_matrix.device
        num_queries, num_classes = sim_matrix.shape

        # 硬正样本位置（对角线）
        row_indices = torch.arange(num_queries, device=device)
        if self.world_size > 1 and self.local_loss:
            col_indices = row_indices + num_logits * self.rank
        else:
            col_indices = row_indices
        col_indices = torch.clamp(col_indices, 0, num_classes - 1)

        # 硬正样本掩码
        hard_positive_mask = torch.zeros(num_queries, num_classes, device=device)
        hard_positive_mask[row_indices, col_indices] = 1.0

        # 软正样本：使用sigmoid实现可微分的阈值判断
        temperature = 10.0  # 控制sigmoid的陡峭程度
        soft_gate = torch.sigmoid(temperature * (sim_matrix - threshold))

        # 软正样本权重 = gate * soft_weight * similarity
        # 排除硬正样本位置
        soft_positive_weights = soft_gate * soft_weight * sim_matrix * (1 - hard_positive_mask)

        # 最终软标签 = 硬正样本 + 软正样本
        soft_labels = hard_positive_mask + soft_positive_weights

        return soft_labels

    def soft_cross_entropy(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        """
        计算软交叉熵损失
        """
        log_probs = F.log_softmax(logits, dim=1)
        # 归一化软标签
        soft_targets = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss

    def get_logits(self, ecg_features, text_features, logit_scale):
        """计算logits（支持多GPU）"""
        if self.world_size > 1:
            all_ecg_features, all_text_features = gather_features(
                ecg_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_ecg = logit_scale * ecg_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_ecg_features.T
            else:
                logits_per_ecg = logit_scale * all_ecg_features @ all_text_features.T
                logits_per_text = logits_per_ecg.T
        else:
            logits_per_ecg = logit_scale * ecg_features @ text_features.T
            logits_per_text = logit_scale * text_features @ ecg_features.T

        return logits_per_ecg, logits_per_text

    def forward(
        self,
        ecg_features: torch.Tensor,
        text_features: torch.Tensor,
        text_emb: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        output_dict: bool = False
    ):
        """
        Args:
            ecg_features: (B, D) 投影后的ECG特征（已L2归一化）
            text_features: (B, D) 投影后的文本特征（已L2归一化）
            text_emb: (B, D') 文本编码器原始输出（用于计算语义相似度）
            logit_scale: 可选，外部传入的temperature，默认使用内部参数
            output_dict: 是否返回字典

        Returns:
            loss or dict
        """
        if logit_scale is None:
            logit_scale = self.logit_scale.exp()

        # 1. 计算语义相似度矩阵（detach）
        local_sim_matrix = self.compute_semantic_similarity(text_emb)

        # 2. Gather相似度矩阵
        gathered_sim = self.gather_sim_matrix(local_sim_matrix)

        # 3. 计算logits
        logits_per_ecg, logits_per_text = self.get_logits(ecg_features, text_features, logit_scale)

        num_logits = logits_per_ecg.shape[0]

        # 4. 应用sigmoid到可学习参数
        threshold = torch.sigmoid(self.learnable_threshold)
        soft_weight = torch.sigmoid(self.learnable_soft_weight)

        # 5. 构建软标签
        soft_labels = self.build_soft_labels(gathered_sim, num_logits, threshold, soft_weight)

        # 6. 计算双向loss
        loss_ecg = self.soft_cross_entropy(logits_per_ecg, soft_labels)
        loss_text = self.soft_cross_entropy(logits_per_text, soft_labels)

        total_loss = (loss_ecg + loss_text) / 2

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "loss_ecg2text": loss_ecg,
                "loss_text2ecg": loss_text,
                "threshold": threshold.detach(),
                "soft_weight": soft_weight.detach(),
            }
        return total_loss


class MultiScaleSemanticEnhancedClipLoss(nn.Module):
    """
    多尺度语义增强对比损失

    特点：
    1. 三个尺度(wave/beat/rhythm)各自有独立的投影头和可学习参数
    2. 共享语义相似度矩阵（基于detach后的文本原始嵌入）
    3. 使用可微分的软标签构建

    Args:
        embed_dim: ECG/Text encoder输出维度
        proj_dim: 投影后的对齐空间维度
        init_threshold_*: 各尺度的阈值初始值
        init_soft_weight_*: 各尺度的软权重初始值
    """

    def __init__(
        self,
        embed_dim: int = 256,
        proj_dim: int = 256,
        # 各尺度参数初始值
        init_threshold_wave: float = 3.0,
        init_threshold_beat: float = 3.0,
        init_threshold_rhythm: float = 3.0,
        init_soft_weight_wave: float = -3.0,
        init_soft_weight_beat: float = -3.0,
        init_soft_weight_rhythm: float = -3.0,
        # 分布式参数
        local_loss: bool = True,
        gather_with_grad: bool = True,
        cache_labels: bool = True,
        rank: int = 0,
        world_size: int = 1,
        use_horovod: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # ========== ECG投影头（各尺度独立）==========
        self.wave_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.beat_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.rhythm_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        # ========== Text投影头（各尺度独立）==========
        self.text_wave_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.text_beat_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.text_rhythm_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        # ========== 各尺度的可学习参数（独立）==========
        self.threshold_wave = nn.Parameter(torch.tensor(init_threshold_wave))
        self.soft_weight_wave = nn.Parameter(torch.tensor(init_soft_weight_wave))

        self.threshold_beat = nn.Parameter(torch.tensor(init_threshold_beat))
        self.soft_weight_beat = nn.Parameter(torch.tensor(init_soft_weight_beat))

        self.threshold_rhythm = nn.Parameter(torch.tensor(init_threshold_rhythm))
        self.soft_weight_rhythm = nn.Parameter(torch.tensor(init_soft_weight_rhythm))

        # ========== 共享的temperature ==========
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ========== 底层ClipLoss用于计算logits ==========
        self.clip_loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )

    def compute_semantic_similarity(self, text_emb: torch.Tensor) -> torch.Tensor:
        """计算batch内语义相似度矩阵（detach）"""
        with torch.no_grad():
            text_emb_detached = text_emb.detach()
            text_emb_norm = F.normalize(text_emb_detached, dim=-1)
            sim_matrix = text_emb_norm @ text_emb_norm.T
        return sim_matrix

    def gather_sim_matrix(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """Gather语义相似度矩阵"""
        if self.world_size == 1:
            return sim_matrix

        assert has_distributed, 'torch.distributed did not import correctly'

        gathered_sim = [torch.zeros_like(sim_matrix) for _ in range(self.world_size)]
        dist.all_gather(gathered_sim, sim_matrix)
        all_sim = torch.cat(gathered_sim, dim=1)

        if not self.local_loss:
            gathered_rows = [torch.zeros_like(all_sim) for _ in range(self.world_size)]
            dist.all_gather(gathered_rows, all_sim)
            all_sim = torch.cat(gathered_rows, dim=0)

        return all_sim

    def build_soft_labels(
        self,
        sim_matrix: torch.Tensor,
        num_logits: int,
        threshold: torch.Tensor,
        soft_weight: torch.Tensor
    ) -> torch.Tensor:
        """基于语义相似度构建软标签"""
        device = sim_matrix.device
        num_queries, num_classes = sim_matrix.shape

        # 硬正样本位置
        row_indices = torch.arange(num_queries, device=device)
        if self.world_size > 1 and self.local_loss:
            col_indices = row_indices + num_logits * self.rank
        else:
            col_indices = row_indices
        col_indices = torch.clamp(col_indices, 0, num_classes - 1)

        # 硬正样本掩码
        hard_positive_mask = torch.zeros(num_queries, num_classes, device=device)
        hard_positive_mask[row_indices, col_indices] = 1.0

        # 软正样本
        temperature = 10.0
        soft_gate = torch.sigmoid(temperature * (sim_matrix - threshold))
        soft_positive_weights = soft_gate * soft_weight * sim_matrix * (1 - hard_positive_mask)

        soft_labels = hard_positive_mask + soft_positive_weights
        return soft_labels

    def soft_cross_entropy(self, logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
        """计算软交叉熵损失"""
        log_probs = F.log_softmax(logits, dim=1)
        soft_targets = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss

    def compute_scale_loss(
        self,
        ecg_z: torch.Tensor,
        text_z: torch.Tensor,
        gathered_sim: torch.Tensor,
        threshold: torch.Tensor,
        soft_weight: torch.Tensor,
        logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """计算单个尺度的loss"""
        # 获取logits
        logits_per_ecg, logits_per_text = self.clip_loss.get_logits(
            ecg_z, text_z, logit_scale
        )

        num_logits = logits_per_ecg.shape[0]

        # 应用sigmoid
        threshold_val = torch.sigmoid(threshold)
        soft_weight_val = torch.sigmoid(soft_weight)

        # 构建软标签
        soft_labels = self.build_soft_labels(
            gathered_sim, num_logits, threshold_val, soft_weight_val
        )

        # 计算双向loss
        loss_ecg = self.soft_cross_entropy(logits_per_ecg, soft_labels)
        loss_text = self.soft_cross_entropy(logits_per_text, soft_labels)

        return (loss_ecg + loss_text) / 2

    def forward(
        self,
        wave_feat: torch.Tensor,
        beat_feat: torch.Tensor,
        rhythm_feat: torch.Tensor,
        text_feat: torch.Tensor,
        text_emb: torch.Tensor,
        output_dict: bool = False
    ):
        """
        Args:
            wave_feat: (B, D) 波段级ECG特征
            beat_feat: (B, D) 心拍级ECG特征
            rhythm_feat: (B, D) 节律级ECG特征
            text_feat: (B, D) 投影后的文本特征
            text_emb: (B, D') 文本编码器原始输出（用于计算语义相似度）
            output_dict: 是否返回字典

        Returns:
            loss or dict
        """
        logit_scale = self.logit_scale.exp()

        # 1. 计算共享的语义相似度矩阵
        local_sim_matrix = self.compute_semantic_similarity(text_emb)
        gathered_sim = self.gather_sim_matrix(local_sim_matrix)

        # 2. ECG投影 + L2归一化
        wave_z = F.normalize(self.wave_proj(wave_feat), dim=-1)
        beat_z = F.normalize(self.beat_proj(beat_feat), dim=-1)
        rhythm_z = F.normalize(self.rhythm_proj(rhythm_feat), dim=-1)

        # 3. Text投影 + L2归一化
        text_wave_z = F.normalize(self.text_wave_proj(text_feat), dim=-1)
        text_beat_z = F.normalize(self.text_beat_proj(text_feat), dim=-1)
        text_rhythm_z = F.normalize(self.text_rhythm_proj(text_feat), dim=-1)

        # 4. 更新ClipLoss的分布式参数
        self.clip_loss.rank = self.rank
        self.clip_loss.world_size = self.world_size

        # 5. 计算各尺度loss（使用各自的可学习参数）
        loss_wave = self.compute_scale_loss(
            wave_z, text_wave_z, gathered_sim,
            self.threshold_wave, self.soft_weight_wave, logit_scale
        )
        loss_beat = self.compute_scale_loss(
            beat_z, text_beat_z, gathered_sim,
            self.threshold_beat, self.soft_weight_beat, logit_scale
        )
        loss_rhythm = self.compute_scale_loss(
            rhythm_z, text_rhythm_z, gathered_sim,
            self.threshold_rhythm, self.soft_weight_rhythm, logit_scale
        )

        total_loss = loss_wave + loss_beat + loss_rhythm

        if output_dict:
            return {
                "contrastive_loss": total_loss,
                "loss_wave": loss_wave,
                "loss_beat": loss_beat,
                "loss_rhythm": loss_rhythm,
                # 监控可学习参数
                "threshold_wave": torch.sigmoid(self.threshold_wave).detach(),
                "threshold_beat": torch.sigmoid(self.threshold_beat).detach(),
                "threshold_rhythm": torch.sigmoid(self.threshold_rhythm).detach(),
                "soft_weight_wave": torch.sigmoid(self.soft_weight_wave).detach(),
                "soft_weight_beat": torch.sigmoid(self.soft_weight_beat).detach(),
                "soft_weight_rhythm": torch.sigmoid(self.soft_weight_rhythm).detach(),
            }
        return total_loss

    def get_ecg_embedding(
        self,
        wave_feat: torch.Tensor,
        beat_feat: torch.Tensor,
        rhythm_feat: torch.Tensor,
        mode: str = 'concat'
    ) -> torch.Tensor:
        """
        推理时获取ECG的最终表示

        Args:
            wave_feat, beat_feat, rhythm_feat: (B, D) 各层级特征
            mode: 'concat' 拼接, 'mean' 平均, 'rhythm' 只用rhythm

        Returns:
            (B, proj_dim) 或 (B, 3*proj_dim) 取决于mode
        """
        wave_z = F.normalize(self.wave_proj(wave_feat), dim=-1)
        beat_z = F.normalize(self.beat_proj(beat_feat), dim=-1)
        rhythm_z = F.normalize(self.rhythm_proj(rhythm_feat), dim=-1)

        if mode == 'concat':
            return torch.cat([wave_z, beat_z, rhythm_z], dim=-1)
        elif mode == 'mean':
            return (wave_z + beat_z + rhythm_z) / 3
        elif mode == 'rhythm':
            return rhythm_z
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_text_embedding(
        self,
        text_feat: torch.Tensor,
        mode: str = 'concat'
    ) -> torch.Tensor:
        """推理时获取Text的最终表示"""
        text_wave_z = F.normalize(self.text_wave_proj(text_feat), dim=-1)
        text_beat_z = F.normalize(self.text_beat_proj(text_feat), dim=-1)
        text_rhythm_z = F.normalize(self.text_rhythm_proj(text_feat), dim=-1)

        if mode == 'concat':
            return torch.cat([text_wave_z, text_beat_z, text_rhythm_z], dim=-1)
        elif mode == 'mean':
            return (text_wave_z + text_beat_z + text_rhythm_z) / 3
        elif mode == 'rhythm':
            return text_rhythm_z
        else:
            raise ValueError(f"Unknown mode: {mode}")


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss