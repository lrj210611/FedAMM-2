import copy

import torch
import torch.nn.functional as F

from utils.fedmass_missing_proto import extract_missing_pattern_prototypes


def _zero_scalar(reference):
    """返回与参考张量同 device/dtype 的 0 标量。"""
    return reference.new_zeros(())


def create_ema_teacher(student_model):
    """
    创建 EMA teacher 模型。

    输入：
    - student_model: 当前客户端的 student 模型。

    输出：
    - teacher_model: 深拷贝后的 teacher 模型。

    作用：
    - teacher 用于 weak view 推理；
    - 参数不参与梯度更新；
    - 默认设置为 eval 模式。
    """
    teacher_model = copy.deepcopy(student_model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


def update_ema_teacher(student_model, teacher_model, ema_decay):
    """
    使用 student 参数更新 teacher 参数。

    输入：
    - student_model: 当前 student 模型。
    - teacher_model: EMA teacher 模型。
    - ema_decay: EMA 动量系数。

    输出：
    - teacher_model: 更新后的 teacher。
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(ema_decay).add_(student_param.data, alpha=1.0 - ema_decay)
        for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
            teacher_buffer.data.copy_(student_buffer.data)
    teacher_model.eval()
    return teacher_model


def get_available_modality_count(mask):
    """
    统计当前样本可用模态数量 |s|。

    输入：
    - mask: shape [B, 4] 或 [4]，bool/0-1 tensor 均可。

    输出：
    - available_count: shape [B] 或标量张量。
    """
    mask_tensor = torch.as_tensor(mask)
    if mask_tensor.dim() == 1:
        return mask_tensor.to(dtype=torch.float32).sum()
    return mask_tensor.to(dtype=torch.float32).sum(dim=1)


def compute_mask_aware_weight(mask):
    """
    计算 teacher 与 anchor 融合权重 omega_s = |s| / 4。

    输入：
    - mask: shape [B, 4] 或 [4]

    输出：
    - omega_s: shape [B] 或标量，范围 [0.25, 1.0]
    """
    available_count = get_available_modality_count(mask)
    return torch.clamp(available_count / 4.0, min=0.25, max=1.0)


def compute_mask_aware_threshold(mask, tau_base, gamma):
    """
    计算缺失模式感知阈值 tau_s = tau_base + gamma * (1 - |s| / 4)。

    输入：
    - mask: shape [B, 4] 或 [4]
    - tau_base: 基础阈值 tau_0
    - gamma: 缺失程度增量

    输出：
    - tau_s: shape [B] 或标量
    """
    available_count = get_available_modality_count(mask).to(dtype=torch.float32)
    return tau_base + gamma * (1.0 - available_count / 4.0)


def _resolve_anchor_bank(global_anchor_bank, device, dtype):
    """解析 anchor bank，兼容 tensor 或 dict 输入。"""
    if global_anchor_bank is None:
        return None, None
    valid_mask = None
    anchors = global_anchor_bank
    if isinstance(global_anchor_bank, dict):
        anchors = global_anchor_bank.get('anchors')
        valid_mask = global_anchor_bank.get('valid_mask')
    if anchors is None:
        return None, None
    anchors = torch.as_tensor(anchors, device=device, dtype=dtype)
    if valid_mask is None:
        valid_mask = torch.any(anchors != 0, dim=1)
    else:
        valid_mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
    return anchors, valid_mask


def compute_anchor_similarity_probs(features, global_anchor_bank, temperature):
    """
    根据 voxel feature 与 global full-modal anchor 的余弦相似度计算 anchor 概率。

    输入：
    - features: shape [B, C_feat, D, H, W]
    - global_anchor_bank: shape [num_classes, C_feat] 或 {'anchors': ..., 'valid_mask': ...}
    - temperature: softmax 温度

    输出：
    - anchor_probs: shape [B, num_classes, D, H, W]

    说明：
    - 如果 global_anchor_bank 为 None，返回 None；
    - 如果 anchor channel 与 feature channel 不匹配，返回 None；
    - 如果没有有效 anchor，返回 None。
    """
    anchors, valid_mask = _resolve_anchor_bank(global_anchor_bank, features.device, features.dtype)
    if anchors is None:
        return None
    if anchors.dim() != 2 or anchors.size(1) != features.size(1):
        return None
    if valid_mask is None or not torch.any(valid_mask):
        return None

    anchors = F.normalize(anchors, dim=1)
    flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, features.size(1))
    flat_features = F.normalize(flat_features, dim=1)
    similarity = torch.matmul(flat_features, anchors.t()) / max(float(temperature), 1e-6)
    if valid_mask is not None:
        similarity[:, ~valid_mask] = -1e4
    probs = F.softmax(similarity, dim=1)
    probs = probs.reshape(features.size(0), features.size(2), features.size(3), features.size(4), -1)
    return probs.permute(0, 4, 1, 2, 3).contiguous()


def refine_pseudo_probs(teacher_probs, anchor_probs, mask):
    """
    融合 teacher prediction 与 anchor prediction。

    输入：
    - teacher_probs: shape [B, num_classes, D, H, W]
    - anchor_probs: shape [B, num_classes, D, H, W] 或 None
    - mask: shape [B, 4]

    输出：
    - refined_probs: shape [B, num_classes, D, H, W]
    """
    if anchor_probs is None:
        return teacher_probs
    omega_s = compute_mask_aware_weight(mask).to(device=teacher_probs.device, dtype=teacher_probs.dtype)
    while omega_s.dim() < teacher_probs.dim():
        omega_s = omega_s.unsqueeze(-1)
    return omega_s * teacher_probs + (1.0 - omega_s) * anchor_probs


def build_pseudo_label_filters(
    refined_probs,
    teacher_probs,
    student_probs,
    anchor_probs,
    mask,
    tau_base,
    gamma,
    consistency_eps,
):
    """
    构造三重伪标签过滤 mask。

    输入：
    - refined_probs: shape [B, num_classes, D, H, W]
    - teacher_probs: shape [B, num_classes, D, H, W]
    - student_probs: shape [B, num_classes, D, H, W]
    - anchor_probs: shape [B, num_classes, D, H, W] 或 None
    - mask: shape [B, 4]
    - tau_base: 基础置信度阈值
    - gamma: mask-aware 阈值增量
    - consistency_eps: teacher-student 一致性阈值

    输出：
    - selected_mask: shape [B, D, H, W]，bool
    - pseudo_labels: shape [B, D, H, W]，long
    - stats: dict，包含 mean_confidence / selected_ratio / anchor_agreement_ratio / consistency_ratio
    """
    max_confidence, pseudo_labels = torch.max(refined_probs, dim=1)
    tau_s = compute_mask_aware_threshold(mask, tau_base, gamma).to(device=refined_probs.device, dtype=refined_probs.dtype)
    while tau_s.dim() < max_confidence.dim():
        tau_s = tau_s.unsqueeze(-1)
    confidence_filter = max_confidence > tau_s

    consistency_distance = torch.mean(torch.abs(teacher_probs - student_probs), dim=1)
    consistency_filter = consistency_distance < consistency_eps

    if anchor_probs is None:
        anchor_filter = torch.ones_like(confidence_filter, dtype=torch.bool)
    else:
        anchor_filter = torch.argmax(teacher_probs, dim=1) == torch.argmax(anchor_probs, dim=1)

    selected_mask = confidence_filter & consistency_filter & anchor_filter

    stats = {
        'mean_confidence': max_confidence.mean().detach(),
        'selected_ratio': selected_mask.float().mean().detach(),
        'anchor_agreement_ratio': anchor_filter.float().mean().detach(),
        'consistency_ratio': consistency_filter.float().mean().detach(),
    }
    return selected_mask, pseudo_labels.long(), stats


def compute_unsupervised_pseudo_loss(
    student_logits_or_probs,
    pseudo_labels,
    selected_mask,
    num_classes,
    lambda_ce,
    lambda_dice,
):
    """
    在高可信区域计算无监督 CE + Dice loss。

    输入：
    - student_logits_or_probs: shape [B, num_classes, D, H, W]
    - pseudo_labels: shape [B, D, H, W]
    - selected_mask: shape [B, D, H, W]，bool
    - num_classes: 类别数
    - lambda_ce / lambda_dice: CE 与 Dice 权重

    输出：
    - loss: 标量张量
    - loss_dict: 包含 ce_loss / dice_loss
    """
    selected_mask = selected_mask.bool()
    if not torch.any(selected_mask):
        zero = _zero_scalar(student_logits_or_probs)
        return zero, {'ce_loss': zero.detach(), 'dice_loss': zero.detach()}

    probs_sum = student_logits_or_probs.detach().sum(dim=1).mean()
    if torch.all(student_logits_or_probs >= 0) and torch.all(student_logits_or_probs <= 1.0) and torch.isfinite(probs_sum):
        student_probs = student_logits_or_probs
        student_logits = torch.log(torch.clamp(student_probs, min=1e-6, max=1.0))
    else:
        student_logits = student_logits_or_probs
        student_probs = F.softmax(student_logits, dim=1)

    ce_map = F.cross_entropy(student_logits, pseudo_labels, reduction='none')
    selected_float = selected_mask.float()
    ce_loss = (ce_map * selected_float).sum() / selected_float.sum().clamp_min(1.0)

    pseudo_one_hot = F.one_hot(pseudo_labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    selected_expand = selected_float.unsqueeze(1)
    masked_probs = student_probs * selected_expand
    masked_target = pseudo_one_hot * selected_expand
    eps = 1e-6
    inter = torch.sum(masked_probs * masked_target, dim=(0, 2, 3, 4))
    denom = torch.sum(masked_probs, dim=(0, 2, 3, 4)) + torch.sum(masked_target, dim=(0, 2, 3, 4))
    valid_class = denom > 0
    if torch.any(valid_class):
        dice_per_class = 1.0 - (2.0 * inter[valid_class] + eps) / (denom[valid_class] + eps)
        dice_loss = dice_per_class.mean()
    else:
        dice_loss = _zero_scalar(student_probs)

    total_loss = lambda_ce * ce_loss + lambda_dice * dice_loss
    return total_loss, {'ce_loss': ce_loss.detach(), 'dice_loss': dice_loss.detach()}


def extract_unlabeled_prototypes(features, pseudo_labels, selected_mask, num_classes, detach_features=False):
    """
    从高可信伪标签区域提取无标注 prototype。

    输入：
    - features: shape [B, C_feat, D, H, W]
    - pseudo_labels: shape [B, D, H, W]
    - selected_mask: shape [B, D, H, W]
    - num_classes: 类别数

    输出：
    - prototypes: shape [num_classes, C_feat]
    - valid_classes: shape [num_classes]，bool
    """
    # detach_features=False: prototype 保留梯度，用于 anchor alignment loss。
    # detach_features=True: prototype 仅用于上传或日志，不参与反向传播。
    if detach_features:
        features = features.detach()

    feature_dim = features.size(1)
    prototypes = torch.zeros(num_classes, feature_dim, device=features.device, dtype=features.dtype)
    valid_classes = torch.zeros(num_classes, device=features.device, dtype=torch.bool)

    for cls_idx in range(num_classes):
        class_mask = selected_mask & (pseudo_labels == cls_idx)
        if not torch.any(class_mask):
            continue
        denom = class_mask.float().sum().clamp_min(1.0)
        prototypes[cls_idx] = torch.sum(features * class_mask.unsqueeze(1).float(), dim=(0, 2, 3, 4)) / denom
        valid_classes[cls_idx] = True
    return prototypes, valid_classes


def compute_unlabeled_anchor_alignment_loss(unlabeled_prototypes, valid_classes, global_anchor_bank):
    """
    计算无标注 prototype 与 global full-modal anchor 的对齐损失。

    输入：
    - unlabeled_prototypes: shape [num_classes, C_feat]
    - valid_classes: shape [num_classes]，bool
    - global_anchor_bank: shape [num_classes, C_feat] 或 dict

    输出：
    - anchor_loss: 标量张量
    """
    anchors, anchor_valid = _resolve_anchor_bank(global_anchor_bank, unlabeled_prototypes.device, unlabeled_prototypes.dtype)
    if anchors is None:
        return _zero_scalar(unlabeled_prototypes)
    if anchors.size(1) != unlabeled_prototypes.size(1):
        return _zero_scalar(unlabeled_prototypes)
    active = valid_classes
    if anchor_valid is not None:
        active = active & anchor_valid
    if not torch.any(active):
        return _zero_scalar(unlabeled_prototypes)
    return F.mse_loss(unlabeled_prototypes[active], anchors[active])


def compute_pseudo_filtering(
    student_model,
    teacher_model,
    weak_batch,
    strong_batch,
    mask,
    pseudo_mask_id,
    global_anchor_bank,
    args,
    num_classes,
    num_patterns=15,
):
    """
    Module 2 总入口。

    输入：
    - student_model / teacher_model: 当前客户端的 student 与 EMA teacher
    - weak_batch: shape [B, 4, D, H, W]，weak view
    - strong_batch: shape [B, 4, D, H, W]，strong view
    - mask: shape [B, 4]
    - global_anchor_bank: 全局 full-modal anchor bank，可为 None
    - args: 配置
    - num_classes: 类别数

    输出：
    - result: dict
      - total_loss
      - stats
      - unlabeled_prototypes
      - valid_classes
      - selected_mask
      - pseudo_labels
      - teacher_probs
      - anchor_probs
      - student_probs
      - loss_dict
    """
    with torch.no_grad():
        teacher_outputs = teacher_model.forward_fused_features(weak_batch, mask)
    student_outputs = student_model.forward_fused_features(strong_batch, mask)

    teacher_probs = teacher_outputs['fuse_prob']
    teacher_features = teacher_outputs['fused_feature']
    student_logits = student_outputs['fuse_logits']
    student_probs = student_outputs['fuse_prob']
    student_features = student_outputs['fused_feature']

    # teacher-student 伪标签学习要求 weak / strong 来自同一个空间 patch。
    # 一旦 shape 不一致，逐 voxel consistency 和 pseudo-label 监督都不成立。
    if teacher_probs.shape != student_probs.shape:
        raise ValueError(
            'Module 2 requires aligned weak/strong views, '
            f'but got teacher_probs {tuple(teacher_probs.shape)} and '
            f'student_probs {tuple(student_probs.shape)}.'
        )
    if teacher_features.shape != student_features.shape:
        raise ValueError(
            'Module 2 requires aligned teacher/student features, '
            f'but got teacher_features {tuple(teacher_features.shape)} and '
            f'student_features {tuple(student_features.shape)}.'
        )

    anchor_probs = compute_anchor_similarity_probs(
        teacher_features,
        global_anchor_bank,
        temperature=args.pseudo_temperature,
    )
    refined_probs = refine_pseudo_probs(teacher_probs, anchor_probs, mask)
    selected_mask, pseudo_labels, stats = build_pseudo_label_filters(
        refined_probs=refined_probs,
        teacher_probs=teacher_probs,
        student_probs=student_probs,
        anchor_probs=anchor_probs,
        mask=mask,
        tau_base=args.pseudo_conf_base,
        gamma=args.pseudo_conf_gamma,
        consistency_eps=args.pseudo_consistency_eps,
    )
    pseudo_loss, pseudo_loss_dict = compute_unsupervised_pseudo_loss(
        student_logits_or_probs=student_logits,
        pseudo_labels=pseudo_labels,
        selected_mask=selected_mask,
        num_classes=num_classes,
        lambda_ce=args.pseudo_lambda_ce,
        lambda_dice=args.pseudo_lambda_dice,
    )
    unlabeled_prototypes_for_loss, valid_classes = extract_unlabeled_prototypes(
        features=student_features,
        pseudo_labels=pseudo_labels,
        selected_mask=selected_mask,
        num_classes=num_classes,
        detach_features=False,
    )
    anchor_loss = compute_unlabeled_anchor_alignment_loss(
        unlabeled_prototypes=unlabeled_prototypes_for_loss,
        valid_classes=valid_classes,
        global_anchor_bank=global_anchor_bank,
    )
    total_loss = pseudo_loss + args.pseudo_lambda_anchor * anchor_loss
    # 返回给 train.py 的 prototype 不需要继续保留计算图。
    unlabeled_prototypes_for_upload = unlabeled_prototypes_for_loss.detach()
    missing_pattern_prototypes = extract_missing_pattern_prototypes(
        features=student_features,
        pseudo_labels=pseudo_labels,
        selected_mask=selected_mask,
        pattern_ids=pseudo_mask_id,
        num_patterns=num_patterns,
        num_classes=num_classes,
        detach_features=True,
    )

    loss_dict = {
        'ce_loss': pseudo_loss_dict['ce_loss'],
        'dice_loss': pseudo_loss_dict['dice_loss'],
        'anchor_loss': anchor_loss.detach(),
        'total_loss': total_loss.detach(),
    }
    stats['used_anchor'] = torch.tensor(float(anchor_probs is not None), device=student_logits.device)

    return {
        'total_loss': total_loss,
        'stats': stats,
        'unlabeled_prototypes': unlabeled_prototypes_for_upload,
        'missing_pattern_prototypes': missing_pattern_prototypes,
        'unlabeled_prototypes_for_loss': unlabeled_prototypes_for_loss,
        'valid_classes': valid_classes.detach(),
        'selected_mask': selected_mask.detach(),
        'pseudo_labels': pseudo_labels.detach(),
        'teacher_probs': teacher_probs.detach(),
        'anchor_probs': None if anchor_probs is None else anchor_probs.detach(),
        'student_probs': student_probs.detach(),
        'loss_dict': loss_dict,
    }
