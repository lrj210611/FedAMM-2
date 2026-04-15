import torch

from utils import criterions


def normalize_anchor_bank(anchor_bank):
    if anchor_bank is None:
        return None

    anchors = anchor_bank
    valid_mask = None
    if isinstance(anchor_bank, dict):
        anchors = anchor_bank.get('anchors')
        valid_mask = anchor_bank.get('valid_mask')
    if anchors is None:
        return None

    anchors = torch.as_tensor(anchors).detach().cpu().clone()
    if anchors.dim() != 2:
        return None

    if valid_mask is None:
        valid_mask = torch.any(anchors != 0, dim=1)
    else:
        valid_mask = torch.as_tensor(valid_mask).detach().cpu().bool().clone()
        if valid_mask.dim() != 1 or valid_mask.size(0) != anchors.size(0):
            valid_mask = torch.any(anchors != 0, dim=1)

    anchors[~valid_mask] = 0
    return {
        'anchors': anchors,
        'valid_mask': valid_mask.bool(),
    }


def aggregate_global_anchor_bank(client_anchor_list, prev_anchor_bank=None, ema=0.9):
    prev_anchor_bank = normalize_anchor_bank(prev_anchor_bank)
    stats = {
        'num_clients_total': len(client_anchor_list),
        'num_clients_seen': 0,
        'num_contributing_clients': 0,
        'updated_classes': 0,
        'total_valid_classes': 0,
        'reinitialized': False,
    }

    anchor_sum = None
    anchor_count = None

    for client_anchor in client_anchor_list:
        client_anchor = normalize_anchor_bank(client_anchor)
        if client_anchor is None:
            continue

        stats['num_clients_seen'] += 1
        anchors = client_anchor['anchors']
        valid_mask = client_anchor['valid_mask']
        if not torch.any(valid_mask):
            continue

        if anchor_sum is None:
            anchor_sum = torch.zeros_like(anchors)
            anchor_count = torch.zeros(anchors.size(0), dtype=torch.float32)
        elif anchor_sum.shape != anchors.shape:
            raise ValueError(
                'Inconsistent client anchor shape: '
                f'expected {tuple(anchor_sum.shape)}, got {tuple(anchors.shape)}.'
            )

        stats['num_contributing_clients'] += 1
        anchor_sum[valid_mask] += anchors[valid_mask]
        anchor_count[valid_mask] += 1

    if anchor_sum is None:
        if prev_anchor_bank is not None:
            stats['total_valid_classes'] = int(prev_anchor_bank['valid_mask'].sum().item())
        return prev_anchor_bank, stats

    aggregated_valid_mask = anchor_count > 0
    aggregated_anchors = torch.zeros_like(anchor_sum)
    if torch.any(aggregated_valid_mask):
        aggregated_anchors[aggregated_valid_mask] = (
            anchor_sum[aggregated_valid_mask]
            / anchor_count[aggregated_valid_mask].unsqueeze(1)
        )
    stats['updated_classes'] = int(aggregated_valid_mask.sum().item())

    if prev_anchor_bank is None:
        next_anchor_bank = {
            'anchors': aggregated_anchors,
            'valid_mask': aggregated_valid_mask,
        }
    else:
        prev_anchors = prev_anchor_bank['anchors']
        prev_valid_mask = prev_anchor_bank['valid_mask']
        if prev_anchors.shape != aggregated_anchors.shape:
            next_anchor_bank = {
                'anchors': aggregated_anchors,
                'valid_mask': aggregated_valid_mask,
            }
            stats['reinitialized'] = True
        else:
            ema = min(max(float(ema), 0.0), 1.0)
            next_anchors = prev_anchors.clone()
            next_valid_mask = prev_valid_mask.clone()

            update_mask = aggregated_valid_mask & prev_valid_mask
            new_mask = aggregated_valid_mask & (~prev_valid_mask)

            if torch.any(update_mask):
                next_anchors[update_mask] = (
                    ema * prev_anchors[update_mask]
                    + (1.0 - ema) * aggregated_anchors[update_mask]
                )
            if torch.any(new_mask):
                next_anchors[new_mask] = aggregated_anchors[new_mask]

            next_valid_mask = prev_valid_mask | aggregated_valid_mask
            next_anchors[~next_valid_mask] = 0
            next_anchor_bank = {
                'anchors': next_anchors,
                'valid_mask': next_valid_mask,
            }

    stats['total_valid_classes'] = int(next_anchor_bank['valid_mask'].sum().item())
    return next_anchor_bank, stats


def build_full_modal_mask(batch_size, device):
    return torch.ones((batch_size, 4), dtype=torch.bool, device=device)


def extract_class_anchors(feature_map, target, num_cls):
    target = target.float()
    eps = 1e-5
    feature_dim = feature_map.size(1)
    anchors = torch.zeros(num_cls, feature_dim, device=feature_map.device, dtype=feature_map.dtype)
    valid_mask = torch.zeros(num_cls, device=feature_map.device, dtype=torch.bool)

    for cls_idx in range(num_cls):
        target_cls = target[:, cls_idx, ...]
        denom = torch.sum(target_cls)
        if denom.item() <= 0:
            continue
        anchors[cls_idx] = torch.sum(feature_map * target_cls[:, None, ...], dim=(0, 2, 3, 4)) / (denom + eps)
        valid_mask[cls_idx] = True

    return anchors, valid_mask


def init_local_anchor_state(num_cls):
    return {
        'sum': None,
        'count': torch.zeros(num_cls, dtype=torch.float32),
    }


def update_local_anchor_state(anchor_state, batch_anchors, valid_mask):
    if batch_anchors is None or valid_mask is None:
        return anchor_state
    if anchor_state['sum'] is None:
        anchor_state['sum'] = torch.zeros_like(batch_anchors.detach().cpu())
    batch_anchors = batch_anchors.detach().cpu()
    valid_mask = valid_mask.detach().cpu()
    anchor_state['sum'][valid_mask] += batch_anchors[valid_mask]
    anchor_state['count'][valid_mask] += 1
    return anchor_state


def finalize_local_anchor_state(anchor_state):
    if anchor_state is None or anchor_state['sum'] is None:
        return None
    final_anchors = anchor_state['sum'].clone()
    counts = anchor_state['count'].unsqueeze(1).clamp_min(1.0)
    final_anchors = final_anchors / counts
    zero_mask = anchor_state['count'] == 0
    if torch.any(zero_mask):
        final_anchors[zero_mask] = 0
    return final_anchors


def compute_anchor_supervision(args, model, x, target):
    full_mask = build_full_modal_mask(x.size(0), x.device)
    (
        fuse_pred,
        prm_loss_bs,
        sep_loss_bs,
        kl_loss_bs,
        proto_loss_bs,
        _,
        _,
        fused_feature,
    ) = model(x, full_mask, target=target, temp=args.temp, return_features=True)

    seg_loss_bs = (
        criterions.softmax_weighted_loss_bs(fuse_pred, target, num_cls=args.num_class)
        + criterions.dice_loss_bs(fuse_pred, target, num_cls=args.num_class)
    )
    seg_loss = torch.sum(seg_loss_bs)
    sep_loss = torch.sum(sep_loss_bs)
    kd_loss = torch.sum(kl_loss_bs)
    proto_loss = torch.sum(proto_loss_bs)
    prm_loss = torch.sum(prm_loss_bs)

    total_loss = args.anchor_loss_weight * (
        args.anchor_lambda_seg * seg_loss
        + args.anchor_lambda_sep * sep_loss
        + args.anchor_lambda_kd * kd_loss
        + args.anchor_lambda_proto * proto_loss
        + args.anchor_lambda_prm * prm_loss
    )

    anchors, valid_mask = extract_class_anchors(fused_feature.detach(), target, args.num_class)
    return {
        'loss': total_loss,
        'seg_loss': seg_loss.detach(),
        'sep_loss': sep_loss.detach(),
        'kd_loss': kd_loss.detach(),
        'proto_loss': proto_loss.detach(),
        'prm_loss': prm_loss.detach(),
        'anchors': anchors.detach(),
        'anchor_valid_mask': valid_mask.detach(),
    }
