import torch
import torch.nn.functional as F

from utils.fl_utils import normalize_client_weights


def normalize_missing_proto_bank(missing_proto_bank):
    if missing_proto_bank is None:
        return None

    prototypes = missing_proto_bank
    valid_mask = None
    if isinstance(missing_proto_bank, dict):
        prototypes = missing_proto_bank.get('prototypes')
        valid_mask = missing_proto_bank.get('valid_mask')
    if prototypes is None:
        return None

    prototypes = torch.as_tensor(prototypes).detach().cpu().clone()
    if prototypes.dim() != 3:
        return None

    if valid_mask is None:
        valid_mask = torch.any(prototypes != 0, dim=2)
    else:
        valid_mask = torch.as_tensor(valid_mask).detach().cpu().bool().clone()
        if valid_mask.shape != prototypes.shape[:2]:
            valid_mask = torch.any(prototypes != 0, dim=2)

    prototypes[~valid_mask] = 0
    return {
        'prototypes': prototypes,
        'valid_mask': valid_mask.bool(),
    }


def count_valid_pattern_classes(missing_proto_bank):
    missing_proto_bank = normalize_missing_proto_bank(missing_proto_bank)
    if missing_proto_bank is None:
        return 0
    return int(missing_proto_bank['valid_mask'].sum().item())


def _resolve_missing_proto_bank(missing_proto_bank, device, dtype):
    if missing_proto_bank is None:
        return None, None

    prototypes = missing_proto_bank
    valid_mask = None
    if isinstance(missing_proto_bank, dict):
        prototypes = missing_proto_bank.get('prototypes')
        valid_mask = missing_proto_bank.get('valid_mask')
    if prototypes is None:
        return None, None

    prototypes = torch.as_tensor(prototypes, device=device, dtype=dtype)
    if prototypes.dim() != 3:
        return None, None

    if valid_mask is None:
        valid_mask = torch.any(prototypes != 0, dim=2)
    else:
        valid_mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
        if valid_mask.shape != prototypes.shape[:2]:
            valid_mask = torch.any(prototypes != 0, dim=2)

    return prototypes, valid_mask


def init_local_missing_proto_state(num_patterns, num_cls):
    return {
        'sum': None,
        'count': torch.zeros((num_patterns, num_cls), dtype=torch.float32),
    }


def update_local_missing_proto_state(proto_state, batch_proto_bank):
    if proto_state is None or batch_proto_bank is None:
        return proto_state

    prototypes = batch_proto_bank.get('prototypes')
    valid_mask = batch_proto_bank.get('valid_mask')
    counts = batch_proto_bank.get('counts')
    if prototypes is None or valid_mask is None or counts is None:
        return proto_state

    prototypes = torch.as_tensor(prototypes).detach().cpu()
    valid_mask = torch.as_tensor(valid_mask).detach().cpu().bool()
    counts = torch.as_tensor(counts).detach().cpu().float()

    if proto_state['sum'] is None:
        proto_state['sum'] = torch.zeros_like(prototypes)
    elif proto_state['sum'].shape != prototypes.shape:
        raise ValueError(
            'Inconsistent local missing prototype shape: '
            f'expected {tuple(proto_state["sum"].shape)}, got {tuple(prototypes.shape)}.'
        )

    if proto_state['count'].shape != counts.shape:
        raise ValueError(
            'Inconsistent local missing prototype count shape: '
            f'expected {tuple(proto_state["count"].shape)}, got {tuple(counts.shape)}.'
        )

    weighted_sum = prototypes * counts.unsqueeze(-1)
    proto_state['sum'][valid_mask] += weighted_sum[valid_mask]
    proto_state['count'][valid_mask] += counts[valid_mask]
    return proto_state


def finalize_local_missing_proto_state(proto_state):
    if proto_state is None or proto_state['sum'] is None:
        return None

    final_prototypes = proto_state['sum'].clone()
    valid_mask = proto_state['count'] > 0
    if torch.any(valid_mask):
        final_prototypes[valid_mask] = (
            final_prototypes[valid_mask]
            / proto_state['count'][valid_mask].unsqueeze(1)
        )
    final_prototypes[~valid_mask] = 0
    return {
        'prototypes': final_prototypes,
        'valid_mask': valid_mask,
    }


def extract_missing_pattern_prototypes(
    features,
    pseudo_labels,
    selected_mask,
    pattern_ids,
    num_patterns,
    num_classes,
    detach_features=False,
):
    if detach_features:
        features = features.detach()

    pattern_ids = torch.as_tensor(pattern_ids, device=features.device, dtype=torch.long).view(-1)
    if pattern_ids.numel() != features.size(0):
        raise ValueError(
            'Pattern id count mismatch: '
            f'expected {features.size(0)}, got {pattern_ids.numel()}.'
        )

    feature_dim = features.size(1)
    prototype_sum = torch.zeros(
        num_patterns,
        num_classes,
        feature_dim,
        device=features.device,
        dtype=features.dtype,
    )
    counts = torch.zeros(num_patterns, num_classes, device=features.device, dtype=torch.float32)

    for batch_idx in range(features.size(0)):
        pattern_id = int(pattern_ids[batch_idx].item())
        if pattern_id < 0 or pattern_id >= num_patterns:
            continue

        batch_features = features[batch_idx]
        batch_labels = pseudo_labels[batch_idx]
        batch_selected = selected_mask[batch_idx]
        for cls_idx in range(num_classes):
            class_mask = batch_selected & (batch_labels == cls_idx)
            if not torch.any(class_mask):
                continue
            voxel_count = class_mask.float().sum()
            prototype_sum[pattern_id, cls_idx] += torch.sum(
                batch_features * class_mask.unsqueeze(0).float(),
                dim=(1, 2, 3),
            )
            counts[pattern_id, cls_idx] += voxel_count

    valid_mask = counts > 0
    prototypes = torch.zeros_like(prototype_sum)
    if torch.any(valid_mask):
        prototypes[valid_mask] = prototype_sum[valid_mask] / counts[valid_mask].unsqueeze(1)

    return {
        'prototypes': prototypes,
        'valid_mask': valid_mask,
        'counts': counts,
    }


def compute_missing_pattern_alignment_loss(local_missing_proto_bank, global_missing_proto_bank):
    if local_missing_proto_bank is None:
        return None, 0

    local_ref = (
        local_missing_proto_bank.get('prototypes')
        if isinstance(local_missing_proto_bank, dict)
        else local_missing_proto_bank
    )
    if local_ref is None:
        return None, 0
    local_ref = torch.as_tensor(local_ref)
    local_prototypes, local_valid_mask = _resolve_missing_proto_bank(
        local_missing_proto_bank,
        device=local_ref.device,
        dtype=local_ref.dtype if torch.is_floating_point(local_ref) else torch.float32,
    )
    if local_prototypes is None:
        return None, 0

    global_prototypes, global_valid_mask = _resolve_missing_proto_bank(
        global_missing_proto_bank,
        device=local_prototypes.device,
        dtype=local_prototypes.dtype,
    )
    if global_prototypes is None:
        return local_prototypes.new_zeros(()), 0
    if global_prototypes.shape != local_prototypes.shape:
        return local_prototypes.new_zeros(()), 0

    active_mask = local_valid_mask & global_valid_mask
    active_count = int(active_mask.sum().item())
    if active_count == 0:
        return local_prototypes.new_zeros(()), 0

    return F.mse_loss(local_prototypes[active_mask], global_prototypes[active_mask]), active_count


def aggregate_global_missing_proto_bank(
    client_proto_list,
    prev_proto_bank=None,
    ema=0.9,
    client_weights=None,
):
    prev_proto_bank = normalize_missing_proto_bank(prev_proto_bank)
    num_clients = len(client_proto_list)
    aggregation_weights = normalize_client_weights(
        torch.ones(num_clients, dtype=torch.float32) if client_weights is None else client_weights,
        num_clients=num_clients,
    )

    stats = {
        'num_clients_total': num_clients,
        'num_clients_seen': 0,
        'num_contributing_clients': 0,
        'updated_pattern_classes': 0,
        'total_valid_pattern_classes': 0,
        'reinitialized': False,
    }

    proto_sum = None
    weight_sum = None

    for client_idx, client_proto in enumerate(client_proto_list):
        client_proto = normalize_missing_proto_bank(client_proto)
        if client_proto is None:
            continue

        stats['num_clients_seen'] += 1
        prototypes = client_proto['prototypes']
        valid_mask = client_proto['valid_mask']
        if not torch.any(valid_mask):
            continue

        if proto_sum is None:
            proto_sum = torch.zeros_like(prototypes)
            weight_sum = torch.zeros_like(valid_mask, dtype=torch.float32)
        elif proto_sum.shape != prototypes.shape:
            raise ValueError(
                'Inconsistent client missing prototype shape: '
                f'expected {tuple(proto_sum.shape)}, got {tuple(prototypes.shape)}.'
            )

        stats['num_contributing_clients'] += 1
        client_weight = float(aggregation_weights[client_idx].item())
        if client_weight <= 0:
            continue

        proto_sum[valid_mask] += prototypes[valid_mask] * client_weight
        weight_sum[valid_mask] += client_weight

    if proto_sum is None:
        if prev_proto_bank is not None:
            stats['total_valid_pattern_classes'] = int(prev_proto_bank['valid_mask'].sum().item())
        return prev_proto_bank, stats

    aggregated_valid_mask = weight_sum > 0
    aggregated_prototypes = torch.zeros_like(proto_sum)
    if torch.any(aggregated_valid_mask):
        aggregated_prototypes[aggregated_valid_mask] = (
            proto_sum[aggregated_valid_mask]
            / weight_sum[aggregated_valid_mask].unsqueeze(1)
        )
    stats['updated_pattern_classes'] = int(aggregated_valid_mask.sum().item())

    if prev_proto_bank is None:
        next_proto_bank = {
            'prototypes': aggregated_prototypes,
            'valid_mask': aggregated_valid_mask,
        }
    else:
        prev_prototypes = prev_proto_bank['prototypes']
        prev_valid_mask = prev_proto_bank['valid_mask']
        if prev_prototypes.shape != aggregated_prototypes.shape:
            next_proto_bank = {
                'prototypes': aggregated_prototypes,
                'valid_mask': aggregated_valid_mask,
            }
            stats['reinitialized'] = True
        else:
            ema = min(max(float(ema), 0.0), 1.0)
            next_prototypes = prev_prototypes.clone()
            next_valid_mask = prev_valid_mask.clone()

            update_mask = aggregated_valid_mask & prev_valid_mask
            new_mask = aggregated_valid_mask & (~prev_valid_mask)

            if torch.any(update_mask):
                next_prototypes[update_mask] = (
                    ema * prev_prototypes[update_mask]
                    + (1.0 - ema) * aggregated_prototypes[update_mask]
                )
            if torch.any(new_mask):
                next_prototypes[new_mask] = aggregated_prototypes[new_mask]

            next_valid_mask = prev_valid_mask | aggregated_valid_mask
            next_prototypes[~next_valid_mask] = 0
            next_proto_bank = {
                'prototypes': next_prototypes,
                'valid_mask': next_valid_mask,
            }

    stats['total_valid_pattern_classes'] = int(next_proto_bank['valid_mask'].sum().item())
    return next_proto_bank, stats
