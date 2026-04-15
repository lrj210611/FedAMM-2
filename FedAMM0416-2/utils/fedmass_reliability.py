import torch
import torch.nn.functional as F

from utils.fl_utils import combine_client_weights, normalize_client_weights
from utils.fedmass_anchor import normalize_anchor_bank


MODALITY_NAMES = ('flair', 't1ce', 't1', 't2')


def init_reliability_state(num_clients):
    uniform = 1.0 / float(max(num_clients, 1))
    return {
        'sup_enabled': False,
        'client_sup_history': {client_idx: [] for client_idx in range(num_clients)},
        'client_last_sup_dice': {client_idx: None for client_idx in range(num_clients)},
        'client_last_sup_var': {client_idx: None for client_idx in range(num_clients)},
        'last_raw_scores': [1.0 for _ in range(num_clients)],
        'last_normalized_scores': [uniform for _ in range(num_clients)],
        'last_round': -1,
        'last_details': [],
    }


def restore_reliability_state(state, num_clients):
    restored = init_reliability_state(num_clients)
    if not isinstance(state, dict):
        return restored

    restored['sup_enabled'] = bool(state.get('sup_enabled', False))
    restored['last_round'] = int(state.get('last_round', -1))

    raw_scores = state.get('last_raw_scores')
    if isinstance(raw_scores, (list, tuple)) and len(raw_scores) == num_clients:
        restored['last_raw_scores'] = [float(score) for score in raw_scores]

    normalized_scores = state.get('last_normalized_scores')
    if isinstance(normalized_scores, (list, tuple)) and len(normalized_scores) == num_clients:
        restored['last_normalized_scores'] = [float(score) for score in normalized_scores]

    details = state.get('last_details')
    if isinstance(details, list):
        restored['last_details'] = details

    sup_history = state.get('client_sup_history')
    if isinstance(sup_history, dict):
        normalized_history = {}
        for client_idx in range(num_clients):
            history = sup_history.get(client_idx)
            if history is None:
                history = sup_history.get(str(client_idx), [])
            if not isinstance(history, list):
                history = []
            normalized_history[client_idx] = [float(score) for score in history]
        restored['client_sup_history'] = normalized_history

    for field_name in ('client_last_sup_dice', 'client_last_sup_var'):
        field_value = state.get(field_name)
        if isinstance(field_value, dict):
            normalized_field = {}
            for client_idx in range(num_clients):
                value = field_value.get(client_idx)
                if value is None:
                    value = field_value.get(str(client_idx))
                normalized_field[client_idx] = None if value is None else float(value)
            restored[field_name] = normalized_field

    return restored


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().float().view(-1)[0].item()
    return float(value)


def _compute_pseudo_reliability(pseudo_stats, pseudo_active):
    if not pseudo_active or not isinstance(pseudo_stats, dict):
        return None

    mean_confidence = _to_float(pseudo_stats.get('mean_confidence'))
    selected_ratio = _to_float(pseudo_stats.get('selected_ratio'))
    if mean_confidence is None or selected_ratio is None:
        return None

    score = max(mean_confidence, 0.0) * max(selected_ratio, 0.0)
    return max(0.0, min(score, 1.0))


def _compute_anchor_alignment_reliability(local_unlabeled_prototypes, global_anchor_bank):
    if local_unlabeled_prototypes is None:
        return None

    anchor_bank = normalize_anchor_bank(global_anchor_bank)
    if anchor_bank is None:
        return None

    prototypes = torch.as_tensor(local_unlabeled_prototypes, dtype=torch.float32).detach().cpu()
    anchors = anchor_bank['anchors'].float()
    anchor_valid_mask = anchor_bank['valid_mask'].bool()
    if prototypes.dim() != 2 or anchors.shape != prototypes.shape:
        return None

    prototype_valid_mask = torch.any(prototypes != 0, dim=1)
    active_mask = prototype_valid_mask & anchor_valid_mask
    if not torch.any(active_mask):
        return None

    proto_active = F.normalize(prototypes[active_mask], dim=1)
    anchor_active = F.normalize(anchors[active_mask], dim=1)
    cosine = F.cosine_similarity(proto_active, anchor_active, dim=1)
    score = torch.clamp((cosine + 1.0) * 0.5, min=0.0, max=1.0).mean().item()
    return float(score)


def update_client_sup_history(reliability_state, client_idx, sup_dice, window_size):
    if reliability_state is None:
        return None
    if sup_dice is None:
        return reliability_state

    sup_dice = float(max(min(sup_dice, 1.0), 0.0))
    history = reliability_state['client_sup_history'].setdefault(client_idx, [])
    history.append(sup_dice)
    if window_size is not None and window_size > 0 and len(history) > window_size:
        del history[:-window_size]

    reliability_state['client_last_sup_dice'][client_idx] = sup_dice
    if len(history) >= 2:
        history_tensor = torch.tensor(history, dtype=torch.float32)
        sup_var = float(torch.var(history_tensor, unbiased=False).item())
    else:
        sup_var = None
    reliability_state['client_last_sup_var'][client_idx] = sup_var
    reliability_state['sup_enabled'] = any(
        len(client_history) >= 2 for client_history in reliability_state['client_sup_history'].values()
    )
    return reliability_state


def _compute_sup_reliability(reliability_state, client_idx):
    history = reliability_state.get('client_sup_history', {}).get(client_idx, [])
    last_sup_dice = reliability_state.get('client_last_sup_dice', {}).get(client_idx)
    sup_var = reliability_state.get('client_last_sup_var', {}).get(client_idx)
    if len(history) < 2:
        return {
            'r_sup': None,
            'history_length': len(history),
            'variance': sup_var,
            'last_dice': last_sup_dice,
        }

    if sup_var is None:
        history_tensor = torch.tensor(history, dtype=torch.float32)
        sup_var = float(torch.var(history_tensor, unbiased=False).item())
    r_sup = 1.0 / (1.0 + max(float(sup_var), 0.0))
    return {
        'r_sup': float(max(min(r_sup, 1.0), 0.0)),
        'history_length': len(history),
        'variance': float(max(float(sup_var), 0.0)),
        'last_dice': last_sup_dice,
    }


def compute_client_reliability(
    client_idx,
    pseudo_stats,
    pseudo_active,
    local_unlabeled_prototypes,
    global_anchor_bank,
    reliability_state=None,
):
    reliability_state = reliability_state or {}
    components = {
        'r_sup': None,
        'r_pl': None,
        'r_align': None,
    }
    available_components = []

    sup_detail = _compute_sup_reliability(reliability_state, client_idx)
    if sup_detail['r_sup'] is not None:
        components['r_sup'] = sup_detail['r_sup']
        available_components.append(sup_detail['r_sup'])

    r_pl = _compute_pseudo_reliability(pseudo_stats, pseudo_active)
    if r_pl is not None:
        components['r_pl'] = r_pl
        available_components.append(r_pl)

    r_align = _compute_anchor_alignment_reliability(local_unlabeled_prototypes, global_anchor_bank)
    if r_align is not None:
        components['r_align'] = r_align
        available_components.append(r_align)

    sup_enabled = bool(reliability_state.get('sup_enabled', False))
    raw_score = sum(available_components) / float(len(available_components)) if available_components else 1.0

    return {
        'components': components,
        'available_components': len(available_components),
        'sup_enabled': sup_enabled,
        'sup_history_length': sup_detail['history_length'],
        'sup_variance': sup_detail['variance'],
        'sup_last_dice': sup_detail['last_dice'],
        'used_fallback': len(available_components) == 0,
        'raw_score': float(max(raw_score, 0.0)),
    }


def build_reliability_aggregation(
    client_payloads,
    global_anchor_bank,
    client_modal_weight,
    reliability_state=None,
):
    num_clients = len(client_payloads)
    reliability_state = restore_reliability_state(reliability_state, num_clients)

    details = []
    raw_scores = []
    for client_idx, payload in enumerate(client_payloads):
        detail = compute_client_reliability(
            client_idx=client_idx,
            pseudo_stats=payload.get('pseudo_stats'),
            pseudo_active=bool(payload.get('pseudo_active', False)),
            local_unlabeled_prototypes=payload.get('local_unlabeled_prototypes'),
            global_anchor_bank=global_anchor_bank,
            reliability_state=reliability_state,
        )
        detail['client_idx'] = client_idx
        details.append(detail)
        raw_scores.append(detail['raw_score'])

    reliability_scores = normalize_client_weights(raw_scores, num_clients=num_clients)
    uniform_model_base = torch.full((num_clients,), 1.0 / float(max(num_clients, 1)), dtype=torch.float32)
    model_weights = combine_client_weights(uniform_model_base, raw_scores)

    client_modal_weight = torch.as_tensor(client_modal_weight, dtype=torch.float32)
    encoder_weights = {}
    for modality_idx, modality_name in enumerate(MODALITY_NAMES):
        encoder_weights[modality_name] = combine_client_weights(
            client_modal_weight[:, modality_idx],
            raw_scores,
        )

    for client_idx, detail in enumerate(details):
        detail['normalized_score'] = float(reliability_scores[client_idx].item())
        detail['model_weight'] = float(model_weights[client_idx].item())
        detail['encoder_weights'] = {
            modality_name: float(encoder_weights[modality_name][client_idx].item())
            for modality_name in MODALITY_NAMES
        }

    reliability_state['last_raw_scores'] = [float(score) for score in raw_scores]
    reliability_state['last_normalized_scores'] = [float(score.item()) for score in reliability_scores]
    reliability_state['last_details'] = details

    return {
        'client_details': details,
        'raw_scores': raw_scores,
        'reliability_scores': reliability_scores.tolist(),
        'model_weights': model_weights.tolist(),
        'encoder_weights': {
            modality_name: encoder_weights[modality_name].tolist()
            for modality_name in MODALITY_NAMES
        },
        'reliability_state': reliability_state,
    }
