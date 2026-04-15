import torch


def normalize_client_weights(weights, num_clients=None, eps=1e-12):
    weights = torch.as_tensor(weights, dtype=torch.float32).flatten()
    if num_clients is None:
        num_clients = int(weights.numel())
    if weights.numel() != num_clients:
        raise ValueError(
            'Client weight length mismatch: '
            f'expected {num_clients}, got {int(weights.numel())}.'
        )

    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, min=0.0)
    weight_sum = float(weights.sum().item())
    if num_clients == 0:
        return weights
    if weight_sum <= eps:
        return torch.full((num_clients,), 1.0 / float(num_clients), dtype=torch.float32)
    return weights / weight_sum


def combine_client_weights(base_weights, reliability_scores=None, eps=1e-12):
    base_weights = normalize_client_weights(base_weights)
    if reliability_scores is None:
        return base_weights

    reliability_scores = torch.as_tensor(reliability_scores, dtype=torch.float32).flatten()
    if reliability_scores.numel() != base_weights.numel():
        raise ValueError(
            'Reliability score length mismatch: '
            f'expected {int(base_weights.numel())}, got {int(reliability_scores.numel())}.'
        )

    reliability_scores = torch.nan_to_num(reliability_scores, nan=0.0, posinf=0.0, neginf=0.0)
    reliability_scores = torch.clamp(reliability_scores, min=0.0)
    combined = base_weights * reliability_scores
    combined_sum = float(combined.sum().item())
    if combined_sum <= eps:
        return base_weights
    return combined / combined_sum


def aggregate_state_dicts(state_dicts, client_weights):
    if len(state_dicts) == 0:
        raise ValueError('state_dicts must not be empty.')

    client_weights = normalize_client_weights(client_weights, num_clients=len(state_dicts))
    aggregated_weights = {}
    first_state = state_dicts[0]

    for key in first_state.keys():
        sum_weight = None
        for client_idx, client_weight in enumerate(state_dicts):
            weighted_value = client_weight[key].data.cpu() * client_weights[client_idx]
            if sum_weight is None:
                sum_weight = weighted_value.clone()
            else:
                sum_weight += weighted_value
        aggregated_weights[key] = sum_weight

    return aggregated_weights


def avg_local_weights(w1, w2, w3, w4, client_weights):
    return aggregate_state_dicts([w1, w2, w3, w4], client_weights)


def avg_encoder_weights(w1, w2, w3, w4, client_weights):
    return aggregate_state_dicts([w1, w2, w3, w4], client_weights)
