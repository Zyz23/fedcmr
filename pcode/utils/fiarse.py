# -*- coding: utf-8 -*-
import copy
import math

import torch


def build_modelwise_fiarse_mask(state_dict, capacity):
    """Build FIARSE's model-wise top-|w| mask and threshold."""
    floating_items = [
        (key, value.detach().float().abs().reshape(-1))
        for key, value in state_dict.items()
        if value.dtype.is_floating_point
    ]
    total = sum(value.numel() for _, value in floating_items)
    keep = max(1, min(total, int(math.ceil(float(capacity) * total))))

    scores = torch.cat([value.cpu() for _, value in floating_items], dim=0)
    if keep >= total:
        threshold = scores.min()
        global_mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        topk_indices = torch.topk(scores, keep, largest=True, sorted=False).indices
        global_mask = torch.zeros_like(scores, dtype=torch.bool)
        global_mask[topk_indices] = True
        threshold = scores[topk_indices].min()

    masks = {}
    offset = 0
    for key, value in state_dict.items():
        if not value.dtype.is_floating_point:
            masks[key] = torch.ones_like(value, dtype=torch.bool)
            continue

        end = offset + value.numel()
        flat_mask = global_mask[offset:end]
        offset = end
        masks[key] = flat_mask.reshape(value.shape).to(value.device)

    return masks, float(threshold.item())


def mask_state_dict(state_dict, masks):
    masked = copy.deepcopy(state_dict)
    for key, value in masked.items():
        if value.dtype.is_floating_point:
            value.mul_(masks[key].to(value.device, dtype=value.dtype))
    return masked


def aggregate_fiarse_states(global_state_dict, local_state_dicts, masks, global_lr=1.0):
    aggregated = copy.deepcopy(global_state_dict)

    for key, global_value in global_state_dict.items():
        if not global_value.dtype.is_floating_point:
            values = [
                local_state[key].to(global_value.device)
                for local_state in local_state_dicts
                if key in local_state and local_state[key].shape == global_value.shape
            ]
            if values:
                stacked = torch.stack([value.float() for value in values], dim=0)
                aggregated[key] = torch.round(stacked.mean(dim=0)).to(global_value.dtype)
            continue

        sum_delta = torch.zeros_like(global_value, dtype=torch.float32)
        count = torch.zeros_like(global_value, dtype=torch.float32)
        global_float = global_value.detach().float()

        for local_state, mask in zip(local_state_dicts, masks):
            mask_value = mask[key].to(global_value.device)
            local_float = local_state[key].to(global_value.device).float()
            sum_delta[mask_value] += global_float[mask_value] - local_float[mask_value]
            count[mask_value] += 1

        updated = count > 0
        new_value = global_float.clone()
        avg_delta = torch.zeros_like(global_float)
        avg_delta[updated] = sum_delta[updated] / count[updated]
        new_value[updated] = global_float[updated] - float(global_lr) * avg_delta[updated]
        aggregated[key] = new_value.to(global_value.dtype)

    return aggregated


def apply_fiarse_mask_to_model_(model, threshold):
    threshold = float(threshold)
    with torch.no_grad():
        for parameter in model.parameters():
            if threshold > 0:
                parameter.mul_(parameter.detach().abs() >= threshold)
    return model


def apply_fiarse_tcb_gradients_(model, threshold):
    threshold = float(threshold)
    if threshold <= 0:
        return

    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        abs_parameter = parameter.detach().abs()
        mask = abs_parameter >= threshold
        bias = 1.0 + (2.0 * abs_parameter * threshold) / ((abs_parameter + threshold) ** 2)
        parameter.grad.mul_(mask.to(parameter.grad.dtype) * bias.to(parameter.grad.dtype))