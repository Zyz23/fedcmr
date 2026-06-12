# -*- coding: utf-8 -*-
import copy

import torch


def _prefix_slices(shape):
    return tuple(slice(0, size) for size in shape)


def extract_submodel_state(global_state_dict, sub_state_dict):
    """Extract the prefix-width HeteroFL submodel from a full global state."""
    extracted = copy.deepcopy(sub_state_dict)
    for key, sub_value in extracted.items():
        global_value = global_state_dict[key]
        if sub_value.shape == global_value.shape:
            sub_value.copy_(global_value.to(sub_value.device, dtype=sub_value.dtype))
            continue

        slices = _prefix_slices(sub_value.shape)
        sub_value.copy_(global_value[slices].to(sub_value.device, dtype=sub_value.dtype))
    return extracted


def aggregate_submodel_states(global_state_dict, local_state_dicts):
    """Aggregate nested HeteroFL submodels back into the full global model."""
    aggregated = copy.deepcopy(global_state_dict)

    for key, global_value in global_state_dict.items():
        if not global_value.dtype.is_floating_point:
            included = [
                local_state[key].to(global_value.device)
                for local_state in local_state_dicts
                if key in local_state and local_state[key].shape == global_value.shape
            ]
            if included:
                stacked = torch.stack([value.float() for value in included], dim=0)
                aggregated[key] = torch.round(stacked.mean(dim=0)).to(global_value.dtype)
            continue

        sum_value = torch.zeros_like(global_value, dtype=torch.float32)
        count_value = torch.zeros_like(global_value, dtype=torch.float32)

        for local_state in local_state_dicts:
            local_value = local_state[key].to(global_value.device)
            slices = _prefix_slices(local_value.shape)
            sum_value[slices] += local_value.float()
            count_value[slices] += 1

        updated = count_value > 0
        new_value = global_value.detach().clone().float()
        new_value[updated] = sum_value[updated] / count_value[updated]
        aggregated[key] = new_value.to(global_value.dtype)

    return aggregated


def _rolling_indices(global_size, local_size, round_idx, roll_step=1):
    if local_size == global_size:
        return torch.arange(global_size, dtype=torch.long)
    offset = ((max(1, round_idx) - 1) * max(1, roll_step)) % global_size
    return (torch.arange(local_size, dtype=torch.long) + offset) % global_size


def _cnn_fc1_input_indices(global_size, local_size, round_idx, roll_step=1):
    spatial_size = 4 * 4
    if global_size % spatial_size != 0 or local_size % spatial_size != 0:
        return _rolling_indices(global_size, local_size, round_idx, roll_step)

    global_channels = global_size // spatial_size
    local_channels = local_size // spatial_size
    channel_indices = _rolling_indices(global_channels, local_channels, round_idx, roll_step)
    spatial_indices = torch.arange(spatial_size, dtype=torch.long)
    return (channel_indices[:, None] * spatial_size + spatial_indices[None, :]).reshape(-1)


def _parameter_indices(key, global_shape, local_shape, round_idx, roll_step=1):
    indices = []
    for dim, (global_size, local_size) in enumerate(zip(global_shape, local_shape)):
        if key.endswith("fc1.weight") and dim == 1:
            index = _cnn_fc1_input_indices(global_size, local_size, round_idx, roll_step)
        elif global_size == local_size:
            index = torch.arange(global_size, dtype=torch.long)
        else:
            index = _rolling_indices(global_size, local_size, round_idx, roll_step)
        indices.append(index)
    return tuple(indices)


def _take_with_indices(value, indices):
    result = value
    for dim, index in enumerate(indices):
        result = result.index_select(dim, index.to(value.device))
    return result


def _mesh_indices(indices, device):
    tensors = [index.to(device) for index in indices]
    if len(tensors) == 0:
        return ()
    return torch.meshgrid(*tensors, indexing="ij")


def extract_rolling_submodel_state(global_state_dict, sub_state_dict, round_idx, roll_step=1):
    """Extract FedRolex's rolling submodel from a full global state."""
    extracted = copy.deepcopy(sub_state_dict)
    index_map = {}

    for key, sub_value in extracted.items():
        global_value = global_state_dict[key]
        indices = _parameter_indices(
            key,
            tuple(global_value.shape),
            tuple(sub_value.shape),
            round_idx,
            roll_step,
        )
        index_map[key] = indices
        selected = _take_with_indices(global_value, indices)
        sub_value.copy_(selected.to(sub_value.device, dtype=sub_value.dtype))

    return extracted, index_map


def aggregate_rolling_submodel_states(global_state_dict, local_state_dicts, index_maps):
    """Aggregate FedRolex rolling submodels back into the full global model."""
    aggregated = copy.deepcopy(global_state_dict)

    for key, global_value in global_state_dict.items():
        if not global_value.dtype.is_floating_point:
            full_values = [
                local_state[key].to(global_value.device)
                for local_state in local_state_dicts
                if key in local_state and local_state[key].shape == global_value.shape
            ]
            if full_values:
                stacked = torch.stack([value.float() for value in full_values], dim=0)
                aggregated[key] = torch.round(stacked.mean(dim=0)).to(global_value.dtype)
            continue

        sum_value = torch.zeros_like(global_value, dtype=torch.float32)
        count_value = torch.zeros_like(global_value, dtype=torch.float32)

        for local_state, index_map in zip(local_state_dicts, index_maps):
            local_value = local_state[key].to(global_value.device).float()
            mesh = _mesh_indices(index_map[key], global_value.device)
            sum_value.index_put_(mesh, local_value, accumulate=True)
            count_value.index_put_(mesh, torch.ones_like(local_value), accumulate=True)

        updated = count_value > 0
        new_value = global_value.detach().clone().float()
        new_value[updated] = sum_value[updated] / count_value[updated]
        aggregated[key] = new_value.to(global_value.dtype)

    return aggregated


def _random_indices(global_size, local_size, round_idx, client_id):
    if local_size == global_size:
        return torch.arange(global_size, dtype=torch.long)
    seed = (
        int(max(1, round_idx)) * 1000003
        + int(max(1, client_id)) * 9176
        + int(global_size) * 131
        + int(local_size) * 17
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(global_size, generator=generator)[:local_size].sort().values


def _random_cnn_fc1_input_indices(global_size, local_size, round_idx, client_id):
    spatial_size = 4 * 4
    if global_size % spatial_size != 0 or local_size % spatial_size != 0:
        return _random_indices(global_size, local_size, round_idx, client_id)

    global_channels = global_size // spatial_size
    local_channels = local_size // spatial_size
    channel_indices = _random_indices(global_channels, local_channels, round_idx, client_id)
    spatial_indices = torch.arange(spatial_size, dtype=torch.long)
    return (channel_indices[:, None] * spatial_size + spatial_indices[None, :]).reshape(-1)


def _random_parameter_indices(key, global_shape, local_shape, round_idx, client_id):
    indices = []
    for dim, (global_size, local_size) in enumerate(zip(global_shape, local_shape)):
        if key.endswith("fc1.weight") and dim == 1:
            index = _random_cnn_fc1_input_indices(global_size, local_size, round_idx, client_id)
        elif global_size == local_size:
            index = torch.arange(global_size, dtype=torch.long)
        else:
            index = _random_indices(global_size, local_size, round_idx, client_id)
        indices.append(index)
    return tuple(indices)


def extract_random_submodel_state(global_state_dict, sub_state_dict, round_idx, client_id):
    """Extract a Federated Dropout random-width submodel from a full global state."""
    extracted = copy.deepcopy(sub_state_dict)
    index_map = {}