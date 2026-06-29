# -*- coding: utf-8 -*-

import copy
import torch


NEURON_LAYERS = (
    "conv1",
    "conv2",
    "conv3",
    "conv4",
    "fc1",
)


def clone_state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def _all_indices(size):
    return torch.arange(size, dtype=torch.long)


def _take_with_indices(value, indices):
    result = value

    for dim, index in enumerate(indices):
        result = result.index_select(
            dim,
            index.to(result.device),
        )

    return result


def _mesh_indices(indices, device):
    tensors = [
        index.to(device)
        for index in indices
    ]

    if len(tensors) == 0:
        return ()

    return torch.meshgrid(
        *tensors,
        indexing="ij",
    )


def first_nonfinite_state_key(state_dict):
    for key, value in state_dict.items():
        if value.is_floating_point():
            if not torch.isfinite(value).all().item():
                return key

    return None


def get_neuron_counts(model):
    required_layers = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "fc1",
        "classifier",
    ]

    for layer_name in required_layers:
        if not hasattr(model, layer_name):
            raise NotImplementedError(
                "FedDSE currently supports HeteroCNNCifar "
                "and HeteroCNNtiny only."
            )

    return {
        "conv1": int(model.conv1.out_channels),
        "conv2": int(model.conv2.out_channels),
        "conv3": int(model.conv3.out_channels),
        "conv4": int(model.conv4.out_channels),
        "fc1": int(model.fc1.out_features),
    }


def validate_neuron_selections(
    selections,
    global_state_dict,
    local_state_dict,
):
    required_keys = [
        "conv1.weight",
        "conv2.weight",
        "conv3.weight",
        "conv4.weight",
        "fc1.weight",
        "classifier.weight",
    ]

    for key in required_keys:
        if key not in global_state_dict:
            raise RuntimeError(
                f"FedDSE global state misses '{key}'."
            )

        if key not in local_state_dict:
            raise RuntimeError(
                f"FedDSE local state misses '{key}'."
            )

    expected_counts = {
        "conv1": local_state_dict["conv1.weight"].shape[0],
        "conv2": local_state_dict["conv2.weight"].shape[0],
        "conv3": local_state_dict["conv3.weight"].shape[0],
        "conv4": local_state_dict["conv4.weight"].shape[0],
        "fc1": local_state_dict["fc1.weight"].shape[0],
    }

    global_counts = {
        "conv1": global_state_dict["conv1.weight"].shape[0],
        "conv2": global_state_dict["conv2.weight"].shape[0],
        "conv3": global_state_dict["conv3.weight"].shape[0],
        "conv4": global_state_dict["conv4.weight"].shape[0],
        "fc1": global_state_dict["fc1.weight"].shape[0],
    }

    for layer_name in NEURON_LAYERS:
        if layer_name not in selections:
            raise RuntimeError(
                f"FedDSE selection misses '{layer_name}'."
            )

        selected = selections[layer_name].long().cpu()

        if selected.numel() != expected_counts[layer_name]:
            raise RuntimeError(
                f"FedDSE selection size mismatch for "
                f"{layer_name}: "
                f"got={selected.numel()}, "
                f"expected={expected_counts[layer_name]}."
            )

        if selected.numel() != selected.unique().numel():
            raise RuntimeError(
                f"FedDSE selection contains duplicate neurons "
                f"in {layer_name}."
            )

        if selected.min().item() < 0:
            raise RuntimeError(
                f"FedDSE selection contains a negative index "
                f"in {layer_name}."
            )

        if selected.max().item() >= global_counts[layer_name]:
            raise RuntimeError(
                f"FedDSE selection index exceeds global size "
                f"in {layer_name}."
            )


def build_cnn_index_map(
    global_state_dict,
    local_state_dict,
    selections,
):
    """
    Build a consistent arbitrary-channel submodel map.

    conv1 out -> conv2 in
    conv2 out -> conv3 in
    conv3 out -> conv4 in
    conv4 out -> fc1 input
    fc1 out  -> classifier input
    """
    validate_neuron_selections(
        selections,
        global_state_dict,
        local_state_dict,
    )

    idx_c1 = selections["conv1"].long().cpu()
    idx_c2 = selections["conv2"].long().cpu()
    idx_c3 = selections["conv3"].long().cpu()
    idx_c4 = selections["conv4"].long().cpu()
    idx_h1 = selections["fc1"].long().cpu()

    global_conv1 = global_state_dict["conv1.weight"]
    global_conv2 = global_state_dict["conv2.weight"]
    global_conv3 = global_state_dict["conv3.weight"]
    global_conv4 = global_state_dict["conv4.weight"]
    global_fc1 = global_state_dict["fc1.weight"]
    global_classifier = global_state_dict["classifier.weight"]

    idx_rgb = _all_indices(global_conv1.shape[1])

    idx_k1_h = _all_indices(global_conv1.shape[2])
    idx_k1_w = _all_indices(global_conv1.shape[3])

    idx_k2_h = _all_indices(global_conv2.shape[2])
    idx_k2_w = _all_indices(global_conv2.shape[3])

    idx_k3_h = _all_indices(global_conv3.shape[2])
    idx_k3_w = _all_indices(global_conv3.shape[3])

    idx_k4_h = _all_indices(global_conv4.shape[2])
    idx_k4_w = _all_indices(global_conv4.shape[3])

    global_c4 = global_conv4.shape[0]
    global_fc1_input = global_fc1.shape[1]

    if global_fc1_input % global_c4 != 0:
        raise RuntimeError(
            "FedDSE cannot infer fc1 spatial dimension: "
            f"fc1_in={global_fc1_input}, "
            f"conv4_out={global_c4}."
        )

    spatial_area = global_fc1_input // global_c4

    idx_spatial = _all_indices(spatial_area)

    idx_fc1_input = (
        idx_c4[:, None] * spatial_area
        + idx_spatial[None, :]
    ).reshape(-1)

    idx_classes = _all_indices(
        global_classifier.shape[0]
    )

    index_map = {
        "conv1.weight": (
            idx_c1,
            idx_rgb,
            idx_k1_h,
            idx_k1_w,
        ),
        "conv2.weight": (
            idx_c2,
            idx_c1,
            idx_k2_h,
            idx_k2_w,
        ),
        "conv3.weight": (
            idx_c3,
            idx_c2,
            idx_k3_h,
            idx_k3_w,
        ),
        "conv4.weight": (
            idx_c4,
            idx_c3,
            idx_k4_h,
            idx_k4_w,
        ),
        "fc1.weight": (
            idx_h1,
            idx_fc1_input,
        ),
        "classifier.weight": (
            idx_classes,
            idx_h1,
        ),
    }

    if "conv1.bias" in local_state_dict:
        index_map["conv1.bias"] = (idx_c1,)

    if "conv2.bias" in local_state_dict:
        index_map["conv2.bias"] = (idx_c2,)

    if "conv3.bias" in local_state_dict:
        index_map["conv3.bias"] = (idx_c3,)

    if "conv4.bias" in local_state_dict:
        index_map["conv4.bias"] = (idx_c4,)

    if "fc1.bias" in local_state_dict:
        index_map["fc1.bias"] = (idx_h1,)

    if "classifier.bias" in local_state_dict:
        index_map["classifier.bias"] = (
            idx_classes,
        )

    for key, local_value in local_state_dict.items():
        if key in index_map:
            continue

        global_value = global_state_dict[key]

        if global_value.shape != local_value.shape:
            raise RuntimeError(
                f"FedDSE does not know how to map '{key}': "
                f"global={tuple(global_value.shape)}, "
                f"local={tuple(local_value.shape)}."
            )

        index_map[key] = tuple(
            _all_indices(dim)
            for dim in global_value.shape
        )

    return index_map


def extract_indexed_submodel_state(
    global_state_dict,
    local_state_dict,
    index_map,
):
    extracted = copy.deepcopy(local_state_dict)

    for key, local_value in extracted.items():
        global_value = global_state_dict[key]

        selected = _take_with_indices(
            global_value,
            index_map[key],
        )

        if selected.shape != local_value.shape:
            raise RuntimeError(
                f"FedDSE extraction mismatch for '{key}': "
                f"selected={tuple(selected.shape)}, "
                f"local={tuple(local_value.shape)}."
            )

        local_value.copy_(
            selected.to(
                device=local_value.device,
                dtype=local_value.dtype,
            )
        )

    return extracted


def aggregate_indexed_submodel_states(
    global_state_dict,
    local_state_dicts,
    index_maps,
):
    """
    Coordinate-wise aggregation:

    Each global parameter coordinate is averaged only across
    clients that selected this coordinate.
    """
    aggregated = copy.deepcopy(global_state_dict)

    for key, global_value in global_state_dict.items():
        if not global_value.dtype.is_floating_point:
            continue

        sum_value = torch.zeros_like(
            global_value,
            dtype=torch.float32,
        )

        count_value = torch.zeros_like(
            global_value,
            dtype=torch.float32,
        )

        for local_state_dict, index_map in zip(
            local_state_dicts,
            index_maps,
        ):
            local_value = local_state_dict[key].to(
                device=global_value.device,
                dtype=torch.float32,
            )

            if not torch.isfinite(local_value).all().item():
                raise RuntimeError(
                    f"FedDSE local model contains NaN/Inf "
                    f"at '{key}'."
                )

            mesh = _mesh_indices(
                index_map[key],
                global_value.device,
            )

            sum_value.index_put_(
                mesh,
                local_value,
                accumulate=True,
            )

            count_value.index_put_(
                mesh,
                torch.ones_like(local_value),
                accumulate=True,
            )

        updated = count_value > 0

        new_value = global_value.detach().clone().float()

        new_value[updated] = (
            sum_value[updated]
            / count_value[updated]
        )

        aggregated[key] = new_value.to(
            dtype=global_value.dtype
        )

    return aggregated