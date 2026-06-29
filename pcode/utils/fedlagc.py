# -*- coding: utf-8 -*-

import math
from collections import OrderedDict

import torch


def clone_state_dict_to_cpu(state_dict):
    return OrderedDict(
        (
            name,
            value.detach().cpu().clone(),
        )
        for name, value in state_dict.items()
    )


def clone_parameter_dict_to_cpu(model):
    return OrderedDict(
        (
            name,
            parameter.detach().cpu().clone(),
        )
        for name, parameter in model.named_parameters()
    )


def get_parameter_names(model):
    return [
        name
        for name, _ in model.named_parameters()
    ]


def first_nonfinite_parameter_name(model):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            return name

    return None


def _is_normalization_parameter(name):
    name = name.lower()

    keywords = [
        "bn",
        "norm",
        "running_mean",
        "running_var",
    ]

    return any(keyword in name for keyword in keywords)


def _is_fixed_parameter(name):
    """
    按 FedLAGC 论文：

    1. 输入层完整保留；
    2. 输出层完整保留；
    3. 所有 bias 完整保留；
    4. normalization 层完整保留。
    """
    if name.endswith(".bias"):
        return True

    if name.startswith("conv1."):
        return True

    if name.startswith("classifier."):
        return True

    if _is_normalization_parameter(name):
        return True

    return False


def _allocate_layer_budget(layer_sizes, layer_scores, budget):
    """
    按论文的层级重要性分配：

    k_l ∝ S_l * d_l

    同时保证：
    0 <= k_l <= d_l
    sum(k_l) = budget
    """
    if budget <= 0:
        return {
            layer_name: 0
            for layer_name in layer_sizes
        }

    total_size = sum(layer_sizes.values())

    if budget >= total_size:
        return dict(layer_sizes)

    denominator = sum(
        layer_scores[layer_name]
        * layer_sizes[layer_name]
        for layer_name in layer_sizes
    )

    if denominator <= 0:
        layer_scores = {
            layer_name: 1.0 / len(layer_sizes)
            for layer_name in layer_sizes
        }

        denominator = sum(
            layer_scores[layer_name]
            * layer_sizes[layer_name]
            for layer_name in layer_sizes
        )

    raw_values = {}

    for layer_name in layer_sizes:
        raw_values[layer_name] = (
            budget
            * layer_scores[layer_name]
            * layer_sizes[layer_name]
            / denominator
        )

    allocated = {
        layer_name: min(
            layer_sizes[layer_name],
            int(math.floor(raw_values[layer_name])),
        )
        for layer_name in layer_sizes
    }

    remaining = budget - sum(allocated.values())

    while remaining > 0:
        candidates = [
            layer_name
            for layer_name in layer_sizes
            if allocated[layer_name] < layer_sizes[layer_name]
        ]

        if len(candidates) == 0:
            break

        candidates = sorted(
            candidates,
            key=lambda name: (
                raw_values[name]
                - math.floor(raw_values[name]),
                layer_scores[name],
                layer_sizes[name]
                - allocated[name],
            ),
            reverse=True,
        )

        for layer_name in candidates:
            if remaining <= 0:
                break

            if allocated[layer_name] < layer_sizes[layer_name]:
                allocated[layer_name] += 1
                remaining -= 1

    return allocated


def build_layer_adaptive_masks(
    state_dict,
    parameter_names,
    capacity,
):
    """
    FedLAGC Layer-Adaptive Submodel Extraction.

    返回：
        masks:
            每个可训练参数对应的 bool mask。

        thresholds:
            每层 Top-K 边界阈值。

        report:
            用于日志记录的预算分配信息。
    """
    capacity = float(capacity)

    if not 0.0 < capacity <= 1.0:
        raise ValueError(
            f"fedlagc capacity must be in (0, 1], got {capacity}."
        )

    parameter_names = list(parameter_names)

    total_numel = sum(
        state_dict[name].numel()
        for name in parameter_names
    )

    fixed_names = [
        name
        for name in parameter_names
        if _is_fixed_parameter(name)
    ]

    prunable_names = [
        name
        for name in parameter_names
        if name not in fixed_names
    ]

    fixed_numel = sum(
        state_dict[name].numel()
        for name in fixed_names
    )

    target_numel = int(
        math.ceil(capacity * total_numel)
    )

    if target_numel < fixed_numel:
        raise RuntimeError(
            "FedLAGC capacity is too small for the mandatory "
            "input/output/bias parameters. "
            f"capacity={capacity}, "
            f"target={target_numel}, "
            f"fixed={fixed_numel}."
        )

    prunable_budget = min(
        sum(state_dict[name].numel() for name in prunable_names),
        target_numel - fixed_numel,
    )

    raw_layer_scores = {}

    for name in prunable_names:
        raw_layer_scores[name] = float(
            state_dict[name]
            .detach()
            .abs()
            .float()
            .mean()
            .item()
        )

    log_scores = {
        name: math.log1p(score)
        for name, score in raw_layer_scores.items()
    }

    log_score_sum = sum(log_scores.values())

    if log_score_sum <= 0:
        normalized_scores = {
            name: 1.0 / max(1, len(prunable_names))
            for name in prunable_names
        }
    else:
        normalized_scores = {
            name: log_scores[name] / log_score_sum
            for name in prunable_names
        }

    layer_sizes = {
        name: state_dict[name].numel()
        for name in prunable_names
    }

    allocated_counts = _allocate_layer_budget(
        layer_sizes=layer_sizes,
        layer_scores=normalized_scores,
        budget=prunable_budget,
    )

    masks = OrderedDict()
    thresholds = OrderedDict()

    for name in fixed_names:
        masks[name] = torch.ones_like(
            state_dict[name],
            dtype=torch.bool,
            device="cpu",
        )

        thresholds[name] = 0.0

    for name in prunable_names:
        value = state_dict[name].detach().cpu()

        flat_value = value.abs().reshape(-1)
        numel = flat_value.numel()
        keep_count = int(allocated_counts[name])

        if keep_count <= 0:
            masks[name] = torch.zeros_like(
                value,
                dtype=torch.bool,
            )

            thresholds[name] = float("inf")
            continue

        if keep_count >= numel:
            masks[name] = torch.ones_like(
                value,
                dtype=torch.bool,
            )

            thresholds[name] = 0.0
            continue

        selected_indices = torch.topk(
            flat_value,
            k=keep_count,
            largest=True,
            sorted=False,
        ).indices

        flat_mask = torch.zeros(
            numel,
            dtype=torch.bool,
        )

        flat_mask[selected_indices] = True

        masks[name] = flat_mask.reshape_as(value)

        thresholds[name] = float(
            flat_value[selected_indices]
            .min()
            .item()
        )

    report = {
        "capacity": capacity,
        "total_numel": total_numel,
        "target_numel": target_numel,
        "fixed_numel": fixed_numel,
        "prunable_budget": prunable_budget,
        "allocated_counts": allocated_counts,
        "normalized_scores": normalized_scores,
        "thresholds": thresholds,
    }

    return masks, thresholds, report


def apply_mask_to_state_dict(
    state_dict,
    masks,
):
    """
    得到稀疏子模型 θ ⊙ M。
    """
    output = OrderedDict()

    for name, value in state_dict.items():
        copied = value.detach().clone()

        if name in masks:
            mask = masks[name].to(
                device=copied.device,
                dtype=copied.dtype,
            )

            copied.mul_(mask)

        output[name] = copied

    return output


def apply_mask_to_model_(
    model,
    masks,
):
    """
    每次优化前后强制施加固定 mask。
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in masks:
                continue

            mask = masks[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

            parameter.mul_(mask)

    return model


def apply_mask_to_optimizer_state_(
    optimizer,
    model,
    masks,
):
    """
    防止 momentum 将已剪枝参数重新推离零。
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in masks:
                continue

            state = optimizer.state.get(parameter, {})

            mask = masks[name].to(
                device=parameter.device,
            )

            for value in state.values():
                if (
                    torch.is_tensor(value)
                    and value.shape == parameter.shape
                ):
                    value.mul_(
                        mask.to(value.dtype)
                    )


def create_zero_correction(
    model,
):
    return OrderedDict(
        (
            name,
            torch.zeros_like(
                parameter.detach(),
                device="cpu",
            ),
        )
        for name, parameter in model.named_parameters()
    )


def apply_lagc_gradient_correction_(
    model,
    masks,
    thresholds,
    correction_state,
    correction_active,
):
    """
    FedLAGC gradient:

    raw_grad =
        grad ⊙ M ⊙
        (1 + 2 |theta| threshold / (|theta| + threshold)^2)

    corrected_grad =
        (raw_grad - h(t) * lambda) ⊙ M
    """
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue

            if name not in masks:
                continue

            mask = masks[name].to(
                device=parameter.device,
                dtype=parameter.grad.dtype,
            )

            threshold = float(
                thresholds.get(name, 0.0)
            )

            corrected_gradient = parameter.grad

            if threshold > 0.0 and math.isfinite(threshold):
                abs_parameter = parameter.detach().abs()

                denominator = (
                    abs_parameter + threshold
                ).pow(2)

                denominator = denominator.clamp_min(
                    torch.finfo(
                        parameter.dtype
                    ).eps
                )

                factor = 1.0 + (
                    2.0
                    * abs_parameter
                    * threshold
                    / denominator
                )

                corrected_gradient = (
                    corrected_gradient * factor
                )

            corrected_gradient = corrected_gradient * mask

            if correction_active:
                correction = correction_state[name].to(
                    device=parameter.device,
                    dtype=parameter.grad.dtype,
                )

                corrected_gradient = (
                    corrected_gradient
                    - correction * mask
                )

            parameter.grad.copy_(corrected_gradient)


def update_correction_state_(
    correction_state,
    round_start_parameter_state,
    model,
    masks,
    beta,
):
    """
    lambda_{t+1}
      =
    lambda_t
      +
    beta * (theta_{t+1} - theta_{t,0}) ⊙ M
    """
    beta = float(beta)

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in masks:
                continue

            start_value = round_start_parameter_state[
                name
            ].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

            mask = masks[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

            correction_state[name].add_(
                beta
                * (
                    parameter.detach()
                    - start_value
                )
                * mask
            )


def correction_state_to_cpu(
    correction_state,
):
    return OrderedDict(
        (
            name,
            value.detach().cpu().clone(),
        )
        for name, value in correction_state.items()
    )


def flatten_masks(
    masks,
    parameter_names,
):
    return torch.cat(
        [
            masks[name]
            .reshape(-1)
            .to(dtype=torch.uint8, device="cpu")
            for name in parameter_names
        ],
        dim=0,
    )


def unflatten_masks(
    mask_vector,
    model,
    parameter_names,
):
    parameter_dict = dict(
        model.named_parameters()
    )

    masks = OrderedDict()
    offset = 0

    for name in parameter_names:
        parameter = parameter_dict[name]

        numel = parameter.numel()

        current = mask_vector[
            offset : offset + numel
        ]

        masks[name] = (
            current
            .reshape(parameter.shape)
            .to(dtype=torch.bool, device="cpu")
        )

        offset += numel

    if offset != mask_vector.numel():
        raise RuntimeError(
            "FedLAGC mask vector length mismatch."
        )

    return masks


def aggregate_overlap_aware_states(
    global_state_dict,
    local_state_dicts,
    local_masks,
    parameter_names,
):
    """
    对每个被客户端训练的参数坐标做重叠平均。

    没有任何客户端更新的位置保持服务器旧值。
    """
    parameter_names = set(parameter_names)

    aggregated = OrderedDict()

    for name, global_value in global_state_dict.items():
        if (
            name not in parameter_names
            or not global_value.dtype.is_floating_point
        ):
            aggregated[name] = global_value.detach().clone()
            continue

        sum_value = torch.zeros_like(
            global_value,
            dtype=torch.float32,
        )

        count_value = torch.zeros_like(
            global_value,
            dtype=torch.float32,
        )

        for local_state, local_mask in zip(
            local_state_dicts,
            local_masks,
        ):
            local_value = local_state[name].to(
                device=global_value.device,
                dtype=torch.float32,
            )

            if not torch.isfinite(local_value).all().item():
                raise RuntimeError(
                    f"FedLAGC local parameter '{name}' "
                    "contains NaN/Inf."
                )

            mask = local_mask[name].to(
                device=global_value.device,
                dtype=torch.float32,
            )

            sum_value.add_(local_value * mask)
            count_value.add_(mask)

        updated = count_value > 0

        new_value = global_value.detach().clone().float()

        new_value[updated] = (
            sum_value[updated]
            / count_value[updated]
        )

        aggregated[name] = new_value.to(
            dtype=global_value.dtype
        )

    return aggregated