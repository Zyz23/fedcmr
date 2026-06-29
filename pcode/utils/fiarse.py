# -*- coding: utf-8 -*-

import math
import torch


def clone_state_dict_to_cpu(state_dict):
    """Clone a state_dict to CPU while preserving key order."""
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def _is_selected_parameter(name, value, parameter_names):
    return (
        name in parameter_names
        and value.is_floating_point()
    )


def build_modelwise_fiarse_threshold(
    state_dict,
    parameter_names,
    capacity,
):
    """
    Model-wise FIARSE threshold.

    Keep the largest `capacity` fraction of trainable parameter magnitudes:
        theta = TopK_gamma(|x|)

    Parameters outside parameter_names, such as BN buffers, are excluded.
    """
    capacity = float(capacity)

    if not 0.0 < capacity <= 1.0:
        raise ValueError(
            f"FIARSE capacity must be in (0, 1], got {capacity}."
        )

    scores = []

    for name, value in state_dict.items():
        if _is_selected_parameter(name, value, parameter_names):
            scores.append(
                value.detach()
                .abs()
                .reshape(-1)
                .float()
                .cpu()
            )

    if len(scores) == 0:
        raise RuntimeError(
            "FIARSE cannot find floating trainable parameters."
        )

    scores = torch.cat(scores, dim=0)
    total = scores.numel()

    keep = int(math.ceil(capacity * total))
    keep = max(1, min(total, keep))

    if keep >= total:
        threshold = scores.min()
    else:
        threshold = torch.topk(
            scores,
            k=keep,
            largest=True,
            sorted=False,
        ).values.min()

    return float(threshold.item())


def mask_state_dict_by_threshold(
    state_dict,
    parameter_names,
    threshold,
    to_cpu=False,
):
    """
    Build x ⊙ M_i(x).

    Only trainable parameters are sparsified. Buffers are copied unchanged.
    """
    threshold = float(threshold)
    masked_state = {}

    for name, value in state_dict.items():
        copied_value = (
            value.detach().cpu().clone()
            if to_cpu
            else value.detach().clone()
        )

        if (
            threshold > 0.0
            and _is_selected_parameter(
                name,
                copied_value,
                parameter_names,
            )
        ):
            mask = copied_value.abs() >= threshold
            copied_value.mul_(mask.to(copied_value.dtype))

        masked_state[name] = copied_value

    return masked_state


def apply_fiarse_mask_to_model_(model, threshold):
    """
    Apply the fixed FIARSE threshold mask to all trainable parameters.

    This is executed:
    1. after receiving the sparse global model;
    2. before local forward;
    3. after every optimizer step.
    """
    threshold = float(threshold)

    if threshold <= 0.0:
        return model

    with torch.no_grad():
        for parameter in model.parameters():
            mask = parameter.detach().abs() >= threshold
            parameter.mul_(mask.to(parameter.dtype))

    return model


def apply_fiarse_tcb_gradients_(model, threshold):
    """
    Apply Threshold-Controlled Biased Gradient Descent.

    grad <- grad ⊙ M(x) ⊙
            [1 + 2|x|theta / (|x| + theta)^2]
    """
    threshold = float(threshold)

    if threshold <= 0.0:
        return

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            abs_parameter = parameter.detach().abs()
            active_mask = abs_parameter >= threshold

            denominator = (abs_parameter + threshold).pow(2)
            denominator = denominator.clamp_min(
                torch.finfo(parameter.dtype).eps
            )

            bias = 1.0 + (
                2.0 * abs_parameter * threshold
            ) / denominator

            parameter.grad.mul_(
                active_mask.to(parameter.grad.dtype)
                * bias.to(parameter.grad.dtype)
            )


def mask_optimizer_state_(optimizer, model, threshold):
    """
    Clear momentum entries belonging to pruned coordinates.

    Without this, SGD momentum can repeatedly push zeroed parameters
    away from zero before the post-step mask is re-applied.
    """
    threshold = float(threshold)

    if threshold <= 0.0:
        return

    with torch.no_grad():
        for parameter in model.parameters():
            active_mask = (
                parameter.detach().abs() >= threshold
            )

            parameter_state = optimizer.state.get(parameter, {})

            for value in parameter_state.values():
                if (
                    torch.is_tensor(value)
                    and value.shape == parameter.shape
                ):
                    value.mul_(active_mask.to(value.dtype))


def build_fiarse_delta_state_dict(
    sparse_start_state,
    current_state,
    parameter_names,
):
    """
    Build delta:
        Delta_i = x_t ⊙ M_i(x_t) - x_{t,K}^{(i)}

    Non-parameter entries are set to zero because FIARSE aggregates
    trainable parameter updates rather than state buffers.
    """
    if sparse_start_state.keys() != current_state.keys():
        raise RuntimeError(
            "FIARSE start/current state_dict keys are inconsistent."
        )

    delta_state = {}

    for name, current_value in current_state.items():
        current_cpu = current_value.detach().cpu()

        if _is_selected_parameter(
            name,
            current_cpu,
            parameter_names,
        ):
            start_value = sparse_start_state[name].to(
                dtype=current_cpu.dtype,
                device="cpu",
            )

            delta_state[name] = (
                start_value - current_cpu
            ).clone()
        else:
            delta_state[name] = torch.zeros_like(current_cpu)

    return delta_state


def aggregate_fiarse_deltas(
    global_state_dict,
    local_delta_states,
    parameter_names,
    global_lr=1.0,
):
    """
    FIARSE coordinate-wise partial aggregation.

    For each coordinate with nonzero client delta:
        x_{t+1}[j] = x_t[j] - eta_s *
                     mean_{i: Delta_i[j] != 0}(Delta_i[j])

    Coordinates not updated by any participating client retain x_t[j].
    """
    global_lr = float(global_lr)

    if global_lr <= 0.0:
        raise ValueError(
            f"fiarse_global_lr must be positive, got {global_lr}."
        )

    aggregated_state = {
        name: value.detach().clone()
        for name, value in global_state_dict.items()
    }

    for name, global_value in global_state_dict.items():
        if not _is_selected_parameter(
            name,
            global_value,
            parameter_names,
        ):
            continue

        global_float = global_value.detach().float()

        sum_delta = torch.zeros_like(global_float)
        update_count = torch.zeros_like(global_float)

        for local_delta_state in local_delta_states:
            if name not in local_delta_state:
                raise RuntimeError(
                    f"Missing FIARSE delta for parameter '{name}'."
                )

            local_delta = local_delta_state[name]

            if local_delta.shape != global_value.shape:
                raise RuntimeError(
                    f"FIARSE delta shape mismatch for '{name}': "
                    f"global={tuple(global_value.shape)}, "
                    f"local={tuple(local_delta.shape)}."
                )

            local_delta = local_delta.to(
                device=global_float.device,
                dtype=torch.float32,
            )

            valid_update = (
                torch.isfinite(local_delta)
                & local_delta.ne(0)
            )

            sum_delta.add_(
                torch.where(
                    valid_update,
                    local_delta,
                    torch.zeros_like(local_delta),
                )
            )

            update_count.add_(
                valid_update.to(update_count.dtype)
            )

        has_update = update_count > 0.0

        average_delta = sum_delta / update_count.clamp_min(1.0)

        new_value = global_float.clone()
        new_value[has_update] = (
            global_float[has_update]
            - global_lr * average_delta[has_update]
        )

        aggregated_state[name] = new_value.to(
            dtype=global_value.dtype
        )

    return aggregated_state


def first_nonfinite_parameter_name(
    state_dict,
    parameter_names,
):
    """Return the first trainable parameter containing NaN or Inf."""
    for name, value in state_dict.items():
        if _is_selected_parameter(
            name,
            value,
            parameter_names,
        ):
            if not torch.isfinite(value).all().item():
                return name

    return None