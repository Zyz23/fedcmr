# -*- coding: utf-8 -*-
"""Core utilities for the synchronous FedGMR integration.

FedGMR keeps a dense global model at the server and lets clients optimize
masked sub-models.  The helpers in this module implement the three pieces
that are independent of the distributed runtime:

1. validation-plateau driven gradual density restoration;
2. nested importance masks derived from global weight changes;
3. coordinate-wise mask-aware model aggregation.

The asynchronous buffer and virtual-time simulator from the paper are not
implemented here.  In a synchronous runtime their staleness weights reduce
to one, leaving the mask-aware aggregation operator analysed by the paper.
"""

import math

import torch


def parse_density_ladder(raw_ladder, initial_densities):
    """Return a validated, sorted density ladder that always ends at 1.0."""
    initial = [float(value) for value in initial_densities]

    if len(initial) == 0:
        raise ValueError("FedGMR requires at least one initial client density.")

    if raw_ladder is None or len(str(raw_ladder).strip()) == 0:
        configured = []
    elif isinstance(raw_ladder, str):
        configured = [
            float(token.strip())
            for token in raw_ladder.split(",")
            if len(token.strip()) > 0
        ]
    else:
        configured = [float(value) for value in raw_ladder]

    values = initial + configured + [1.0]

    for value in values:
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                "FedGMR densities must be finite and in (0, 1], "
                f"got {value}."
            )

    return sorted(set(values))


class FedGMRRestorationController(object):
    """Unified ESM controller for gradual, ladder-based model restoration."""

    def __init__(
        self,
        initial_densities,
        density_ladder,
        patience=5,
        min_delta=0.0,
        check_interval=1,
    ):
        self.client_densities = [float(value) for value in initial_densities]
        self.density_ladder = parse_density_ladder(
            density_ladder,
            self.client_densities,
        )
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.check_interval = int(check_interval)

        if self.patience <= 0:
            raise ValueError("fedgmr_patience must be positive.")
        if self.check_interval <= 0:
            raise ValueError("fedgmr_check_interval must be positive.")
        if self.min_delta < 0.0 or not math.isfinite(self.min_delta):
            raise ValueError("fedgmr_min_delta must be finite and non-negative.")

        self.current_floor = min(self.client_densities)
        self.best_metric = None
        self.bad_checks = 0
        self.num_restorations = 0

    def should_check(self, comm_round):
        return int(comm_round) % self.check_interval == 0

    def _next_floor(self):
        for density in self.density_ladder:
            if density > self.current_floor + 1e-12:
                return density
        return 1.0

    def observe(self, metric, comm_round):
        """Observe one validation metric and return a restoration event or None."""
        if not self.should_check(comm_round):
            return None

        metric = float(metric)
        if not math.isfinite(metric):
            return None

        if (
            self.best_metric is None
            or metric > self.best_metric + self.min_delta
        ):
            self.best_metric = metric
            self.bad_checks = 0
            return None

        self.bad_checks += 1

        if self.bad_checks < self.patience or self.current_floor >= 1.0 - 1e-12:
            return None

        old_floor = self.current_floor
        old_densities = list(self.client_densities)
        new_floor = self._next_floor()

        self.current_floor = new_floor
        self.client_densities = [
            max(density, new_floor)
            for density in self.client_densities
        ]
        self.best_metric = None
        self.bad_checks = 0
        self.num_restorations += 1

        return {
            "round": int(comm_round),
            "metric": metric,
            "old_floor": old_floor,
            "new_floor": new_floor,
            "old_densities": old_densities,
            "new_densities": list(self.client_densities),
            "restoration_index": self.num_restorations,
        }

    def state_dict(self):
        return {
            "client_densities": list(self.client_densities),
            "density_ladder": list(self.density_ladder),
            "patience": self.patience,
            "min_delta": self.min_delta,
            "check_interval": self.check_interval,
            "current_floor": self.current_floor,
            "best_metric": self.best_metric,
            "bad_checks": self.bad_checks,
            "num_restorations": self.num_restorations,
        }


def get_prunable_parameter_names(model):
    """Return Conv/Linear-like trainable weights; biases and norm stay dense."""
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.is_floating_point()
        and parameter.ndim > 1
    )


def clone_state_dict_to_cpu(state_dict):
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def _importance_vector(
    global_state_dict,
    previous_global_state_dict,
    parameter_names,
):
    chunks = []

    for name in parameter_names:
        if name not in global_state_dict:
            raise KeyError(f"Missing FedGMR global parameter '{name}'.")

        current = global_state_dict[name].detach().float().cpu()

        if previous_global_state_dict is None:
            previous = torch.zeros_like(current)
        else:
            if name not in previous_global_state_dict:
                raise KeyError(f"Missing FedGMR previous parameter '{name}'.")
            previous = previous_global_state_dict[name].detach().float().cpu()

        if current.shape != previous.shape:
            raise ValueError(
                f"FedGMR history shape mismatch for '{name}': "
                f"current={tuple(current.shape)}, previous={tuple(previous.shape)}."
            )

        # Paper Eq. (18): approximate gradient with the global weight change,
        # then rank by the squared weight-gradient product.
        delta = current - previous
        score = (current * delta).pow(2).reshape(-1)
        chunks.append(score)

    if len(chunks) == 0:
        raise RuntimeError("FedGMR cannot find a prunable parameter.")

    scores = torch.cat(chunks, dim=0)
    scores = torch.where(
        torch.isfinite(scores),
        scores,
        torch.full_like(scores, -float("inf")),
    )

    # If two consecutive global models are identical, use magnitude as the
    # deterministic fallback instead of producing an arbitrary zero ranking.
    finite_scores = scores[torch.isfinite(scores)]
    if finite_scores.numel() == 0 or finite_scores.max().item() <= 0.0:
        magnitude_chunks = [
            global_state_dict[name]
            .detach()
            .float()
            .cpu()
            .abs()
            .pow(2)
            .reshape(-1)
            for name in parameter_names
        ]
        scores = torch.cat(magnitude_chunks, dim=0)
        scores = torch.where(
            torch.isfinite(scores),
            scores,
            torch.full_like(scores, -float("inf")),
        )

    return scores


def build_nested_importance_masks(
    global_state_dict,
    previous_global_state_dict,
    parameter_names,
    densities,
):
    """Build exact top-density masks sharing one ranking, hence nested."""
    parameter_names = tuple(parameter_names)
    densities = [float(value) for value in densities]

    for density in densities:
        if not 0.0 < density <= 1.0:
            raise ValueError(
                f"FedGMR mask density must be in (0, 1], got {density}."
            )

    scores = _importance_vector(
        global_state_dict,
        previous_global_state_dict,
        parameter_names,
    )
    total = int(scores.numel())
    order = torch.argsort(scores, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(total, dtype=order.dtype)

    shapes = [tuple(global_state_dict[name].shape) for name in parameter_names]
    sizes = [int(global_state_dict[name].numel()) for name in parameter_names]

    results = []
    for density in densities:
        keep = max(1, min(total, int(math.ceil(density * total))))
        flat_mask = rank < keep
        mask_dict = {}
        offset = 0

        for name, shape, size in zip(parameter_names, shapes, sizes):
            mask_dict[name] = flat_mask[offset : offset + size].reshape(shape).clone()
            offset += size

        results.append(mask_dict)

    return results


def flatten_mask_dict(mask_dict, parameter_names, dtype=torch.float32):
    chunks = []
    for name in parameter_names:
        if name not in mask_dict:
            raise KeyError(f"Missing FedGMR mask for '{name}'.")
        chunks.append(mask_dict[name].reshape(-1).to(dtype=dtype, device="cpu"))
    return torch.cat(chunks, dim=0)


def unflatten_mask_tensor(flat_mask, named_parameters):
    """Reconstruct a CPU boolean mask dict in named-parameter order."""
    mask_dict = {}
    offset = 0

    for name, parameter in named_parameters:
        size = int(parameter.numel())
        mask_dict[name] = (
            flat_mask[offset : offset + size]
            .reshape(parameter.shape)
            .to(device="cpu", dtype=torch.bool)
            .clone()
        )
        offset += size

    if offset != int(flat_mask.numel()):
        raise ValueError(
            f"FedGMR mask payload has {flat_mask.numel()} entries, "
            f"but the model consumes {offset}."
        )

    return mask_dict


def mask_state_dict(global_state_dict, mask_dict, to_cpu=True):
    """Extract one masked client sub-model from the full global model."""
    result = {}

    for name, value in global_state_dict.items():
        copied = (
            value.detach().cpu().clone()
            if to_cpu
            else value.detach().clone()
        )

        if name in mask_dict:
            mask = mask_dict[name].to(copied.device)
            if mask.shape != copied.shape:
                raise ValueError(
                    f"FedGMR mask shape mismatch for '{name}': "
                    f"mask={tuple(mask.shape)}, value={tuple(copied.shape)}."
                )
            copied.mul_(mask.to(copied.dtype))

        result[name] = copied

    return result


def apply_model_mask_(model, mask_dict):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in mask_dict:
                mask = mask_dict[name].to(parameter.device)
                parameter.mul_(mask.to(parameter.dtype))


def apply_gradient_mask_(model, mask_dict):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in mask_dict:
                mask = mask_dict[name].to(parameter.grad.device)
                parameter.grad.mul_(mask.to(parameter.grad.dtype))


def mask_optimizer_state_(optimizer, model, mask_dict):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in mask_dict:
                continue

            mask = mask_dict[name].to(parameter.device)
            parameter_state = optimizer.state.get(parameter, {})

            for value in parameter_state.values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.mul_(mask.to(value.device, dtype=value.dtype))


def first_nonfinite_tensor_name(state_dict):
    for name, value in state_dict.items():
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            return name
    return None


def aggregate_masked_model_states(
    global_state_dict,
    local_states_and_masks,
    parameter_names,
):
    """Synchronous mask-aware aggregation into a complete global model."""
    if len(local_states_and_masks) == 0:
        raise ValueError("FedGMR cannot aggregate an empty client set.")

    parameter_names = set(parameter_names)
    aggregated = {
        name: value.detach().clone()
        for name, value in global_state_dict.items()
    }

    for name, global_value in global_state_dict.items():
        if name in parameter_names:
            sum_value = torch.zeros_like(global_value, dtype=torch.float32)
            coverage = torch.zeros_like(global_value, dtype=torch.float32)

            for local_state, mask_dict in local_states_and_masks:
                if name not in local_state or name not in mask_dict:
                    raise KeyError(f"Incomplete FedGMR update for '{name}'.")

                local_value = local_state[name].to(
                    device=global_value.device,
                    dtype=torch.float32,
                )
                mask = mask_dict[name].to(
                    device=global_value.device,
                    dtype=torch.bool,
                )

                if local_value.shape != global_value.shape or mask.shape != global_value.shape:
                    raise ValueError(
                        f"FedGMR aggregation shape mismatch for '{name}'."
                    )
                if not torch.isfinite(local_value[mask]).all().item():
                    raise FloatingPointError(
                        f"FedGMR client update is non-finite at '{name}'."
                    )

                sum_value.add_(torch.where(mask, local_value, torch.zeros_like(local_value)))
                coverage.add_(mask.to(coverage.dtype))

            covered = coverage > 0
            new_value = global_value.detach().float().clone()
            new_value[covered] = sum_value[covered] / coverage[covered]
            aggregated[name] = new_value.to(global_value.dtype)
            continue

        values = []
        for local_state, _ in local_states_and_masks:
            if name not in local_state:
                raise KeyError(f"Missing FedGMR dense state '{name}'.")
            value = local_state[name].to(global_value.device)
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise FloatingPointError(
                    f"FedGMR client dense state is non-finite at '{name}'."
                )
            values.append(value)

        stacked = torch.stack([value.float() for value in values], dim=0)
        mean_value = stacked.mean(dim=0)
        if global_value.dtype.is_floating_point:
            aggregated[name] = mean_value.to(global_value.dtype)
        else:
            aggregated[name] = torch.round(mean_value).to(global_value.dtype)

    return aggregated
