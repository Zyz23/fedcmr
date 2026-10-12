# -*- coding: utf-8 -*-
"""Utilities for the CVPR 2026 FedAlign implementation.

The paper leaves ``Align`` in Algorithm 2 abstract.  Its main experiments
explicitly align mean and variance, so that is the stable default here.  An
optional Cornish-Fisher approximation is provided for aligning skewness and
kurtosis as well.
"""

import copy
import math

import torch


MOMENT_NAMES = ("mean", "variance", "skewness", "kurtosis")


def infer_num_channels(sample):
    """Infer the image-channel dimension from one transformed sample."""
    if isinstance(sample, (tuple, list)):
        sample = sample[0]
    if not torch.is_tensor(sample):
        raise TypeError(
            "FedAlign expects transformed image samples to be torch.Tensor, "
            "but received {}.".format(type(sample).__name__)
        )
    if sample.dim() == 2:
        return 1
    if sample.dim() >= 3:
        return int(sample.size(0))
    raise ValueError(
        "FedAlign expects an image shaped [C,H,W] (or [H,W]); got {}.".format(
            tuple(sample.size())
        )
    )


def _as_batched_images(inputs):
    if isinstance(inputs, (tuple, list)):
        raise TypeError(
            "FedAlign currently supports tensor image inputs only; tuple/list "
            "inputs such as tokenized NLP batches are not supported."
        )
    if not torch.is_tensor(inputs):
        raise TypeError("FedAlign inputs must be torch.Tensor.")
    if inputs.dim() == 3:
        # Treat [B,H,W] as single-channel images.
        inputs = inputs.unsqueeze(1)
    if inputs.dim() < 4:
        raise ValueError(
            "FedAlign expects a batch shaped [B,C,...]; got {}.".format(
                tuple(inputs.size())
            )
        )
    return inputs


def compute_channel_moments(data_loader, max_batches=0, eps=1e-8):
    """Compute per-channel mean, variance, skewness and kurtosis.

    Raw moments are accumulated in float64 on CPU to avoid the numerical
    instability of averaging per-batch central moments.
    """
    raw_sums = None
    values_per_channel = 0
    sample_count = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches and batch_index >= int(max_batches):
            break
        if not isinstance(batch, (tuple, list)) or len(batch) < 1:
            raise TypeError("A FedAlign statistics loader must yield (input, target).")

        inputs = _as_batched_images(batch[0]).detach().to(
            device="cpu", dtype=torch.float64
        )
        reduce_dims = (0,) + tuple(range(2, inputs.dim()))
        batch_sums = torch.stack(
            [torch.sum(inputs.pow(order), dim=reduce_dims) for order in range(1, 5)]
        )

        if raw_sums is None:
            raw_sums = batch_sums
        else:
            if raw_sums.size(1) != batch_sums.size(1):
                raise ValueError("FedAlign observed inconsistent channel counts.")
            raw_sums += batch_sums

        sample_count += int(inputs.size(0))
        values_per_channel += int(inputs.numel() // inputs.size(1))

    if raw_sums is None or values_per_channel <= 0:
        raise ValueError("FedAlign cannot compute moments from an empty loader.")

    raw = raw_sums / float(values_per_channel)
    mean = raw[0]
    second = raw[1]
    third = raw[2]
    fourth = raw[3]

    variance = torch.clamp(second - mean.pow(2), min=eps)
    central_third = third - 3.0 * mean * second + 2.0 * mean.pow(3)
    central_fourth = (
        fourth
        - 4.0 * mean * third
        + 6.0 * mean.pow(2) * second
        - 3.0 * mean.pow(4)
    )
    skewness = central_third / variance.pow(1.5)
    kurtosis = central_fourth / variance.pow(2)

    moments = torch.stack((mean, variance, skewness, kurtosis)).float()
    return sanitize_moments(moments, eps=eps), sample_count


def sanitize_moments(moments, eps=1e-8):
    """Replace non-finite values and enforce valid variance/kurtosis."""
    moments = moments.detach().clone().float()
    if moments.dim() != 2 or moments.size(0) != 4:
        raise ValueError(
            "Moments must have shape [4, channels], got {}.".format(
                tuple(moments.size())
            )
        )
    moments = torch.where(torch.isfinite(moments), moments, torch.zeros_like(moments))
    moments[1].clamp_(min=eps)
    # Pearson kurtosis is at least 1. Clamping also prevents an unstable
    # high-order transform when DP noise makes a statistic invalid.
    moments[2].clamp_(min=-3.0, max=3.0)
    moments[3].clamp_(min=1.0, max=12.0)
    return moments


def clip_and_noise_moments(
    moments,
    clip_norm,
    noise_multiplier=0.0,
    seed=0,
):
    """L2-clip each moment vector and apply Gaussian perturbation.

    ``noise_multiplier`` follows DP-SGD terminology: the coordinate-wise
    noise standard deviation is ``noise_multiplier * clip_norm``.  Setting it
    to zero is useful for an alignment-only ablation, but does not provide DP.
    """
    if clip_norm <= 0:
        raise ValueError("fedalign_stat_clip must be positive.")
    if noise_multiplier < 0:
        raise ValueError("fedalign_stat_noise_multiplier cannot be negative.")

    clipped = moments.detach().clone().float()
    for index in range(clipped.size(0)):
        norm = clipped[index].norm(p=2)
        scale = min(1.0, float(clip_norm) / (float(norm) + 1e-12))
        clipped[index].mul_(scale)

    if noise_multiplier > 0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(
            clipped.size(), generator=generator, dtype=clipped.dtype
        )
        clipped.add_(noise, alpha=float(noise_multiplier) * float(clip_norm))
    return clipped


def pack_moment_packet(moments, unsampled_count, sampled_count):
    header = torch.tensor(
        [float(unsampled_count), float(sampled_count)], dtype=torch.float32
    )
    return torch.cat((header, moments.detach().cpu().float().reshape(-1)))


def unpack_moment_packet(packet, num_channels):
    expected = 2 + 4 * int(num_channels)
    if packet.numel() != expected:
        raise ValueError(
            "Invalid FedAlign packet length: expected {}, got {}.".format(
                expected, packet.numel()
            )
        )
    unsampled_count = max(1, int(round(float(packet[0].item()))))
    sampled_count = max(1, int(round(float(packet[1].item()))))
    moments = packet[2:].view(4, int(num_channels)).clone()
    return sanitize_moments(moments), unsampled_count, sampled_count


def aggregate_moments(moment_list, sample_counts, eps=1e-8):
    if len(moment_list) == 0 or len(moment_list) != len(sample_counts):
        raise ValueError("FedAlign requires one positive weight per client moment.")
    total = float(sum(max(0, int(count)) for count in sample_counts))
    if total <= 0:
        raise ValueError("FedAlign received no unsampled examples.")

    result = torch.zeros_like(moment_list[0], dtype=torch.float32)
    for moments, count in zip(moment_list, sample_counts):
        result.add_(moments.float(), alpha=float(count) / total)
    return sanitize_moments(result, eps=eps)


def _broadcast_channel_vector(vector, inputs):
    shape = [1, int(vector.numel())] + [1] * (inputs.dim() - 2)
    return vector.to(device=inputs.device, dtype=inputs.dtype).view(*shape)


def _cornish_fisher(z, skewness, kurtosis):
    """Third-order Cornish-Fisher correction for standardized values."""
    return (
        z
        + skewness / 6.0 * (z.pow(2) - 1.0)
        + (kurtosis - 3.0) / 24.0 * (z.pow(3) - 3.0 * z)
        - skewness.pow(2) / 36.0 * (2.0 * z.pow(3) - 5.0 * z)
    )


def align_batch(
    inputs,
    local_moments,
    global_moments,
    mode="mean_var",
    strength=1.0,
    eps=1e-6,
):
    """Align a sampled image batch to aggregated global moments."""
    original_dim = inputs.dim()
    images = _as_batched_images(inputs)
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("fedalign_alignment_strength must be in [0, 1].")
    if mode not in ("mean_var", "all"):
        raise ValueError("fedalign_moments must be 'mean_var' or 'all'.")
    if images.size(1) != local_moments.size(1):
        raise ValueError(
            "FedAlign channel mismatch: batch has {}, statistics have {}.".format(
                images.size(1), local_moments.size(1)
            )
        )

    local = sanitize_moments(local_moments, eps=eps)
    target = sanitize_moments(global_moments, eps=eps)
    local_mean = _broadcast_channel_vector(local[0], images)
    local_std = _broadcast_channel_vector(local[1].sqrt(), images)
    target_mean = _broadcast_channel_vector(target[0], images)
    target_std = _broadcast_channel_vector(target[1].sqrt(), images)

    standardized = (images - local_mean) / local_std.clamp(min=eps)
    aligned_standardized = standardized

    if mode == "all":
        local_skew = _broadcast_channel_vector(local[2].clamp(-2.0, 2.0), images)
        local_kurt = _broadcast_channel_vector(local[3].clamp(1.0, 10.0), images)
        target_skew = _broadcast_channel_vector(target[2].clamp(-2.0, 2.0), images)
        target_kurt = _broadcast_channel_vector(target[3].clamp(1.0, 10.0), images)

        z = standardized.clamp(min=-6.0, max=6.0)
        # Approximate inverse of the local correction, followed by the global
        # correction.  This makes the otherwise abstract high-order Align
        # operator explicit and reproducible.
        gaussianized = 2.0 * z - _cornish_fisher(z, local_skew, local_kurt)
        aligned_standardized = _cornish_fisher(
            gaussianized, target_skew, target_kurt
        )

        # Preserve the target mean/variance exactly after the approximation.
        reduce_dims = (0,) + tuple(range(2, images.dim()))
        batch_mean = aligned_standardized.mean(dim=reduce_dims, keepdim=True)
        batch_var = aligned_standardized.var(
            dim=reduce_dims, unbiased=False, keepdim=True
        )
        aligned_standardized = (
            aligned_standardized - batch_mean
        ) / batch_var.add(eps).sqrt()

    aligned = aligned_standardized * target_std + target_mean
    result = images + float(strength) * (aligned - images)
    return result.squeeze(1) if original_dim == 3 else result


def privatize_model_state(
    local_state,
    global_state,
    effective_step_size,
    clip_norm,
    noise_multiplier=0.0,
    seed=0,
):
    """Clip and perturb the normalized local model update.

    Algorithm 2 defines ``g_k = (w_t - w_k)/(eta_l * tau)``.  In mini-batch
    training, ``effective_step_size`` is the sum of actual optimizer learning
    rates, which also handles a local learning-rate schedule.
    """
    if effective_step_size <= 0:
        raise ValueError("FedAlign needs at least one local optimizer step.")
    if clip_norm <= 0:
        raise ValueError("fedalign_update_clip must be positive.")
    if noise_multiplier < 0:
        raise ValueError("fedalign_update_noise_multiplier cannot be negative.")
    if list(local_state.keys()) != list(global_state.keys()):
        raise ValueError("FedAlign local/global state dictionaries do not match.")

    gradients = {}
    squared_norm = 0.0
    for key in local_state:
        local_value = local_state[key].detach().cpu()
        global_value = global_state[key].detach().cpu()
        if local_value.is_floating_point():
            gradient = (
                global_value.float() - local_value.float()
            ) / float(effective_step_size)
            gradients[key] = gradient
            squared_norm += float(torch.sum(gradient.pow(2)).item())

    original_norm = math.sqrt(max(0.0, squared_norm))
    clip_factor = min(1.0, float(clip_norm) / (original_norm + 1e-12))
    noise_std = float(noise_multiplier) * float(clip_norm)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    private_state = copy.deepcopy(global_state)
    for key in local_state:
        local_value = local_state[key].detach().cpu()
        global_value = global_state[key].detach().cpu()
        if not local_value.is_floating_point():
            private_state[key] = global_value.clone()
            continue

        private_gradient = gradients[key] * clip_factor
        if noise_std > 0:
            noise = torch.randn(
                private_gradient.size(),
                generator=generator,
                dtype=private_gradient.dtype,
            )
            private_gradient.add_(noise, alpha=noise_std)

        private_value = global_value.float() - (
            float(effective_step_size) * private_gradient
        )
        private_state[key] = private_value.to(dtype=local_value.dtype)

    metrics = {
        "update_norm": original_norm,
        "clip_factor": clip_factor,
        "noise_std": noise_std,
    }
    return private_state, metrics

