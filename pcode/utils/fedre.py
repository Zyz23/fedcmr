# -*- coding: utf-8 -*-
"""Core utilities for FedRE (CVPR 2026).

FedRE maps heterogeneous client representations to a shared dimensionality
and uploads one Random Average Prototype (RAP) pair per participating client.
The helpers in this module are deliberately independent from the distributed
runtime so they can be unit-tested in isolation.
"""

import torch
import torch.nn.functional as F


FEDRE_METRIC_FIELDS = 4


def average_pool_representation(representation, output_dim):
    """Map arbitrary client features to ``output_dim`` with 1-D average pooling.

    The official FedRE implementation uses adaptive average pooling as its
    default Representation Mapping (RM).  Flattening all non-batch dimensions
    also makes the helper work for vector and spatial feature tensors.
    """

    if representation.dim() < 2:
        raise ValueError("FedRE representations must include a batch dimension.")
    if int(output_dim) <= 0:
        raise ValueError("fedre_feature_dim must be positive.")

    flattened = representation.reshape(representation.size(0), 1, -1)
    mapped = F.adaptive_avg_pool1d(flattened, int(output_dim))
    return mapped.squeeze(1)


def random_average_prototype(class_sums, class_counts, generator=None):
    """Construct one FedRE RAP representation and its soft label.

    ``class_sums`` has shape ``[num_classes, feature_dim]`` and
    ``class_counts`` has shape ``[num_classes]``.  Only locally observed
    classes participate.  A fresh Uniform(0, 1) weight is sampled per active
    class and normalized to sum to one, exactly as Eq. (10) in the FedRE
    supplementary material.
    """

    if class_sums.dim() != 2:
        raise ValueError("class_sums must have shape [classes, feature_dim].")
    if class_counts.dim() != 1:
        raise ValueError("class_counts must have shape [classes].")
    if class_sums.size(0) != class_counts.numel():
        raise ValueError("class_sums and class_counts disagree on class count.")

    active = class_counts > 0
    if not bool(active.any().item()):
        raise ValueError("FedRE cannot entangle an empty local dataset.")

    active_indices = active.nonzero(as_tuple=False).reshape(-1)
    counts = class_counts[active_indices].to(
        device=class_sums.device,
        dtype=class_sums.dtype,
    )
    prototypes = class_sums[active_indices] / counts.unsqueeze(1)

    # Generate on CPU for deterministic behavior across CPU/GPU workers and
    # then move the small weight vector to the representation device.
    random_weights = torch.rand(
        active_indices.numel(),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    random_weights = random_weights.clamp_min(1e-12)
    random_weights = random_weights / random_weights.sum()
    weights = random_weights.to(
        device=class_sums.device,
        dtype=class_sums.dtype,
    )

    entangled_representation = (weights.unsqueeze(1) * prototypes).sum(dim=0)
    entangled_label = class_sums.new_zeros(class_sums.size(0))
    entangled_label[active_indices] = weights

    return entangled_representation, entangled_label


def soft_cross_entropy(logits, soft_targets):
    """Cross-entropy for probability targets, compatible with PyTorch 1.9."""

    if logits.shape != soft_targets.shape:
        raise ValueError("FedRE logits and soft targets must have equal shapes.")
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def fedre_payload_size(feature_dim, num_classes):
    """Return the fixed number of transmitted scalars in this simulator."""

    return int(feature_dim) + int(num_classes) + FEDRE_METRIC_FIELDS


def pack_fedre_payload(
    representation,
    soft_label,
    loss_sum=0.0,
    top1_correct=0.0,
    top5_correct=0.0,
    num_samples=0.0,
):
    """Pack the algorithm pair plus local-evaluation control metadata."""

    if representation.dim() != 1 or soft_label.dim() != 1:
        raise ValueError("FedRE payload tensors must be one-dimensional.")

    metadata = representation.new_tensor(
        [loss_sum, top1_correct, top5_correct, num_samples]
    )
    return torch.cat([representation, soft_label, metadata], dim=0)


def unpack_fedre_payload(payload, feature_dim, num_classes):
    """Inverse of :func:`pack_fedre_payload`."""

    expected = fedre_payload_size(feature_dim, num_classes)
    if payload.dim() != 1 or payload.numel() != expected:
        raise ValueError(
            "Invalid FedRE payload size: expected {}, got {}.".format(
                expected,
                payload.numel(),
            )
        )

    feature_end = int(feature_dim)
    label_end = feature_end + int(num_classes)
    representation = payload[:feature_end]
    soft_label = payload[feature_end:label_end]
    metrics = payload[label_end:]
    return representation, soft_label, metrics