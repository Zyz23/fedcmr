# -*- coding: utf-8 -*-
"""
One-round client-efficiency profiler for FedCMR experiments.

Usage:
    PROFILE_EFFICIENCY=1 \
    EFFICIENCY_PROFILE_DIR=./data/efficiency \
    python run_gloo.py ... --n_comm_rounds 1 --local_n_epochs 1

This profiler is called inside an already-running worker, so it measures:
- the actual instantiated active local model;
- the exact model.state_dict() payload sent/received by the worker;
- the peak CUDA memory of the real first local optimization step.

Important:
- A frozen round-initial reference is counted only for FedCMR/FedOur by
  default. Generic workers may still keep an unused init_model for FedAvg or
  FedAdam; this profiler reports that object separately as
  `resident_nonreference_init_model_mib` instead of falsely treating it as
  algorithmic distillation overhead.
- Peak memory is sampled BEFORE the extra profiling forwards used for MAC
  counting, preventing the profiling procedure itself from inflating the
  reported peak-memory metric.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# Methods that genuinely use a frozen round-initial local reference in FedCMR.
# Override this with conf.efficiency_uses_round_initial_reference if needed.
_REFERENCE_METHODS = {"fedour", "fedcmr"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _enabled(conf: Any) -> bool:
    return _as_bool(
        getattr(
            conf,
            "profile_efficiency",
            os.environ.get("PROFILE_EFFICIENCY", "0"),
        )
    )


def _first_worker_only(conf: Any) -> bool:
    return _as_bool(
        getattr(
            conf,
            "efficiency_profile_first_worker_only",
            os.environ.get("EFFICIENCY_PROFILE_FIRST_WORKER_ONLY", "1"),
        )
    )


def _audit_state_dict_enabled(conf: Any) -> bool:
    return _as_bool(
        getattr(
            conf,
            "efficiency_profile_state_dict_audit",
            os.environ.get("PROFILE_STATE_DICT_AUDIT", "0"),
        )
    )


def _output_dir(conf: Any) -> Path:
    path = getattr(
        conf,
        "efficiency_profile_dir",
        os.environ.get("EFFICIENCY_PROFILE_DIR", "./data/efficiency"),
    )
    return Path(path)


def _method_name(conf: Any) -> str:
    return str(getattr(conf, "method", "unknown")).strip().lower()


def _uses_round_initial_reference(conf: Any) -> bool:
    """
    Return whether the current algorithm actually uses init_model as a frozen
    round-initial reference during local optimization.

    Optional explicit override:
        conf.efficiency_uses_round_initial_reference = True / False
    """
    explicit = getattr(conf, "efficiency_uses_round_initial_reference", None)
    if explicit is not None:
        return _as_bool(explicit)
    return _method_name(conf) in _REFERENCE_METHODS


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _state_dict_bytes(model: nn.Module) -> int:
    return sum(
        _tensor_bytes(value)
        for value in model.state_dict().values()
        if torch.is_tensor(value)
    )


def _module_bytes(model: Optional[nn.Module]) -> int:
    if model is None:
        return 0
    return sum(_tensor_bytes(param) for param in model.parameters()) + sum(
        _tensor_bytes(buffer) for buffer in model.buffers()
    )


def _nested_tensor_bytes(obj: Any) -> int:
    """Count tensors in cached anchors/lists/dicts."""
    if torch.is_tensor(obj):
        return _tensor_bytes(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_nested_tensor_bytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(_nested_tensor_bytes(value) for value in obj.values())
    return 0


class _ForwardMACCounter:
    """Count Conv2d and Linear MACs from one real forward pass."""

    def __init__(self) -> None:
        self.macs = 0
        self._hooks = []

    @staticmethod
    def _conv2d_macs(module: nn.Conv2d, output: Any) -> int:
        if not torch.is_tensor(output):
            return 0
        out_batch, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        kernel_mul = (module.in_channels // module.groups) * kernel_h * kernel_w
        return int(out_batch * out_channels * out_h * out_w * kernel_mul)

    @staticmethod
    def _linear_macs(module: nn.Linear, inputs: Tuple[Any, ...]) -> int:
        if not inputs or not torch.is_tensor(inputs[0]):
            return 0
        instances = inputs[0].numel() // module.in_features
        return int(instances * module.in_features * module.out_features)

    def _hook(self, module: nn.Module, inputs, output) -> None:
        if isinstance(module, nn.Conv2d):
            self.macs += self._conv2d_macs(module, output)
        elif isinstance(module, nn.Linear):
            self.macs += self._linear_macs(module, inputs)

    def __enter__(self):
        return self

    def attach(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._hooks.append(module.register_forward_hook(self._hook))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


@torch.no_grad()
def forward_macs_per_sample(
    model: Optional[nn.Module],
    sample_input: torch.Tensor,
) -> int:
    """Return Conv2d+Linear MACs for one active-model forward sample."""
    if model is None:
        return 0

    was_training = model.training
    model.eval()
    one_sample = sample_input[:1]

    try:
        with _ForwardMACCounter() as counter:
            counter.attach(model)
            _ = model(one_sample)
        return int(counter.macs)
    finally:
        model.train(was_training)


def _capture_peak_memory(device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    """
    Capture the real training-step peak before profiler-only MAC forwards.
    """
    if device.type != "cuda":
        return None, None

    torch.cuda.synchronize(device)
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return peak_allocated_mib, peak_reserved_mib


def _profile_this_worker(worker: Any) -> bool:
    if not _enabled(worker.conf):
        return False
    if getattr(worker, "_efficiency_profile_done", False):
        return False

    graph = getattr(worker.conf, "graph", None)
    rank = getattr(graph, "rank", 0)
    return not (_first_worker_only(worker.conf) and rank != 1)


def begin_efficiency_profile(worker: Any) -> None:
    """
    Call immediately before the first real forward pass of the first local batch.
    It only resets CUDA peak statistics and never modifies model parameters.
    """
    if not _profile_this_worker(worker):
        return

    worker._efficiency_profile_active = True

    if worker.device.type == "cuda":
        torch.cuda.synchronize(worker.device)
        # The model, cached anchors, frozen reference (if any), and optimizer
        # already allocated at this point remain part of the measured baseline.
        torch.cuda.reset_peak_memory_stats(worker.device)


def _audit_state_dict(model: nn.Module, conf: Any) -> None:
    """Optional payload audit to explain unexpectedly large communication."""
    if not _audit_state_dict_enabled(conf):
        return

    items = sorted(
        (
            (name, tensor)
            for name, tensor in model.state_dict().items()
            if torch.is_tensor(tensor)
        ),
        key=lambda item: _tensor_bytes(item[1]),
        reverse=True,
    )

    logger = getattr(conf, "logger", None)
    for name, tensor in items[:20]:
        message = (
            f"[StateDictAudit] {name}: "
            f"{_tensor_bytes(tensor) / (1024 ** 2):.3f} MiB"
        )
        if logger is not None and hasattr(logger, "log"):
            logger.log(message)
        else:
            print(message)


def _make_record(
    worker: Any,
    sample_input: torch.Tensor,
    peak_allocated_mib: Optional[float],
    peak_reserved_mib: Optional[float],
) -> Dict[str, Any]:
    model = worker.model
    conf = worker.conf

    # A generic worker can have self.init_model even for FedAvg/FedAdam.
    # Count it as a frozen reference only when the current method uses it.
    raw_init_model = getattr(worker, "init_model", None)
    uses_reference = _uses_round_initial_reference(conf)
    reference_model = raw_init_model if uses_reference else None
    nonreference_resident_model = raw_init_model if not uses_reference else None

    payload_bytes = _state_dict_bytes(model)
    _audit_state_dict(model, conf)

    # MAC forwards occur only after real peak memory has been captured above.
    active_forward_macs = forward_macs_per_sample(model, sample_input)
    reference_forward_macs = forward_macs_per_sample(reference_model, sample_input)

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())

    active_gmacs = active_forward_macs / 1e9
    reference_gmacs = reference_forward_macs / 1e9
    estimated_training_step_gmacs = 3.0 * active_gmacs + reference_gmacs

    n_participated = int(getattr(conf, "n_participated", 1))
    rank_ratio = float(
        getattr(worker, "ratio_LR", getattr(conf, "ratio_LR", 1.0))
    )

    extra_init_model_mib = _module_bytes(nonreference_resident_model) / (1024 ** 2)
    notes = (
        "Communication is the exact active-model state_dict payload. "
        "Forward MACs count Conv2d and Linear operators only. "
        "FLOPs are reported as 2x MACs. "
        "Estimated training-step MACs use 3x active forward MACs plus the "
        "frozen-reference forward only when the method actually uses a "
        "round-initial reference model."
    )
    if extra_init_model_mib > 0:
        notes += (
            " A non-reference init_model is still resident in this worker; "
            "the reported peak GPU memory therefore reflects the actual "
            "implementation and includes this framework overhead."
        )

    return {
        "method": str(getattr(conf, "method", "unknown")),
        "dataset": str(getattr(conf, "data", "unknown")),
        "architecture": str(
            getattr(worker, "arch", getattr(conf, "arch", "unknown"))
        ),
        "rank_ratio": rank_ratio,
        "batch_size": int(sample_input.size(0)),
        "profiled_worker_rank": int(
            getattr(getattr(conf, "graph", None), "rank", -1)
        ),
        "profiled_client_id": int(
            getattr(getattr(conf, "graph", None), "client_id", -1)
        ),
        "uses_round_initial_reference": bool(uses_reference),
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "active_model_state_mib": payload_bytes / (1024 ** 2),
        "cached_anchor_mib": (
            _nested_tensor_bytes(getattr(worker, "anchor", None)) / (1024 ** 2)
        ),
        "frozen_reference_model_mib": (
            _module_bytes(reference_model) / (1024 ** 2)
        ),
        "resident_nonreference_init_model_mib": extra_init_model_mib,
        "forward_macs_per_sample": int(active_forward_macs),
        "forward_gmacs_per_sample": active_gmacs,
        "forward_gflops_per_sample": 2.0 * active_gmacs,
        "frozen_reference_forward_gmacs_per_sample": reference_gmacs,
        "estimated_training_step_gmacs_per_sample": estimated_training_step_gmacs,
        "download_mib_per_selected_client": payload_bytes / (1024 ** 2),
        "upload_mib_per_selected_client": payload_bytes / (1024 ** 2),
        "bidirectional_mib_per_selected_client": 2.0 * payload_bytes / (1024 ** 2),
        "bidirectional_mib_per_global_round": (
            2.0 * payload_bytes * n_participated / (1024 ** 2)
        ),
        "n_participated": n_participated,
        "peak_allocated_gpu_mib": peak_allocated_mib,
        "peak_reserved_gpu_mib": peak_reserved_mib,
        "notes": notes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _safe_name(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace("=", "-")


def _write_record(conf: Any, record: Dict[str, Any]) -> Path:
    output_dir = _output_dir(conf)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{_safe_name(record['method'])}_"
        f"{_safe_name(record['dataset'])}_"
        f"{_safe_name(record['architecture'])}_"
        f"r{record['rank_ratio']:.4f}"
    )
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / "efficiency_profiles.csv"
    fieldnames = list(record.keys())

    # Do not append rows with a new schema beneath an old CSV header.
    # This keeps prior profiler outputs intact and writes corrected records
    # to a separate CSV when necessary.
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            existing_header = next(csv.reader(file), [])
        if existing_header != fieldnames:
            csv_path = output_dir / "efficiency_profiles_corrected.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    return json_path


def finish_efficiency_profile(worker: Any, data_batch: Dict[str, torch.Tensor]) -> None:
    """
    Call immediately after optimizer.step() on the first real local batch.
    Exactly one JSON/CSV record is written; model parameters are not changed.
    """
    if not getattr(worker, "_efficiency_profile_active", False):
        return

    worker._efficiency_profile_active = False
    worker._efficiency_profile_done = True

    # Must happen before MAC hooks run extra forward passes.
    peak_allocated_mib, peak_reserved_mib = _capture_peak_memory(worker.device)

    record = _make_record(
        worker=worker,
        sample_input=data_batch["input"],
        peak_allocated_mib=peak_allocated_mib,
        peak_reserved_mib=peak_reserved_mib,
    )
    path = _write_record(worker.conf, record)

    message = (
        "[EfficiencyProfile] "
        f"trainable={record['trainable_parameters'] / 1e6:.3f}M, "
        f"forward={record['forward_gmacs_per_sample']:.3f} GMACs/sample, "
        f"UL/DL={record['upload_mib_per_selected_client']:.3f}/"
        f"{record['download_mib_per_selected_client']:.3f} MiB, "
        f"peak={record['peak_allocated_gpu_mib']} MiB, "
        f"saved={path}"
    )

    logger = getattr(worker.conf, "logger", None)
    if logger is not None and hasattr(logger, "log"):
        logger.log(message)
    else:
        print(message)
