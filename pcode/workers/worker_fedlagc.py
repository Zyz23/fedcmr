
# -*- coding: utf-8 -*-

import math
import os

import torch
import torch.distributed as dist

import pcode.create_dataset as create_dataset

from pcode.workers.worker import Worker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat
from pcode.utils.fedlagc import (
    apply_mask_to_model_,
    apply_mask_to_optimizer_state_,
    clone_parameter_dict_to_cpu,
    clone_state_dict_to_cpu,
    correction_state_to_cpu,
    create_zero_correction,
    first_nonfinite_parameter_name,
    get_parameter_names,
    unflatten_masks,
    update_correction_state_,
)


class WorkerFedLAGC(Worker):
    """
    FedLAGC Client.

    优化点：
    1. 完整模型和 mask 分开接收。
    2. 用 gradient hook 保证被裁剪参数梯度始终为 0。
    3. 不再每个 batch 前后重复扫描整个模型施加 mask。
    4. correction state 只在 correction 激活阶段读写。
    5. NaN/Inf 参数检查仅在每个 local epoch 完成时进行。
    """

    def extra_init(self):
        self.parameter_names = get_parameter_names(
            self.model
        )

        self.mask_vector_length = sum(
            parameter.numel()
            for _, parameter in self.model.named_parameters()
        )

        self.current_masks = None
        self.current_thresholds = None

        self.correction_state = None
        self.correction_path = None

        self.round_start_state = None
        self.round_start_parameter_state = None

        self._gradient_hook_handles = []

    # --------------------------------------------------------
    # correction state 的读取与保存
    # --------------------------------------------------------
    def _get_correction_directory(self):
        configured_dir = getattr(
            self.conf,
            "fedlagc_state_dir",
            "",
        )

        if (
            configured_dir is not None
            and len(str(configured_dir).strip()) > 0
        ):
            directory = configured_dir
        else:
            directory = os.path.join(
                self.conf.checkpoint_root,
                "fedlagc_corrections",
            )

        os.makedirs(directory, exist_ok=True)

        return directory

    def _get_correction_path(self):
        directory = self._get_correction_directory()

        return os.path.join(
            directory,
            f"client_{self.conf.graph.client_id}.pt",
        )

    def _load_correction_state(self):
        self.correction_path = (
            self._get_correction_path()
        )

        default_state = create_zero_correction(
            self.model
        )

        if not os.path.exists(self.correction_path):
            self.correction_state = {
                name: value.to(self.device)
                for name, value in default_state.items()
            }
            return

        try:
            payload = torch.load(
                self.correction_path,
                map_location="cpu",
            )

            stored_state = payload.get(
                "correction_state",
                {},
            )

            valid = True

            for name, default_value in default_state.items():
                if name not in stored_state:
                    valid = False
                    break

                if (
                    stored_state[name].shape
                    != default_value.shape
                ):
                    valid = False
                    break

            if not valid:
                self.conf.logger.log(
                    f"[FedLAGC] client="
                    f"{self.conf.graph.client_id}, "
                    "correction state incompatible; reset."
                )

                self.correction_state = {
                    name: value.to(self.device)
                    for name, value in default_state.items()
                }
                return

            self.correction_state = {
                name: stored_state[name]
                .detach()
                .to(self.device)
                for name in self.parameter_names
            }

        except Exception as error:
            self.conf.logger.log(
                f"[FedLAGC] client="
                f"{self.conf.graph.client_id}, "
                f"cannot load correction state: "
                f"{repr(error)}. Reset it."
            )

            self.correction_state = {
                name: value.to(self.device)
                for name, value in default_state.items()
            }

    def _save_correction_state(self):
        """
        仅 correction 激活阶段调用。
        """
        if (
            self.correction_state is None
            or self.correction_path is None
        ):
            return

        payload = {
            "client_id": int(
                self.conf.graph.client_id
            ),
            "correction_state": correction_state_to_cpu(
                self.correction_state
            ),
        }

        temp_path = self.correction_path + ".tmp"

        torch.save(payload, temp_path)

        os.replace(
            temp_path,
            self.correction_path,
        )

    # --------------------------------------------------------
    # 通信
    # --------------------------------------------------------
    def recv_extra_info_from_master(self):
        """
        FedLAGC 不使用额外 collective 通信。

        mask 和完整 global model 都在
        _recv_model_from_master() 中点对点接收。
        """
        pass

    def _recv_model_from_master(self):
        """
        接收：
        1. 完整 global model；
        2. 当前 client 对应 binary mask。

        Gloo 通信一律使用 CPU tensor，
        收到后再拷到模型所在设备。
        """
        model_state = self.model.state_dict()

        tensor_buffer = TensorBuffer(
            list(model_state.values())
        )

        recv_model_buffer = torch.empty(
            tensor_buffer.buffer.numel(),
            dtype=tensor_buffer.buffer.dtype,
            device="cpu",
        )

        dist.recv(
            tensor=recv_model_buffer,
            src=0,
        )

        tensor_buffer.buffer.copy_(
            recv_model_buffer.to(
                device=tensor_buffer.buffer.device,
                dtype=tensor_buffer.buffer.dtype,
            )
        )

        tensor_buffer.unpack(
            model_state.values()
        )

        self.model.load_state_dict(
            model_state,
            strict=True,
        )

        mask_vector = torch.empty(
            self.mask_vector_length,
            dtype=torch.uint8,
            device="cpu",
        )

        dist.recv(
            tensor=mask_vector,
            src=0,
        )

        self.current_masks = unflatten_masks(
            mask_vector=mask_vector,
            model=self.model,
            parameter_names=self.parameter_names,
        )

        # 只在每轮开始时施加一次 mask。
        apply_mask_to_model_(
            self.model,
            self.current_masks,
        )

        self._rebuild_thresholds_from_mask()

        correction_active = self._correction_is_active()

        if correction_active:
            self._load_correction_state()
            self.round_start_parameter_state = (
                clone_parameter_dict_to_cpu(
                    self.model
                )
            )
        else:
            # 后续轮次不再读取/写入大 correction state。
            self.correction_state = None
            self.correction_path = None
            self.round_start_parameter_state = None

        self.round_start_state = clone_state_dict_to_cpu(
            self.model.state_dict()
        )

        active_numel = int(
            sum(
                mask.sum().item()
                for mask in self.current_masks.values()
            )
        )

        total_numel = int(
            sum(
                mask.numel()
                for mask in self.current_masks.values()
            )
        )

        self.conf.logger.log(
            f"[FedLAGC Receive] "
            f"worker={self.conf.graph.worker_id}, "
            f"client={self.conf.graph.client_id}, "
            f"active={active_numel}/{total_numel}, "
            f"correction={correction_active}."
        )

        dist.barrier()

    def _send_model_to_master(self):
        """
        上传完整 shape 模型。

        服务器依靠本轮保存的 client mask，
        仅聚合该 client 实际参与训练的位置。
        """
        self._remove_gradient_mask_hooks()

        # 仅在上传前做一次安全 mask。
        apply_mask_to_model_(
            self.model,
            self.current_masks,
        )

        model_state = self.model.state_dict()

        tensor_buffer = TensorBuffer(
            list(model_state.values())
        )

        send_buffer = (
            tensor_buffer.buffer
            .detach()
            .cpu()
            .contiguous()
        )

        dist.barrier()

        dist.send(
            tensor=send_buffer,
            dst=0,
        )

        dist.barrier()

        self.conf.logger.log(
            f"[FedLAGC Send] "
            f"worker={self.conf.graph.worker_id}, "
            f"client={self.conf.graph.client_id}."
        )

    # --------------------------------------------------------
    # threshold / gradient hook
    # --------------------------------------------------------
    def _rebuild_thresholds_from_mask(self):
        """
        根据 full model + binary mask 重建每层阈值。

        仅用于 FedLAGC 的梯度调制项。
        """
        parameter_dict = dict(
            self.model.named_parameters()
        )

        self.current_thresholds = {}

        for name in self.parameter_names:
            parameter = parameter_dict[name]

            mask = self.current_masks[name].to(
                device=parameter.device,
                dtype=torch.bool,
            )

            if bool(mask.all().item()):
                self.current_thresholds[name] = 0.0
                continue

            if not bool(mask.any().item()):
                self.current_thresholds[name] = float("inf")
                continue

            active_values = parameter.detach().abs()[mask]

            self.current_thresholds[name] = float(
                active_values.min().item()
            )

    def _remove_gradient_mask_hooks(self):
        for handle in self._gradient_hook_handles:
            handle.remove()

        self._gradient_hook_handles = []

    def _install_gradient_mask_hooks(
        self,
        correction_active,
    ):
        """
        在 backward 时直接处理：

        grad <- grad * TCB(theta, threshold)
        grad <- grad - correction
        grad <- grad * mask

        这样不再需要每个 batch 额外遍历模型调用：
        apply_lagc_gradient_correction_()
        apply_mask_to_model_()
        apply_mask_to_optimizer_state_()
        """
        self._remove_gradient_mask_hooks()

        parameter_dict = dict(
            self.model.named_parameters()
        )

        for name in self.parameter_names:
            parameter = parameter_dict[name]

            mask = self.current_masks[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )

            mask_all = bool(mask.all().item())

            threshold = float(
                self.current_thresholds.get(
                    name,
                    0.0,
                )
            )

            correction = None

            if correction_active:
                correction = self.correction_state[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )

            def make_hook(
                parameter_ref,
                mask_ref,
                mask_all_ref,
                threshold_ref,
                correction_ref,
            ):
                def hook(gradient):
                    output_gradient = gradient

                    # TCB gradient modulation.
                    if (
                        threshold_ref > 0.0
                        and math.isfinite(threshold_ref)
                    ):
                        abs_parameter = (
                            parameter_ref.detach().abs()
                        )

                        denominator = (
                            abs_parameter + threshold_ref
                        ).pow(2)

                        denominator = denominator.clamp_min(
                            torch.finfo(
                                parameter_ref.dtype
                            ).eps
                        )

                        factor = 1.0 + (
                            2.0
                            * abs_parameter
                            * threshold_ref
                            / denominator
                        )

                        output_gradient = (
                            output_gradient * factor
                        )

                    # Historical gradient correction.
                    if correction_ref is not None:
                        output_gradient = (
                            output_gradient - correction_ref
                        )

                    # 对稀疏参数施加梯度掩码。
                    if not mask_all_ref:
                        output_gradient = (
                            output_gradient * mask_ref
                        )

                    return output_gradient

                return hook

            handle = parameter.register_hook(
                make_hook(
                    parameter_ref=parameter,
                    mask_ref=mask,
                    mask_all_ref=mask_all,
                    threshold_ref=threshold,
                    correction_ref=correction,
                )
            )

            self._gradient_hook_handles.append(handle)

    # --------------------------------------------------------
    # correction / stability
    # --------------------------------------------------------
    def _correction_is_active(self):
        fraction = float(
            getattr(
                self.conf,
                "fedlagc_correction_fraction",
                0.25,
            )
        )

        if fraction <= 0.0:
            return False

        cutoff_round = max(
            1,
            int(
                math.ceil(
                    self.conf.n_comm_rounds * fraction
                )
            ),
        )

        return int(self.conf.graph.comm_round) <= cutoff_round

    def _batches_per_epoch(self):
        return max(
            1,
            int(
                getattr(
                    self.conf,
                    "num_batches_per_device_per_epoch",
                    1,
                )
            ),
        )

    def _restore_round_start_model(self):
        self.model.load_state_dict(
            self.round_start_state,
            strict=True,
        )

        apply_mask_to_model_(
            self.model,
            self.current_masks,
        )

        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
        ):
            self.optimizer.zero_grad(
                set_to_none=True
            )

    def _abort_round(self, reason):
        self.conf.logger.log(
            f"[FedLAGC Abort] "
            f"worker={self.conf.graph.worker_id}, "
            f"client={self.conf.graph.client_id}, "
            f"round={self.conf.graph.comm_round}, "
            f"reason={reason}."
        )

        self._remove_gradient_mask_hooks()
        self._restore_round_start_model()

        # 如果 correction 启用，保存旧 correction；
        # 但不使用异常训练得到的新 correction。
        if self._correction_is_active():
            self._save_correction_state()

        self._terminate_comm_round()

    def _check_parameters_finite(self):
        """
        仅每个 local epoch 结束或本轮结束时调用，
        不再每个 batch 扫描 888 万参数。
        """
        return first_nonfinite_parameter_name(
            self.model
        )

    # --------------------------------------------------------
    # 本地训练
    # --------------------------------------------------------
    def _train(self):
        self.prepare_train()

        correction_active = self._correction_is_active()

        beta = float(
            getattr(
                self.conf,
                "fedlagc_beta",
                0.1,
            )
        )

        grad_clip = float(
            getattr(
                self.conf,
                "fedlagc_grad_clip",
                0.0,
            )
        )

        # 只在本轮开始时清理一次 momentum / optimizer state。
        apply_mask_to_model_(
            self.model,
            self.current_masks,
        )

        apply_mask_to_optimizer_state_(
            optimizer=self.optimizer,
            model=self.model,
            masks=self.current_masks,
        )

        # 用 Hook 替代每个 batch 多次 mask 扫描。
        self._install_gradient_mask_hooks(
            correction_active=correction_active,
        )

        batches_per_epoch = self._batches_per_epoch()

        while True:
            for _input, _target in self.train_loader:
                with self.timer(
                    "load_data",
                    epoch=self.scheduler.epoch_,
                ):
                    data_batch = create_dataset.load_data_batch(
                        self.conf,
                        _input,
                        _target,
                        is_training=True,
                        device=self.device,
                    )

                with self.timer(
                    "forward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    (
                        loss,
                        performance,
                        feature,
                        output,
                    ) = self._inference(data_batch)

                with self.timer(
                    "extra_forward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    loss = self.local_training_with_extra_calculate(
                        loss,
                        output,
                        data_batch,
                        feature,
                    )

                # loss 检查保留，每 batch 成本极低。
                if not torch.isfinite(loss).all().item():
                    self._abort_round(
                        "nonfinite_loss"
                    )
                    return

                if float(loss.detach().item()) > 1e4:
                    self._abort_round(
                        "loss_too_large"
                    )
                    return

                with self.timer(
                    "backward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                    loss.backward()

                    # 默认关闭 gradient clipping。
                    # 若后续出现 NaN，可在 parameters.py 加参数后设为 10。
                    if grad_clip > 0.0:
                        grad_norm = (
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=grad_clip,
                            )
                        )

                        if not torch.isfinite(
                            grad_norm
                        ).all().item():
                            self._abort_round(
                                "nonfinite_gradient_norm"
                            )
                            return

                    self.optimizer.step()
                    self.scheduler.step()

                round_finished = (
                    self._is_finished_one_comm_round()
                )

                epoch_finished = (
                    self.scheduler.local_index
                    % batches_per_epoch
                    == 0
                )

                # 只在 epoch 末或 round 末检查一次全模型参数。
                if epoch_finished or round_finished:
                    bad_parameter = (
                        self._check_parameters_finite()
                    )

                    if bad_parameter is not None:
                        self._abort_round(
                            f"nonfinite_parameter:"
                            f"{bad_parameter}"
                        )
                        return

                if round_finished:
                    self._remove_gradient_mask_hooks()

                    # correction 只在有效阶段更新和写盘。
                    if correction_active:
                        update_correction_state_(
                            correction_state=(
                                self.correction_state
                            ),
                            round_start_parameter_state=(
                                self.round_start_parameter_state
                            ),
                            model=self.model,
                            masks=self.current_masks,
                            beta=beta,
                        )

                        self._save_correction_state()

                    timer_summary = self.timer.summary()

                    if (
                        isinstance(timer_summary, str)
                        and len(timer_summary) > 0
                    ):
                        self.conf.logger.log(
                            timer_summary
                        )

                    self._terminate_comm_round()
                    return

                if epoch_finished:
                    display_training_stat(
                        self.conf,
                        self.scheduler,
                        self.tracker,
                    )

                    self.tracker.reset()

                if self.conf.logger.meet_cache_limit():
                    self.conf.logger.save_json()
