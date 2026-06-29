# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist

import pcode.create_dataset as create_dataset

from pcode.workers.worker import Worker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat
from pcode.utils.fiarse import (
    apply_fiarse_mask_to_model_,
    apply_fiarse_tcb_gradients_,
    build_fiarse_delta_state_dict,
    clone_state_dict_to_cpu,
    first_nonfinite_parameter_name,
    mask_optimizer_state_,
)


class WorkerFIARSE(Worker):
    """
    Correct FIARSE client.

    Receives x_t ⊙ M_i(x_t), trains under a fixed threshold, and uploads:
        Delta_i = x_t ⊙ M_i(x_t) - x_{t,K}^{(i)}.
    """

    def recv_extra_info_from_master(self):
        threshold_message = torch.zeros(
            self.conf.n_participated,
            dtype=torch.float32,
        )

        dist.broadcast(
            tensor=threshold_message,
            src=0,
        )

        self.fiarse_threshold = float(
            threshold_message[
                self.conf.graph.rank - 1
            ].item()
        )

    def _recv_model_from_master(self):
        """
        Receive sparse global model from Master and cache it as x_t ⊙ M_i.
        """
        super(WorkerFIARSE, self)._recv_model_from_master()

        self.fiarse_parameter_names = {
            name
            for name, _ in self.model.named_parameters()
        }

        apply_fiarse_mask_to_model_(
            self.model,
            self.fiarse_threshold,
        )

        self.fiarse_round_start_state = clone_state_dict_to_cpu(
            self.model.state_dict()
        )

        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} "
            f"(client-{self.conf.graph.client_id}) received "
            f"FIARSE sparse model, threshold="
            f"{self.fiarse_threshold:.8f}."
        )

    def _restore_round_start_model(self):
        """
        Restore x_t ⊙ M_i if local training becomes non-finite.

        The subsequent uploaded delta becomes exactly zero.
        """
        self.model.load_state_dict(
            self.fiarse_round_start_state,
            strict=True,
        )

        apply_fiarse_mask_to_model_(
            self.model,
            self.fiarse_threshold,
        )

    def _abort_round(self, stage, parameter_name=None):
        message = (
            f"[FIARSE ABORT] worker={self.conf.graph.worker_id}, "
            f"client={self.conf.graph.client_id}, "
            f"round={self.conf.graph.comm_round}, "
            f"stage={stage}"
        )

        if parameter_name is not None:
            message += f", parameter={parameter_name}"

        self.conf.logger.log(
            message + ". Restore sparse start model."
        )

        self._restore_round_start_model()
        self._terminate_comm_round()

    def _train(self):
        self.prepare_train()

        if not hasattr(self, "fiarse_round_start_state"):
            raise RuntimeError(
                "FIARSE sparse start state is missing. "
                "Check _recv_model_from_master()."
            )

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

                # Enforce x ⊙ M_i before every local iteration.
                apply_fiarse_mask_to_model_(
                    self.model,
                    self.fiarse_threshold,
                )

                with self.timer(
                    "forward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    loss, performance, feature, output = self._inference(
                        data_batch
                    )

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

                if not torch.isfinite(loss).all().item():
                    self._abort_round(stage="loss_nan_or_inf")
                    return

                if float(loss.detach().item()) > 1e4:
                    self._abort_round(stage="loss_too_large")
                    return

                with self.timer(
                    "backward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    self.optimizer.zero_grad()

                    loss.backward()

                    # TCB-GD: mask gradients and add the FIARSE bias term.
                    apply_fiarse_tcb_gradients_(
                        self.model,
                        self.fiarse_threshold,
                    )

                    bad_gradient = first_nonfinite_parameter_name(
                        {
                            name: parameter.grad
                            for name, parameter
                            in self.model.named_parameters()
                            if parameter.grad is not None
                        },
                        {
                            name
                            for name, parameter
                            in self.model.named_parameters()
                            if parameter.grad is not None
                        },
                    )

                    if bad_gradient is not None:
                        self._abort_round(
                            stage="gradient_nan_or_inf",
                            parameter_name=bad_gradient,
                        )
                        return

                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=10.0,
                    )

                    if not torch.isfinite(gradient_norm).all().item():
                        self._abort_round(
                            stage="gradient_norm_nan_or_inf"
                        )
                        return

                    self.optimizer.step()

                    # Parameters and optimizer momentum must stay masked.
                    apply_fiarse_mask_to_model_(
                        self.model,
                        self.fiarse_threshold,
                    )

                    mask_optimizer_state_(
                        self.optimizer,
                        self.model,
                        self.fiarse_threshold,
                    )

                    bad_parameter = first_nonfinite_parameter_name(
                        self.model.state_dict(),
                        self.fiarse_parameter_names,
                    )

                    if bad_parameter is not None:
                        self._abort_round(
                            stage="parameter_nan_or_inf",
                            parameter_name=bad_parameter,
                        )
                        return

                    self.scheduler.step()

                if self._is_finished_one_comm_round():
                    timer_summary = self.timer.summary()

                    # 兼容不同 Timer.summary() 实现：
                    # 有的版本返回字符串，有的版本仅内部记录并返回 None。
                    if isinstance(timer_summary, str) and len(timer_summary) > 0:
                        self.conf.logger.log(timer_summary)

                    self._terminate_comm_round()
                    return

                # 只在本地一个 epoch 完成时输出日志，避免每个 batch 都打印。
                batches_per_epoch = max(
                    1,
                    getattr(
                        self.conf,
                        "num_batches_per_device_per_epoch",
                        1,
                    ),
                )

                if (
                    self.scheduler.local_index
                    % batches_per_epoch
                    == 0
                ):
                    display_training_stat(
                        self.conf,
                        self.scheduler,
                        self.tracker,
                    )
                    self.tracker.reset()

                if self.conf.logger.meet_cache_limit():
                    self.conf.logger.save_json()

    def _send_model_to_master(self):
        """
        Send Delta_i instead of the complete local sparse model.
        """
        if not hasattr(self, "fiarse_round_start_state"):
            raise RuntimeError(
                "FIARSE start state is missing before upload."
            )

        local_delta_state = build_fiarse_delta_state_dict(
            sparse_start_state=self.fiarse_round_start_state,
            current_state=self.model.state_dict(),
            parameter_names=self.fiarse_parameter_names,
        )

        flatten_delta = TensorBuffer(
            list(local_delta_state.values()),
            use_cuda=False,
        )

        flatten_delta.buffer = flatten_delta.buffer.cpu()

        dist.barrier()

        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} "
            f"(client-{self.conf.graph.client_id}) sends "
            f"FIARSE delta to Master, "
            f"numel={flatten_delta.buffer.numel()}."
        )

        dist.send(
            tensor=flatten_delta.buffer,
            dst=0,
        )

        dist.barrier()