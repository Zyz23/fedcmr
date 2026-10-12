# -*- coding: utf-8 -*-
"""FedGMR-Sync client: fixed-mask local SGD on a server-extracted sub-model."""

import torch
import torch.distributed as dist

import pcode.create_dataset as create_dataset

from pcode.utils.efficiency_profiler import (
    begin_efficiency_profile,
    finish_efficiency_profile,
)
from pcode.utils.fedgmr import (
    apply_gradient_mask_,
    apply_model_mask_,
    clone_state_dict_to_cpu,
    first_nonfinite_tensor_name,
    get_prunable_parameter_names,
    mask_optimizer_state_,
    unflatten_mask_tensor,
)
from pcode.utils.logging import display_training_stat
from pcode.workers.worker import Worker


class WorkerFedGMR(Worker):
    """Train only the coordinates selected by the round-fixed FedGMR mask."""

    def recv_extra_info_from_master(self):
        prunable_names = get_prunable_parameter_names(self.model)
        named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if name in prunable_names
        ]
        mask_numel = sum(
            int(parameter.numel())
            for _, parameter in named_parameters
        )

        payload = torch.zeros(
            1 + mask_numel,
            dtype=torch.float32,
            device="cpu",
        )
        dist.recv(payload, src=0)

        self.fedgmr_density = float(payload[0].item())
        self.fedgmr_mask = unflatten_mask_tensor(
            payload[1:],
            named_parameters,
        )
        self.fedgmr_prunable_parameter_names = tuple(prunable_names)

        active = int(payload[1:].sum().item())
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} "
            f"(client-{self.conf.graph.client_id}) received "
            f"FedGMR density={self.fedgmr_density:.6f}, "
            f"active={active}/{mask_numel}."
        )

    def _recv_model_from_master(self):
        super(WorkerFedGMR, self)._recv_model_from_master()

        if not hasattr(self, "fedgmr_mask"):
            raise RuntimeError(
                "FedGMR mask is missing before receiving the sub-model."
            )

        apply_model_mask_(self.model, self.fedgmr_mask)
        self.fedgmr_round_start_state = clone_state_dict_to_cpu(
            self.model.state_dict()
        )

        # The generic worker creates this copy for distillation methods.
        # FedGMR does not use it and keeping it would double resident memory.
        if hasattr(self, "init_model"):
            del self.init_model

    def _restore_round_start_model(self):
        self.model.load_state_dict(
            self.fedgmr_round_start_state,
            strict=True,
        )
        apply_model_mask_(self.model, self.fedgmr_mask)

    def _abort_round(self, stage, tensor_name=None):
        message = (
            f"[FEDGMR ABORT] worker={self.conf.graph.worker_id}, "
            f"client={self.conf.graph.client_id}, "
            f"round={self.conf.graph.comm_round}, stage={stage}"
        )
        if tensor_name is not None:
            message += f", tensor={tensor_name}"

        self.conf.logger.log(
            message + ". Restore the server-extracted round-start sub-model."
        )
        self._restore_round_start_model()
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad()
        self._terminate_comm_round()

    def _train(self):
        self.prepare_train()

        if not hasattr(self, "fedgmr_round_start_state"):
            raise RuntimeError("FedGMR round-start state is missing.")

        while True:
            for _input, _target in self.train_loader:
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf,
                        _input,
                        _target,
                        is_training=True,
                        device=self.device,
                    )
                    begin_efficiency_profile(self)

                # Keep the assigned subspace fixed for the whole local round.
                apply_model_mask_(self.model, self.fedgmr_mask)

                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
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
                    self._abort_round("loss_nan_or_inf")
                    return
                if float(loss.detach().item()) > 1e4:
                    self._abort_round("loss_too_large")
                    return

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss.backward()
                    apply_gradient_mask_(self.model, self.fedgmr_mask)

                    gradient_state = {
                        name: parameter.grad
                        for name, parameter in self.model.named_parameters()
                        if parameter.grad is not None
                    }
                    bad_gradient = first_nonfinite_tensor_name(gradient_state)
                    if bad_gradient is not None:
                        self._abort_round(
                            "gradient_nan_or_inf",
                            bad_gradient,
                        )
                        return

                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=10.0,
                    )
                    if not torch.isfinite(gradient_norm).all().item():
                        self._abort_round("gradient_norm_nan_or_inf")
                        return

                    self.optimizer.step()
                    apply_model_mask_(self.model, self.fedgmr_mask)
                    mask_optimizer_state_(
                        self.optimizer,
                        self.model,
                        self.fedgmr_mask,
                    )
                    finish_efficiency_profile(self, data_batch)
                    self.scheduler.step()

                bad_parameter = first_nonfinite_tensor_name(
                    self.model.state_dict()
                )
                if bad_parameter is not None:
                    self._abort_round(
                        "parameter_nan_or_inf",
                        bad_parameter,
                    )
                    return

                if self._is_finished_one_comm_round():
                    timer_summary = self.timer.summary()
                    if isinstance(timer_summary, str) and timer_summary:
                        self.conf.logger.log(timer_summary)
                    self._terminate_comm_round()
                    return

            display_training_stat(
                self.conf,
                self.scheduler,
                self.tracker,
            )
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _send_model_to_master(self):
        apply_model_mask_(self.model, self.fedgmr_mask)
        super(WorkerFedGMR, self)._send_model_to_master()
