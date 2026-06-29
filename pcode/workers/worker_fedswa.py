import copy
import math

import torch
import torch.distributed as dist

import pcode.create_dataset as create_dataset
import pcode.create_scheduler as create_scheduler
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.tensor_buffer import TensorBuffer

from .worker import Worker


class WorkerFedSWA(Worker):
    """FedSWA local training from the official ICML 2025 implementation."""

    def _round_base_lr(self):
        base_lr = self.conf.lr
        if self.conf.fedswa_use_cosine_lr:
            current_round = max(0, self.conf.graph.comm_round)
            total_rounds = max(1, self.conf.n_comm_rounds)
            base_lr = 0.5 * self.conf.lr * (
                1.0 + math.cos(math.pi * current_round / total_rounds)
            )
        return base_lr

    def _local_lr_start(self):
        return self._round_base_lr() * self.conf.fedswa_local_lr_multiplier

    def _make_optimizer(self, lr):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.conf.fedswa_momentum,
            weight_decay=self.conf.fedswa_weight_decay,
            nesterov=self.conf.fedswa_momentum > 0 and self.conf.use_nesterov,
        )

    def prepare_train(self):
        self.model.train()
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)

        self.local_lr_start = self._local_lr_start()
        self.optimizer = self._make_optimizer(self.local_lr_start)
        self.scheduler = create_scheduler.Scheduler(
            self.conf, optimizer=self.optimizer, display_status=False
        )
        self._set_optimizer_lr(self.local_lr_start)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) "
            f"enters FedSWA local training (round={self.conf.graph.comm_round}, "
            f"start_lr={self.local_lr_start})."
        )

    def _set_optimizer_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _total_local_steps(self):
        return max(1, int(float(self.conf.local_n_epochs) * len(self.train_loader)))

    def _scheduled_local_lr(self, step_idx, total_steps):
        min_ratio = self.conf.fedswa_local_lr_min_ratio
        progress = min(max(step_idx / max(1, total_steps), 0.0), 1.0)
        ratio = (1.0 - progress) + progress * min_ratio
        return self.local_lr_start * ratio

    def _after_optimizer_step(self, current_lr, total_steps):
        pass

    def _finish_local_round(self):
        self.conf.logger.log(self.timer.summary())
        self._terminate_comm_round()

    def _train(self):
        self.prepare_train()
        total_steps = self._total_local_steps()
        local_step = 0

        while True:
            for _input, _target in self.train_loader:
                current_lr = self._scheduled_local_lr(local_step, total_steps)
                self._set_optimizer_lr(current_lr)

                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf,
                        _input,
                        _target,
                        is_training=True,
                        device=self.device,
                    )

                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    loss, _, _, _ = self._inference(data_batch)
                
                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(), max_norm=10
                    )
                    self.optimizer.step()
                    self._after_optimizer_step(current_lr, total_steps)
                    self.scheduler.update_training_progress()
                    local_step += 1

                if loss > 1e4 or torch.isnan(loss):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} "
                        f"(client-{self.conf.graph.client_id}) diverges. Early stop it."
                    )
                    self._terminate_comm_round()
                    return

                if self._is_finished_one_comm_round():
                    self._finish_local_round()
                    return

            display_training_stat(self.conf, self.scheduler, self.tracker)
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()


class WorkerFedMoSWA(WorkerFedSWA):
    """FedMoSWA local control-variate update from the official code."""

    def __init__(self, conf):
        self.client_controls = {}
        self.server_control = None
        self.control_names = None
        super().__init__(conf)

    def _local_lr_start(self):
        return self._round_base_lr()

    def _control_template(self, device="cpu"):
        return {
            name: torch.zeros_like(param.detach(), device=device)
            for name, param in self.model.named_parameters()
        }

    def _ensure_client_control(self):
        client_id = self.conf.graph.client_id
        if client_id not in self.client_controls:
            self.client_controls[client_id] = self._control_template(device="cpu")
        return self.client_controls[client_id]

    def recv_extra_info_from_master(self):
        self.control_names = [name for name, _ in self.model.named_parameters()]
        recv_control = self._control_template(device="cpu")
        control_tb = TensorBuffer(list(recv_control.values()), use_cuda=False)
        dist.recv(tensor=control_tb.buffer, src=0)
        control_tb.unpack(recv_control.values(), use_cuda=False)
        self.server_control = recv_control
        self._ensure_client_control()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) "
            "received the FedMoSWA global control variate."
        )
        dist.barrier()

    def _after_optimizer_step(self, current_lr, total_steps):
        local_control = self._ensure_client_control()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                correction = (
                    local_control[name].to(param.device)
                    - self.server_control[name].to(param.device)
                )
                param.add_(correction, alpha=current_lr)

    def prepare_train(self):
        self.global_params_before_train = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }
        super().prepare_train()

    def _finish_local_round(self):
        self._update_local_control()
        super()._finish_local_round()

    def _update_local_control(self):
        client_id = self.conf.graph.client_id
        old_control = copy.deepcopy(self._ensure_client_control())
        avg_lr = max(self.local_lr_start * (1.0 + self.conf.fedswa_local_lr_min_ratio) / 2.0, 1e-12)
        total_steps = self._total_local_steps()

        new_control = {}
        send_control = {}
        for name, param in self.model.named_parameters():
            local_param = param.detach().cpu()
            global_param = self.global_params_before_train[name]
            server_c = self.server_control[name]
            new_c = (global_param - local_param) / (total_steps * avg_lr)
            new_c = new_c + old_control[name] - server_c
            new_control[name] = new_c
            send_control[name] = new_c - server_c

        self.client_controls[client_id] = new_control
        self.send_control = send_control

    def send_extra_info_to_master(self):
        control_tb = TensorBuffer(
            [self.send_control[name] for name in self.control_names],
            use_cuda=False,
        )
        dist.send(tensor=control_tb.buffer, dst=0)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) "
            "sent the FedMoSWA local control update."
        )
        dist.barrier()
