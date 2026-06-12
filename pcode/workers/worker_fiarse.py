# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

import pcode.create_dataset as create_dataset
from pcode.utils.fiarse import (
    apply_fiarse_mask_to_model_,
    apply_fiarse_tcb_gradients_,
)
from pcode.utils.logging import display_training_stat
from pcode.workers.worker import Worker


class WorkerFIARSE(Worker):
    """FIARSE client with threshold-controlled biased gradient descent."""

    def recv_extra_info_from_master(self):
        threshold_msg = torch.zeros(self.conf.n_participated)
        dist.broadcast(tensor=threshold_msg, src=0)
        self.fiarse_threshold = float(threshold_msg[self.conf.graph.rank - 1].item())

    def _recv_model_from_master(self):
        super(WorkerFIARSE, self)._recv_model_from_master()
        apply_fiarse_mask_to_model_(self.model, self.fiarse_threshold)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) "
            f"uses FIARSE threshold={self.fiarse_threshold}."
        )

    def _train(self):
        self.prepare_train()

        while True:
            for _input, _target in self.train_loader:
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True, device=self.device
                    )

                apply_fiarse_mask_to_model_(self.model, self.fiarse_threshold)

                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    loss, performance, feature, output = self._inference(data_batch)

                with self.timer("extra_forward_pass", epoch=self.scheduler.epoch_):
                    loss = self.local_training_with_extra_calculate(
                        loss, output, data_batch, feature
                    )

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss.backward()
                    apply_fiarse_tcb_gradients_(self.model, self.fiarse_threshold)
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(), max_norm=10
                    )
                    self.optimizer.step()
                    apply_fiarse_mask_to_model_(self.model, self.fiarse_threshold)
                    self.scheduler.step()

                if loss > 1e4 or torch.isnan(loss):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )
                    self._terminate_comm_round()
                    return

                if self._is_finished_one_comm_round():
                    self.conf.logger.log(self.timer.summary())
                    self._terminate_comm_round()
                    return

            display_training_stat(self.conf, self.scheduler, self.tracker)
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()