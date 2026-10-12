# -*- coding: utf-8 -*-
"""Client implementation of FedAlign (CVPR 2026)."""

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .worker import Worker
from pcode.utils.fedalign import (
    align_batch,
    clip_and_noise_moments,
    compute_channel_moments,
    infer_num_channels,
    pack_moment_packet,
    privatize_model_state,
    sanitize_moments,
)
from pcode.utils.tensor_buffer import TensorBuffer


class WorkerFedAlign(Worker):
    """Align sampled local data and upload a clipped/noisy local update."""

    def __init__(self, conf):
        super().__init__(conf)
        self.fedalign_local_moments = None
        self.fedalign_global_moments = None
        self.fedalign_local_steps = 0
        self.fedalign_effective_step_size = 0.0

    def _loader_for_subset(self, dataset, shuffle, seed):
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs = {
            "dataset": dataset,
            "batch_size": self.conf.batch_size,
            "shuffle": shuffle,
            "num_workers": self.conf.num_workers,
            "pin_memory": self.conf.pin_memory,
            "drop_last": False,
            "collate_fn": self.train_loader.collate_fn,
            "generator": generator,
        }
        if self.conf.num_workers > 0:
            kwargs["multiprocessing_context"] = "fork"
        return DataLoader(**kwargs)

    def _split_sampled_and_unsampled_data(self):
        ratio = float(self.conf.fedalign_unsampled_ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError("fedalign_unsampled_ratio must be in (0, 1).")

        full_dataset = self.train_loader.dataset
        num_examples = len(full_dataset)
        if num_examples < 2:
            raise ValueError("FedAlign needs at least two examples per client.")

        split_seed = (
            int(self.conf.fedalign_seed)
            + 1000003 * int(self.conf.graph.comm_round)
            + 9176 * int(self.conf.graph.client_id)
        )
        random_state = np.random.RandomState(split_seed)
        indices = random_state.permutation(num_examples).tolist()
        num_unsampled = int(round(num_examples * ratio))
        num_unsampled = min(max(1, num_unsampled), num_examples - 1)

        unsampled_dataset = Subset(full_dataset, indices[:num_unsampled])
        sampled_dataset = Subset(full_dataset, indices[num_unsampled:])
        self.fedalign_stat_loader = self._loader_for_subset(
            unsampled_dataset, shuffle=False, seed=split_seed + 1
        )
        self.train_loader = self._loader_for_subset(
            sampled_dataset, shuffle=True, seed=split_seed + 2
        )

        # The local scheduler derives epochs from this value.
        self.conf.num_batches_per_device_per_epoch = len(self.train_loader)
        self.conf.num_whole_batches_per_worker = (
            len(self.train_loader) * self.conf.local_n_epochs
        )
        return len(sampled_dataset)

    def recv_extra_info_from_master(self):
        """Send local DP moments and receive selected-client global moments."""
        sampled_count = self._split_sampled_and_unsampled_data()
        local_moments, unsampled_count = compute_channel_moments(
            self.fedalign_stat_loader,
            max_batches=self.conf.fedalign_max_stat_batches,
        )
        self.fedalign_local_moments = local_moments

        noise_seed = (
            int(self.conf.fedalign_seed)
            + 2000003 * int(self.conf.graph.comm_round)
            + 193 * int(self.conf.graph.client_id)
        )
        private_moments = clip_and_noise_moments(
            local_moments,
            clip_norm=self.conf.fedalign_stat_clip,
            noise_multiplier=self.conf.fedalign_stat_noise_multiplier,
            seed=noise_seed,
        )
        packet = pack_moment_packet(
            private_moments,
            unsampled_count=unsampled_count,
            sampled_count=sampled_count,
        )
        dist.send(tensor=packet, dst=0)

        num_channels = infer_num_channels(self.fedalign_stat_loader.dataset[0][0])
        flat_global = torch.zeros(4 * num_channels, dtype=torch.float32)
        dist.recv(tensor=flat_global, src=0)
        self.fedalign_global_moments = sanitize_moments(
            flat_global.view(4, num_channels)
        )

        self.conf.logger.log(
            "Worker-{} (client-{}) sent FedAlign moments using {} unsampled "
            "examples and retained {} sampled examples.".format(
                self.conf.graph.worker_id,
                self.conf.graph.client_id,
                unsampled_count,
                sampled_count,
            )
        )
        dist.barrier()

    def prepare_train(self):
        self._prepare_train()
        self.fedalign_local_steps = 0
        self.fedalign_effective_step_size = 0.0
        original_step = self.optimizer.step

        def tracked_step(*args, **kwargs):
            learning_rates = {
                float(group["lr"]) for group in self.optimizer.param_groups
            }
            if len(learning_rates) != 1:
                raise ValueError(
                    "FedAlign currently requires one shared local learning rate."
                )
            self.fedalign_effective_step_size += learning_rates.pop()
            self.fedalign_local_steps += 1
            return original_step(*args, **kwargs)

        tracked_step._with_counter = getattr(original_step, "_with_counter", False)
        self.optimizer.step = tracked_step

    def _inference(self, data_batch):
        if self.fedalign_local_moments is None or self.fedalign_global_moments is None:
            raise RuntimeError("FedAlign moments were not exchanged before training.")
        data_batch["input"] = align_batch(
            data_batch["input"],
            local_moments=self.fedalign_local_moments,
            global_moments=self.fedalign_global_moments,
            mode=self.conf.fedalign_moments,
            strength=self.conf.fedalign_alignment_strength,
        )
        return super()._inference(data_batch)

    def _send_model_to_master(self):
        dist.barrier()
        if self.fedalign_local_steps <= 0:
            raise RuntimeError("FedAlign local training produced no optimizer steps.")

        local_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        global_state = {
            key: value.detach().cpu().clone()
            for key, value in self.init_model.state_dict().items()
        }
        noise_seed = (
            int(self.conf.fedalign_seed)
            + 3000017 * int(self.conf.graph.comm_round)
            + 389 * int(self.conf.graph.client_id)
        )
        private_state, privacy_metrics = privatize_model_state(
            local_state=local_state,
            global_state=global_state,
            effective_step_size=self.fedalign_effective_step_size,
            clip_norm=self.conf.fedalign_update_clip,
            noise_multiplier=self.conf.fedalign_update_noise_multiplier,
            seed=noise_seed,
        )
        flatten_model = TensorBuffer(list(private_state.values()), use_cuda=False)
        dist.send(tensor=flatten_model.buffer, dst=0)

        self.conf.logger.log(
            "Worker-{} (client-{}) sent FedAlign update: steps={}, "
            "effective_step_size={:.6g}, norm={:.6g}, clip_factor={:.6g}, "
            "noise_std={:.6g}.".format(
                self.conf.graph.worker_id,
                self.conf.graph.client_id,
                self.fedalign_local_steps,
                self.fedalign_effective_step_size,
                privacy_metrics["update_norm"],
                privacy_metrics["clip_factor"],
                privacy_metrics["noise_std"],
            )
        )
        dist.barrier()
