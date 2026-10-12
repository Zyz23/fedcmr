# -*- coding: utf-8 -*-
"""Server implementation of FedAlign (CVPR 2026)."""

import copy

import torch
import torch.distributed as dist

from .master import Master
from pcode.utils.fedalign import (
    aggregate_moments,
    infer_num_channels,
    unpack_moment_packet,
)


class MasterFedAlign(Master):
    """Aggregate private client moments and sample-weighted model updates."""

    def __init__(self, conf):
        super().__init__(conf)
        self._validate_homogeneous_models()

        sample = self.dataset["train"][0]
        sample_input = sample[0] if isinstance(sample, (tuple, list)) else sample
        self.fedalign_num_channels = infer_num_channels(sample_input)
        self.fedalign_sample_counts = {}
        self.fedalign_global_moments = None

        self.conf.logger.log(
            "FedAlign initialized with {} input channel(s); "
            "alignment={}, stat_noise_multiplier={}, "
            "update_noise_multiplier={}.".format(
                self.fedalign_num_channels,
                self.conf.fedalign_moments,
                self.conf.fedalign_stat_noise_multiplier,
                self.conf.fedalign_update_noise_multiplier,
            )
        )

    def _validate_homogeneous_models(self):
        if self.conf.random_reinit_local_model is not None:
            raise ValueError(
                "FedAlign requires --random_reinit_local_model None so the "
                "clipped update is measured from the distributed global model."
            )
        client_archs = set(self.clientid2arch.values())
        if len(client_archs) != 1:
            raise ValueError(
                "FedAlign is a homogeneous method. Use one worker architecture, "
                "for example --complex_arch master=resnet8,worker=resnet8; "
                "received {}.".format(sorted(client_archs))
            )

        client_model = list(self.client_models.values())[0]
        master_state = self.master_model.state_dict()
        client_state = client_model.state_dict()
        if list(master_state.keys()) != list(client_state.keys()):
            raise ValueError("FedAlign master and client state keys must be identical.")
        for key in master_state:
            if tuple(master_state[key].size()) != tuple(client_state[key].size()):
                raise ValueError(
                    "FedAlign shape mismatch for '{}': master={}, client={}.".format(
                        key,
                        tuple(master_state[key].size()),
                        tuple(client_state[key].size()),
                    )
                )

    def _send_model_to_selected_clients(self, selected_client_ids):
        # The base framework constructs the master and client templates with
        # separate random draws.  Synchronizing here guarantees that round 1
        # (and a resumed run) starts from the actual global model.
        master_state = self.master_model.state_dict()
        for client_model in self.client_models.values():
            client_model.load_state_dict(master_state)
        return super()._send_model_to_selected_clients(selected_client_ids)

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        """Receive private local moments, aggregate, then broadcast the result.

        This hook is called before model transmission.  Workers execute the
        matching send-then-receive protocol in ``recv_extra_info_from_master``.
        """
        packet_size = 2 + 4 * self.fedalign_num_channels
        packets = {}
        requests = []

        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            packet = torch.zeros(packet_size, dtype=torch.float32)
            packets[client_id] = packet
            requests.append(dist.irecv(tensor=packet, src=world_id))

        for request in requests:
            request.wait()

        client_moments = []
        unsampled_counts = []
        self.fedalign_sample_counts = {}
        for client_id in selected_client_ids:
            moments, unsampled_count, sampled_count = unpack_moment_packet(
                packets[client_id], self.fedalign_num_channels
            )
            client_moments.append(moments)
            unsampled_counts.append(unsampled_count)
            self.fedalign_sample_counts[client_id] = sampled_count

        self.fedalign_global_moments = aggregate_moments(
            client_moments, unsampled_counts
        )
        flat_global = self.fedalign_global_moments.reshape(-1).contiguous()

        for worker_rank, _ in enumerate(selected_client_ids, start=1):
            dist.send(tensor=flat_global, dst=worker_rank)

        self.conf.logger.log(
            "FedAlign aggregated private moments from clients {} "
            "with unsampled counts {} and sampled counts {}.".format(
                list(selected_client_ids),
                unsampled_counts,
                [self.fedalign_sample_counts[cid] for cid in selected_client_ids],
            )
        )
        dist.barrier()

    def aggregate(self, flatten_local_models):
        """Sample-weighted FedAvg followed by the paper's server step size."""
        client_ids = list(flatten_local_models.keys())
        raw_weights = [
            float(self.fedalign_sample_counts.get(client_id, 1))
            for client_id in client_ids
        ]
        total_weight = sum(raw_weights)
        weights = [weight / total_weight for weight in raw_weights]

        averaged_model = self._fedavg(flatten_local_models, weights=weights)
        server_lr = float(self.conf.fedalign_server_lr)
        if server_lr <= 0:
            raise ValueError("fedalign_server_lr must be positive.")

        if server_lr != 1.0:
            global_state = self.master_model.state_dict()
            averaged_state = averaged_model.state_dict()
            updated_state = copy.deepcopy(global_state)
            for key in global_state:
                if global_state[key].is_floating_point():
                    updated_state[key] = global_state[key] + server_lr * (
                        averaged_state[key] - global_state[key]
                    )
                else:
                    updated_state[key] = averaged_state[key]
            averaged_model.load_state_dict(updated_state)

        architecture = self.clientid2arch[client_ids[0]]
        self.conf.logger.log(
            "FedAlign aggregated clients {} with normalized weights {}.".format(
                client_ids, [round(weight, 6) for weight in weights]
            )
        )
        return {architecture: averaged_model}
