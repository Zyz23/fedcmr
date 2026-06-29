import copy

import torch
import torch.distributed as dist

from .master import Master
from pcode.utils.tensor_buffer import TensorBuffer


class MasterFedSWA(Master):
    """FedSWA uses the standard server-side FedAvg aggregation.

    The algorithmic change is in the worker: local SGD follows the stochastic
    weight averaging schedule from the official implementation.
    """

    pass


class MasterFedMoSWA(Master):
    """FedMoSWA server with the official momentum-based control update."""

    def __init__(self, conf):
        super().__init__(conf)
        if len(self.used_client_archs) != 1:
            raise ValueError(
                "FedMoSWA is implemented for homogeneous models. "
                f"Got client archs={self.used_client_archs}."
            )
        self.control_names = [name for name, _ in self.master_model.named_parameters()]
        self.global_control = {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in self.master_model.named_parameters()
        }
        self._control_initialized = False

    def _control_tensors(self):
        return [self.global_control[name] for name in self.control_names]

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        for worker_rank, _ in enumerate(selected_client_ids, start=1):
            control_tb = TensorBuffer(self._control_tensors(), use_cuda=False)
            dist.send(tensor=control_tb.buffer, dst=worker_rank)

        self.conf.logger.log("MasterFedMoSWA sent the global control variate to workers.")
        dist.barrier()

    def receive_extra_info_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log("MasterFedMoSWA waits to receive local control updates.")

        control_updates = {}
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            zero_controls = [
                torch.zeros_like(self.global_control[name])
                for name in self.control_names
            ]
            control_tb = TensorBuffer(zero_controls, use_cuda=False)
            control_updates[client_id] = control_tb
            reqs.append(dist.irecv(tensor=control_tb.buffer, src=world_id))

        for req in reqs:
            req.wait()

        avg_update = {}
        for name in self.control_names:
            avg_update[name] = torch.zeros_like(self.global_control[name])

        for control_tb in control_updates.values():
            unpacked = {
                name: torch.zeros_like(self.global_control[name])
                for name in self.control_names
            }
            control_tb.unpack(unpacked.values(), use_cuda=False)
            for name in self.control_names:
                avg_update[name].add_(unpacked[name], alpha=1.0 / len(control_updates))

        if self._control_initialized:
            momentum = self.conf.fedmoswa_control_momentum
            for name in self.control_names:
                self.global_control[name].add_(avg_update[name], alpha=momentum)
        else:
            self.global_control = {
                name: value.detach().clone() for name, value in avg_update.items()
            }
            self._control_initialized = True

        self.conf.logger.log(
            "MasterFedMoSWA updated the global control variate "
            f"(momentum={self.conf.fedmoswa_control_momentum})."
        )
        dist.barrier()

    def aggregate(self, flatten_local_models):
        return super().aggregate(flatten_local_models)