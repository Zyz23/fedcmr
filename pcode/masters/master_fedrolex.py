# -*- coding: utf-8 -*-
import copy

import torch
import torch.distributed as dist

from pcode.masters.master_heterofl import MasterHeteroFL
from pcode.utils.heterofl import (
    aggregate_rolling_submodel_states,
    extract_rolling_submodel_state,
)
from pcode.utils.tensor_buffer import TensorBuffer


class MasterFedRolex(MasterHeteroFL):
    """FedRolex baseline with rolling submodel extraction.

    FedRolex keeps HeteroFL's width-heterogeneous clients but rotates the
    selected parameter/channel subset across communication rounds.
    """

    def _send_model_to_selected_clients(self, selected_client_ids):
        self.conf.logger.log("Master sends FedRolex rolling submodels to workers.")
        global_state = self.master_model.state_dict()
        self.current_index_maps = {}

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model = self.client_models[selected_client_id]
            client_state, index_map = extract_rolling_submodel_state(
                global_state,
                client_model.state_dict(),
                self.conf.graph.comm_round,
                roll_step=getattr(self.conf, "fedrolex_roll_step", 1),
            )
            self.current_index_maps[selected_client_id] = index_map
            client_model.load_state_dict(client_state)

            flatten_model = TensorBuffer(list(client_model.state_dict().values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

            rate = float(self.conf.model_rate_list[selected_client_id - 1])
            self.conf.logger.log(
                f"\tMaster sent FedRolex rate={rate} rolling model to client={selected_client_id}, process_id={worker_rank}."
            )

        dist.barrier()

    def aggregate(self, flatten_local_models):
        local_state_dicts = []
        index_maps = []

        for client_id, flatten_local_model in flatten_local_models.items():
            client_model = self.client_models[client_id]
            client_state = copy.deepcopy(client_model.state_dict())
            flatten_local_model.unpack(client_state.values())
            client_model.load_state_dict(client_state)
            local_state_dicts.append(client_state)
            index_maps.append(self.current_index_maps[client_id])

        aggregated_state = aggregate_rolling_submodel_states(
            self.master_model.state_dict(),
            local_state_dicts,
            index_maps,
        )
        aggregated_model = copy.deepcopy(self.master_model)
        aggregated_model.load_state_dict(aggregated_state)
        return {0: aggregated_model}