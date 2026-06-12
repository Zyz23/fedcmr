# -*- coding: utf-8 -*-
import copy

import torch
import torch.distributed as dist

import pcode.create_model as create_model
import pcode.master_utils as master_utils
from pcode.masters.master import Master
from pcode.utils.heterofl import aggregate_submodel_states, extract_submodel_state
from pcode.utils.tensor_buffer import TensorBuffer


class MasterHeteroFL(Master):
    """HeteroFL baseline with nested width submodels.

    Diao et al.'s HeteroFL trains clients with different model widths and
    aggregates the shared prefix parameters back into a full-width server model.
    """

    def __init__(self, conf):
        conf.model_rate = 1.0
        super(MasterHeteroFL, self).__init__(conf)

        self.client_models = {}
        for client_id in self.client_ids:
            _, model = create_model.define_model(
                conf, show_stat=False, to_consistent_model=False, client_id=client_id
            )
            self.client_models[client_id] = model

        self.clientid2arch = {
            client_id: create_model.determine_arch(conf, client_id, use_complex_arch=True)
            for client_id in self.client_ids
        }
        self.conf.clientid2arch = self.clientid2arch

        if not hasattr(conf, "model_rate_list") or conf.model_rate_list is None:
            conf.model_rate_list = [1.0 for _ in self.client_ids]

        conf.logger.log(
            "HeteroFL client model rates: "
            + str({client_id: float(conf.model_rate_list[client_id - 1]) for client_id in self.client_ids})
        )

    def _send_model_to_selected_clients(self, selected_client_ids):
        self.conf.logger.log("Master sends HeteroFL submodels to workers.")
        global_state = self.master_model.state_dict()

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model = self.client_models[selected_client_id]
            client_state = extract_submodel_state(global_state, client_model.state_dict())
            client_model.load_state_dict(client_state)

            flatten_model = TensorBuffer(list(client_model.state_dict().values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

            rate = float(self.conf.model_rate_list[selected_client_id - 1])
            self.conf.logger.log(
                f"\tMaster sent HeteroFL rate={rate} model to client={selected_client_id}, process_id={worker_rank}."
            )

        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log("Master waits to receive HeteroFL local submodels.")
        dist.barrier()

        flatten_local_models = {}
        for selected_client_id in selected_client_ids:
            client_tb = TensorBuffer(
                list(self.client_models[selected_client_id].state_dict().values())
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb

        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            reqs.append(dist.irecv(tensor=flatten_local_models[client_id].buffer, src=world_id))

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log("Master received all HeteroFL local submodels.")
        return flatten_local_models

    def aggregate(self, flatten_local_models):
        local_state_dicts = []
        for client_id, flatten_local_model in flatten_local_models.items():
            client_model = self.client_models[client_id]
            client_state = copy.deepcopy(client_model.state_dict())
            flatten_local_model.unpack(client_state.values())
            client_model.load_state_dict(client_state)
            local_state_dicts.append(client_state)

        aggregated_state = aggregate_submodel_states(
            self.master_model.state_dict(), local_state_dicts
        )
        aggregated_model = copy.deepcopy(self.master_model)
        aggregated_model.load_state_dict(aggregated_state)
        return {0: aggregated_model}

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        client_models = self.aggregate(flatten_local_models)
        self.master_model.load_state_dict(list(client_models.values())[0].state_dict())

        master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label="aggregated_test_loader_0",
        )

        torch.cuda.empty_cache()