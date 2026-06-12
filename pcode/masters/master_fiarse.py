# -*- coding: utf-8 -*-
import copy

import torch
import torch.distributed as dist

import pcode.master_utils as master_utils
from pcode.masters.master import Master
from pcode.utils.fiarse import (
    aggregate_fiarse_states,
    build_modelwise_fiarse_mask,
    mask_state_dict,
)
from pcode.utils.tensor_buffer import TensorBuffer


class MasterFIARSE(Master):
    """FIARSE server with model-wise importance submodel extraction."""

    def __init__(self, conf):
        super(MasterFIARSE, self).__init__(conf)
        self.fiarse_parameter_names = {
            name for name, _ in self.master_model.named_parameters()
        }
        conf.logger.log(
            "FIARSE uses the full architecture from create_model; client capacity "
            "is applied as an importance mask, not as a width-scaled model."
        )

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        global_state = self.master_model.state_dict()
        self.current_masks = {}
        self.current_thresholds = {}

        threshold_msg = torch.zeros(self.conf.n_participated)
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            capacity = float(self.conf.model_rate_list[selected_client_id - 1])
            mask, threshold = build_modelwise_fiarse_mask(
                global_state,
                capacity,
                parameter_names=self.fiarse_parameter_names,
            )
            self.current_masks[selected_client_id] = mask
            self.current_thresholds[selected_client_id] = threshold
            threshold_msg[worker_rank - 1] = threshold

        dist.broadcast(tensor=threshold_msg, src=0)
        self.conf.logger.log(
            "FIARSE thresholds: "
            + str({client_id: self.current_thresholds[client_id] for client_id in selected_client_ids})
        )

    def _send_model_to_selected_clients(self, selected_client_ids):
        self.conf.logger.log("Master sends FIARSE sparse submodels to workers.")
        global_state = self.master_model.state_dict()

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            masked_state = mask_state_dict(global_state, self.current_masks[selected_client_id])
            client_model_state = copy.deepcopy(self.client_models[self.clientid2arch[selected_client_id]].state_dict())
            client_model_state.update(masked_state)
            flatten_model = TensorBuffer(list(client_model_state.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

            capacity = float(self.conf.model_rate_list[selected_client_id - 1])
            self.conf.logger.log(
                f"\tMaster sent FIARSE capacity={capacity} sparse model to client={selected_client_id}, process_id={worker_rank}."
            )

        dist.barrier()

    def aggregate(self, flatten_local_models):
        local_state_dicts = []
        masks = []

        for client_id, flatten_local_model in flatten_local_models.items():
            arch = self.clientid2arch[client_id]
            client_model = self.client_models[arch]
            client_state = copy.deepcopy(client_model.state_dict())
            flatten_local_model.unpack(client_state.values())
            client_model.load_state_dict(client_state)
            local_state_dicts.append(client_state)
            masks.append(self.current_masks[client_id])

        aggregated_state = aggregate_fiarse_states(
            self.master_model.state_dict(),
            local_state_dicts,
            masks,
            global_lr=getattr(self.conf, "fiarse_global_lr", 1.0),
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