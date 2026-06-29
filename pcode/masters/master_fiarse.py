# -*- coding: utf-8 -*-

import copy
import torch
import torch.distributed as dist

import pcode.master_utils as master_utils

from pcode.masters.master import Master
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.fiarse import (
    aggregate_fiarse_deltas,
    build_modelwise_fiarse_threshold,
    first_nonfinite_parameter_name,
    mask_state_dict_by_threshold,
)


class MasterFIARSE(Master):
    """
    Correct model-wise FIARSE server.

    Protocol:
    1. Build a client-specific threshold theta_i from global model x_t.
    2. Send sparse model x_t ⊙ M_i(x_t).
    3. Receive local delta:
           Delta_i = x_t ⊙ M_i(x_t) - x_{t,K}^{(i)}.
    4. Coordinate-wise aggregate nonzero deltas.
    """

    def __init__(self, conf):
        super(MasterFIARSE, self).__init__(conf)

        self.fiarse_parameter_names = {
            name
            for name, _ in self.master_model.named_parameters()
        }

        self.current_thresholds = {}

        self.conf.logger.log(
            "FIARSE initialized: full architecture + "
            "model-wise magnitude sparsification."
        )

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        """
        Build one threshold for each selected client and broadcast them.

        Worker rank r receives threshold_msg[r - 1].
        """
        global_state = self.master_model.state_dict()

        self.current_thresholds = {}

        threshold_message = torch.zeros(
            self.conf.n_participated,
            dtype=torch.float32,
        )

        for worker_rank, client_id in enumerate(
            selected_client_ids,
            start=1,
        ):
            capacity = float(
                self.conf.model_rate_list[client_id - 1]
            )

            threshold = build_modelwise_fiarse_threshold(
                global_state,
                parameter_names=self.fiarse_parameter_names,
                capacity=capacity,
            )

            self.current_thresholds[client_id] = threshold
            threshold_message[worker_rank - 1] = threshold

        dist.broadcast(
            tensor=threshold_message,
            src=0,
        )

        threshold_log = {
            client_id: round(
                self.current_thresholds[client_id],
                8,
            )
            for client_id in selected_client_ids
        }

        self.conf.logger.log(
            f"FIARSE selected-client thresholds: {threshold_log}."
        )

    def _send_model_to_selected_clients(self, selected_client_ids):
        """
        Send x_t ⊙ M_i(x_t) to every selected client.

        The communication payload remains dense because the existing
        TensorBuffer protocol requires a fixed-size tensor.
        """
        self.conf.logger.log(
            "Master sends FIARSE sparse global models."
        )

        global_state = self.master_model.state_dict()

        for worker_rank, client_id in enumerate(
            selected_client_ids,
            start=1,
        ):
            arch = self.clientid2arch[client_id]
            threshold = self.current_thresholds[client_id]

            sparse_global_state = mask_state_dict_by_threshold(
                global_state,
                parameter_names=self.fiarse_parameter_names,
                threshold=threshold,
                to_cpu=True,
            )

            payload_state = copy.deepcopy(
                self.client_models[arch].state_dict()
            )

            for name in payload_state.keys():
                if name not in sparse_global_state:
                    raise RuntimeError(
                        f"Missing FIARSE state '{name}' in global model."
                    )

                if (
                    payload_state[name].shape
                    != sparse_global_state[name].shape
                ):
                    raise RuntimeError(
                        f"FIARSE model shape mismatch for '{name}': "
                        f"template={tuple(payload_state[name].shape)}, "
                        f"global={tuple(sparse_global_state[name].shape)}."
                    )

                payload_state[name] = sparse_global_state[name]

            flatten_model = TensorBuffer(
                list(payload_state.values()),
                use_cuda=False,
            )

            flatten_model.buffer = flatten_model.buffer.cpu()

            dist.send(
                tensor=flatten_model.buffer,
                dst=worker_rank,
            )

            capacity = float(
                self.conf.model_rate_list[client_id - 1]
            )

            self.conf.logger.log(
                f"\tMaster sent FIARSE sparse model to "
                f"client={client_id}, worker_rank={worker_rank}, "
                f"capacity={capacity}, threshold={threshold:.8f}, "
                f"numel={flatten_model.buffer.numel()}."
            )

        dist.barrier()

    def aggregate(self, flatten_local_models):
        """
        Decode received FIARSE deltas and aggregate them into x_{t+1}.
        """
        local_delta_states = []

        for client_id, flatten_delta in flatten_local_models.items():
            arch = self.clientid2arch[client_id]

            delta_state = copy.deepcopy(
                self.client_models[arch].state_dict()
            )

            flatten_delta.unpack(delta_state.values())

            local_delta_states.append(delta_state)

        aggregated_state = aggregate_fiarse_deltas(
            global_state_dict=self.master_model.state_dict(),
            local_delta_states=local_delta_states,
            parameter_names=self.fiarse_parameter_names,
            global_lr=getattr(
                self.conf,
                "fiarse_global_lr",
                1.0,
            ),
        )

        aggregated_model = copy.deepcopy(self.master_model)
        aggregated_model.load_state_dict(
            aggregated_state,
            strict=True,
        )

        return {0: aggregated_model}

    def _aggregate_model_and_evaluate(
        self,
        flatten_local_models,
        selected_client_ids,
    ):
        """
        Reject a non-finite aggregated result instead of contaminating
        the global model.
        """
        client_models = self.aggregate(flatten_local_models)

        candidate_model = list(client_models.values())[0]

        bad_parameter = first_nonfinite_parameter_name(
            candidate_model.state_dict(),
            self.fiarse_parameter_names,
        )

        if bad_parameter is not None:
            self.conf.logger.log(
                f"[FIARSE SERVER] Reject round "
                f"{self.conf.graph.comm_round}: aggregated model "
                f"contains NaN/Inf at '{bad_parameter}'."
            )
            return

        self.master_model.load_state_dict(
            candidate_model.state_dict(),
            strict=True,
        )

        master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label="aggregated_test_loader_0",
        )

        if self.conf.graph.on_cuda:
            torch.cuda.empty_cache()