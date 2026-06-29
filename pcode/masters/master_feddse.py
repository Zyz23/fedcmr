# -*- coding: utf-8 -*-

import copy

import torch
import torch.distributed as dist

from pcode.masters.master_heterofl import MasterHeteroFL
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.feddse import (
    NEURON_LAYERS,
    aggregate_indexed_submodel_states,
    build_cnn_index_map,
    first_nonfinite_state_key,
)


class MasterFedDSE(MasterHeteroFL):
    """
    FedDSE Server.

    服务器：
    1. 始终维护完整 global model；
    2. 每轮向客户端发送完整模型；
    3. 接收客户端的神经元选择索引；
    4. 接收客户端训练后的子模型；
    5. 对被选择的坐标进行逐元素平均聚合；
    6. 在服务器完整模型上统一测试。
    """

    def __init__(self, conf):
        super(MasterFedDSE, self).__init__(conf)

        self.current_neuron_selections = {}
        self.current_index_maps = {}

        self.conf.logger.log(
            "FedDSE initialized: full-model download, "
            "activation-aware neuron extraction, "
            "coordinate-wise aggregation."
        )

    def send_extra_info_to_selected_clients(
        self,
        selected_client_ids,
    ):
        """
        FedDSE 不需要服务器额外下发 mask。

        客户端收到完整模型后，利用自己本地数据统计激活，
        自主选择符合容量约束的神经元。
        """
        return

    def _send_model_to_selected_clients(
        self,
        selected_client_ids,
    ):
        """
        按 FedDSE 协议，服务器向每个参与客户端发送完整模型。
        """
        self.conf.logger.log(
            "Master sends full global model for FedDSE profiling."
        )

        global_state = self.master_model.state_dict()

        flatten_global_model = TensorBuffer(
            list(global_state.values())
        )

        for worker_rank, client_id in enumerate(
            selected_client_ids,
            start=1,
        ):
            dist.send(
                tensor=flatten_global_model.buffer,
                dst=worker_rank,
            )

            rate = float(
                self.conf.model_rate_list[client_id - 1]
            )

            self.conf.logger.log(
                f"\tFedDSE sent full model to "
                f"client={client_id}, "
                f"worker_rank={worker_rank}, "
                f"capacity={rate}."
            )

        dist.barrier()

    def _selection_sizes_for_client(self, client_id):
        local_model = self.client_models[client_id]

        return {
            "conv1": int(local_model.conv1.out_channels),
            "conv2": int(local_model.conv2.out_channels),
            "conv3": int(local_model.conv3.out_channels),
            "conv4": int(local_model.conv4.out_channels),
            "fc1": int(local_model.fc1.out_features),
        }

    def receive_extra_info_from_selected_clients(
        self,
        selected_client_ids,
    ):
        """
        接收每个客户端选择的五组神经元索引：
        conv1 / conv2 / conv3 / conv4 / fc1。
        """
        self.conf.logger.log(
            "Master receives FedDSE neuron selections."
        )

        selection_buffers = {}
        requests = []

        for client_id, world_id in zip(
            selected_client_ids,
            self.world_ids,
        ):
            sizes = self._selection_sizes_for_client(
                client_id
            )

            selection_buffers[client_id] = {}

            for layer_name in NEURON_LAYERS:
                buffer = torch.empty(
                    sizes[layer_name],
                    dtype=torch.long,
                )

                selection_buffers[client_id][
                    layer_name
                ] = buffer

                requests.append(
                    dist.irecv(
                        tensor=buffer,
                        src=world_id,
                    )
                )

        for request in requests:
            request.wait()

        dist.barrier()

        self.current_neuron_selections = {}
        self.current_index_maps = {}

        global_state = self.master_model.state_dict()

        for client_id in selected_client_ids:
            selections = {
                layer_name: selection_buffers[
                    client_id
                ][layer_name].cpu()
                for layer_name in NEURON_LAYERS
            }

            local_state = self.client_models[
                client_id
            ].state_dict()

            index_map = build_cnn_index_map(
                global_state_dict=global_state,
                local_state_dict=local_state,
                selections=selections,
            )

            self.current_neuron_selections[
                client_id
            ] = selections

            self.current_index_maps[
                client_id
            ] = index_map

            selection_info = {
                layer_name: int(
                    selections[layer_name].numel()
                )
                for layer_name in NEURON_LAYERS
            }

            self.conf.logger.log(
                f"[FedDSE Selection] "
                f"client={client_id}, "
                f"selected={selection_info}."
            )

    def aggregate(self, flatten_local_models):
        """
        使用客户端选择的 index_map 聚合任意通道子模型。
        """
        local_state_dicts = []
        index_maps = []

        for client_id, flatten_local_model in (
            flatten_local_models.items()
        ):
            if client_id not in self.current_index_maps:
                raise RuntimeError(
                    f"FedDSE index map missing for "
                    f"client={client_id}."
                )

            client_model = self.client_models[client_id]

            client_state = copy.deepcopy(
                client_model.state_dict()
            )

            flatten_local_model.unpack(
                client_state.values()
            )

            bad_key = first_nonfinite_state_key(
                client_state
            )

            if bad_key is not None:
                raise RuntimeError(
                    f"FedDSE local update from client={client_id} "
                    f"contains NaN/Inf at '{bad_key}'."
                )

            client_model.load_state_dict(
                client_state,
                strict=True,
            )

            local_state_dicts.append(client_state)
            index_maps.append(
                self.current_index_maps[client_id]
            )

        aggregated_state = aggregate_indexed_submodel_states(
            global_state_dict=self.master_model.state_dict(),
            local_state_dicts=local_state_dicts,
            index_maps=index_maps,
        )

        aggregated_model = copy.deepcopy(
            self.master_model
        )

        aggregated_model.load_state_dict(
            aggregated_state,
            strict=True,
        )

        bad_key = first_nonfinite_state_key(
            aggregated_model.state_dict()
        )

        if bad_key is not None:
            raise RuntimeError(
                f"FedDSE aggregated model contains "
                f"NaN/Inf at '{bad_key}'."
            )

        return {0: aggregated_model}