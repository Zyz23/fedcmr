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
    """
    FedRolex baseline with rolling submodel extraction.

    修改版功能：
    1. 保留 FedRolex rolling 子模型机制；
    2. 保留 HeteroFL 的异构宽度客户端；
    3. 每轮按 model_rate 分层采样，减少容量组合随机性；
    4. 增加 send / recv / aggregate 日志，方便确认 buffer 大小和 rate 是否对应；
    5. 不引入 FedOur、CLIP、anchor、distillation 等任何组件。
    """

    def _random_select_clients(self):
        """
        FedRolex 分层采样。

        目的：
        n_clients=20, participation_ratio=0.2 时，每轮只选 4 个客户端。
        如果完全随机，可能某些轮没有 1.0 客户端，或者容量组合严重不均衡。
        对 FedRolex / HeteroFL 这种子模型方法来说，这会影响全局模型不同部分的更新。

        当前策略：
        - 先按 model_rate 分组；
        - 每轮尽量从每个容量组中选 1 个客户端；
        - 如果参与客户端数量大于容量组数量，则再随机补齐；
        - 如果参与客户端数量小于容量组数量，则随机打乱容量组后选择前几个。
        """

        model_rate_list = self.conf.model_rate_list

        rate_to_clients = {}

        for client_id in self.client_ids:
            rate = float(model_rate_list[client_id - 1])

            if rate not in rate_to_clients:
                rate_to_clients[rate] = []

            rate_to_clients[rate].append(client_id)

        rates = list(rate_to_clients.keys())

        # 打乱容量组顺序，避免 n_participated < rate 组数时总是偏向某些容量
        self.conf.random_state.shuffle(rates)

        selected_client_ids = []

        # 每个容量组优先选一个客户端
        for rate in rates:
            if len(selected_client_ids) >= self.conf.n_participated:
                break

            candidates = rate_to_clients[rate]

            chosen_client = self.conf.random_state.choice(
                candidates,
                size=1,
                replace=False,
            )[0]

            selected_client_ids.append(int(chosen_client))

        # 如果还没有选够，从剩余客户端中随机补齐
        if len(selected_client_ids) < self.conf.n_participated:
            remaining_clients = [
                client_id
                for client_id in self.client_ids
                if client_id not in selected_client_ids
            ]

            num_extra = self.conf.n_participated - len(selected_client_ids)

            extra_clients = self.conf.random_state.choice(
                remaining_clients,
                size=num_extra,
                replace=False,
            ).tolist()

            selected_client_ids.extend([int(client_id) for client_id in extra_clients])

        selected_client_ids = sorted(selected_client_ids)

        selected_rates = [
            float(model_rate_list[client_id - 1])
            for client_id in selected_client_ids
        ]

        self.conf.logger.log(
            f"FedRolex stratified selected "
            f"{self.conf.n_participated} from {self.conf.n_clients} clients: "
            f"{selected_client_ids}, rates={selected_rates}."
        )

        return selected_client_ids

    def _send_model_to_selected_clients(self, selected_client_ids):
        """
        发送 FedRolex rolling 子模型。

        和 HeteroFL 的区别：
        - HeteroFL 是固定 prefix slicing；
        - FedRolex 是 rolling slicing，每轮从 full model 的不同位置抽取子模型；
        - index_map 会保存下来，聚合时需要用它把本地子模型放回 full model 的对应位置。
        """

        self.conf.logger.log("Master sends FedRolex rolling submodels to workers.")

        global_state = self.master_model.state_dict()

        # 保存当前轮每个 client 的 rolling index map
        self.current_index_maps = {}

        roll_step = getattr(self.conf, "fedrolex_roll_step", 1)

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model = self.client_models[selected_client_id]
            client_template_state = client_model.state_dict()

            rate = float(self.conf.model_rate_list[selected_client_id - 1])

            client_state, index_map = extract_rolling_submodel_state(
                global_state,
                client_template_state,
                self.conf.graph.comm_round,
                roll_step=roll_step,
            )

            self.current_index_maps[selected_client_id] = index_map

            client_model.load_state_dict(client_state)

            flatten_model = TensorBuffer(
                list(client_model.state_dict().values())
            )

            self.conf.logger.log(
                f"[FEDROLEX SEND] "
                f"round={self.conf.graph.comm_round}, "
                f"worker_rank={worker_rank}, "
                f"client_id={selected_client_id}, "
                f"rate={rate}, "
                f"roll_step={roll_step}, "
                f"numel={flatten_model.buffer.numel()}, "
                f"bytes={flatten_model.buffer.numel() * flatten_model.buffer.element_size()}."
            )

            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

            self.conf.logger.log(
                f"\tMaster sent FedRolex rate={rate} rolling model "
                f"to client={selected_client_id}, process_id={worker_rank}."
            )

        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        """
        接收 FedRolex 客户端训练后的 rolling 子模型。

        接收 buffer 必须和对应 client_id 的模型大小完全一致。
        这里加日志用于排查通信或模型大小是否错位。
        """

        self.conf.logger.log("Master waits to receive FedRolex local rolling submodels.")

        dist.barrier()

        flatten_local_models = {}

        # 为每个 selected client 准备对应大小的接收 buffer
        for selected_client_id in selected_client_ids:
            client_tb = TensorBuffer(
                list(self.client_models[selected_client_id].state_dict().values())
            )

            client_tb.buffer = torch.zeros_like(client_tb.buffer)

            flatten_local_models[selected_client_id] = client_tb

            rate = float(self.conf.model_rate_list[selected_client_id - 1])

            self.conf.logger.log(
                f"[FEDROLEX RECV-BUF] "
                f"client_id={selected_client_id}, "
                f"rate={rate}, "
                f"numel={client_tb.buffer.numel()}, "
                f"bytes={client_tb.buffer.numel() * client_tb.buffer.element_size()}."
            )

        # 按 selected_client_ids 和 world_ids 的对应关系接收
        reqs = []

        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer,
                src=world_id,
            )
            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()

        self.conf.logger.log("Master received all FedRolex local rolling submodels.")

        return flatten_local_models

    def aggregate(self, flatten_local_models):
        """
        聚合 FedRolex rolling 子模型。

        聚合逻辑：
        1. 把每个客户端上传的 flatten buffer 解包成 local_state_dict；
        2. 取出该 client 在发送时保存的 index_map；
        3. 使用 aggregate_rolling_submodel_states 按 index_map 聚合回 full model。
        """

        local_state_dicts = []
        index_maps = []

        for client_id, flatten_local_model in flatten_local_models.items():
            client_model = self.client_models[client_id]

            client_state = copy.deepcopy(client_model.state_dict())

            flatten_local_model.unpack(client_state.values())

            client_model.load_state_dict(client_state)

            local_state_dicts.append(client_state)

            if client_id not in self.current_index_maps:
                raise RuntimeError(
                    f"Missing FedRolex index_map for client_id={client_id}. "
                    f"Please check whether _send_model_to_selected_clients() "
                    f"was called before aggregate()."
                )

            index_maps.append(self.current_index_maps[client_id])

            rate = float(self.conf.model_rate_list[client_id - 1])

            self.conf.logger.log(
                f"[FEDROLEX AGG] "
                f"client_id={client_id}, "
                f"rate={rate}."
            )

        aggregated_state = aggregate_rolling_submodel_states(
            self.master_model.state_dict(),
            local_state_dicts,
            index_maps,
        )

        aggregated_model = copy.deepcopy(self.master_model)

        aggregated_model.load_state_dict(aggregated_state)

        return {0: aggregated_model}