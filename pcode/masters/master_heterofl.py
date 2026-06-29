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
    """
    HeteroFL baseline with nested width submodels.

    版本 A：
    - 保持标准 HeteroFL 思路；
    - 不加入 CLIP / anchor / prototype / FedOur 蒸馏；
    - 服务器维护完整模型；
    - 客户端使用 width_mult = 0.25 / 0.5 / 0.75 / 1.0 的子模型；
    - 每轮按容量档位分层采样，减少准确率大幅波动。
    """

    def __init__(self, conf):
        # Master 一定是 full model
        conf.model_rate = 1.0

        super(MasterHeteroFL, self).__init__(conf)

        # 为每个 client_id 创建对应宽度的模型模板
        # 注意：client_id 是 1-based，即 1 ~ n_clients
        self.client_models = {}

        for client_id in self.client_ids:
            _, model = create_model.define_model(
                conf,
                show_stat=False,
                to_consistent_model=False,
                client_id=client_id,
            )
            self.client_models[client_id] = model

        # 记录每个客户端对应的结构名
        self.clientid2arch = {
            client_id: create_model.determine_arch(
                conf,
                client_id,
                use_complex_arch=True,
            )
            for client_id in self.client_ids
        }

        self.conf.clientid2arch = self.clientid2arch

        # 如果没有 model_rate_list，则默认所有客户端都是 full model
        if not hasattr(conf, "model_rate_list") or conf.model_rate_list is None:
            conf.model_rate_list = [1.0 for _ in self.client_ids]

        self.conf.logger.log(
            "HeteroFL client model rates: "
            + str(
                {
                    client_id: float(conf.model_rate_list[client_id - 1])
                    for client_id in self.client_ids
                }
            )
        )

    def _random_select_clients(self):
        """
        按容量档位分层采样客户端。

        原始随机采样的问题：
        - n_clients=20, participation_ratio=0.2 时，每轮只选 4 个客户端；
        - 若完全随机，可能某一轮几乎全是 0.25 / 0.5 小模型；
        - HeteroFL 的 prefix 聚合会导致 full model 后半通道长期更新不足；
        - CIFAR100 + non-IID alpha=0.1 下会导致准确率波动很大。

        现在的采样方式：
        - 先按 model_rate 分组；
        - 每轮优先从每个容量组里选 1 个客户端；
        - 如果参与客户端数大于容量组数，再从剩余客户端中随机补齐；
        - 如果参与客户端数小于容量组数，则随机选择部分类别容量组。
        """

        model_rate_list = self.conf.model_rate_list

        # 1. 按容量档位分组：rate -> [client_id]
        rate_to_clients = {}

        for client_id in self.client_ids:
            rate = float(model_rate_list[client_id - 1])

            if rate not in rate_to_clients:
                rate_to_clients[rate] = []

            rate_to_clients[rate].append(client_id)

        all_rates = list(rate_to_clients.keys())

        # 打乱容量组顺序，避免 n_participated < 容量组数量时总是选小模型
        self.conf.random_state.shuffle(all_rates)

        selected_client_ids = []

        # 2. 每个容量组优先选 1 个
        for rate in all_rates:
            if len(selected_client_ids) >= self.conf.n_participated:
                break

            candidates = rate_to_clients[rate]

            chosen_client = self.conf.random_state.choice(
                candidates,
                size=1,
                replace=False,
            )[0]

            selected_client_ids.append(int(chosen_client))

        # 3. 如果还没选够，从剩余客户端随机补齐
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
            f"HeteroFL stratified selected "
            f"{self.conf.n_participated} from {self.conf.n_clients} clients: "
            f"{selected_client_ids}, rates={selected_rates}."
        )

        return selected_client_ids

    def _send_model_to_selected_clients(self, selected_client_ids):
        """
        给选中的客户端发送对应宽度的 HeteroFL 子模型。

        服务器端 master_model 是 full model。
        对于每个客户端：
        - 根据 client_id 找到对应 width_mult 的 client_model 模板；
        - 从 full model 中切出该客户端需要的 prefix 子模型参数；
        - 发送给对应 worker。
        """

        self.conf.logger.log("Master sends HeteroFL submodels to workers.")

        global_state = self.master_model.state_dict()

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model = self.client_models[selected_client_id]

            # 从 full global model 里切出当前客户端对应大小的子模型参数
            client_state = extract_submodel_state(
                global_state,
                client_model.state_dict(),
            )

            client_model.load_state_dict(client_state)

            flatten_model = TensorBuffer(
                list(client_model.state_dict().values())
            )

            rate = float(self.conf.model_rate_list[selected_client_id - 1])

            self.conf.logger.log(
                f"[HETEROFL SEND] "
                f"worker_rank={worker_rank}, "
                f"client_id={selected_client_id}, "
                f"rate={rate}, "
                f"numel={flatten_model.buffer.numel()}, "
                f"bytes={flatten_model.buffer.numel() * flatten_model.buffer.element_size()}."
            )

            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

            self.conf.logger.log(
                f"\tMaster sent HeteroFL rate={rate} model "
                f"to client={selected_client_id}, process_id={worker_rank}."
            )

        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        """
        接收选中客户端训练后的本地子模型。
        每个接收 buffer 的大小必须和该 client_id 的模型大小一致。
        """

        self.conf.logger.log("Master waits to receive HeteroFL local submodels.")

        dist.barrier()

        flatten_local_models = {}

        # 1. 为每个 selected client 准备对应大小的 buffer
        for selected_client_id in selected_client_ids:
            client_tb = TensorBuffer(
                list(self.client_models[selected_client_id].state_dict().values())
            )

            client_tb.buffer = torch.zeros_like(client_tb.buffer)

            flatten_local_models[selected_client_id] = client_tb

            rate = float(self.conf.model_rate_list[selected_client_id - 1])

            self.conf.logger.log(
                f"[HETEROFL RECV-BUF] "
                f"client_id={selected_client_id}, "
                f"rate={rate}, "
                f"numel={client_tb.buffer.numel()}, "
                f"bytes={client_tb.buffer.numel() * client_tb.buffer.element_size()}."
            )

        # 2. 异步接收 worker 上传的本地模型
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

        self.conf.logger.log("Master received all HeteroFL local submodels.")

        return flatten_local_models

    def aggregate(self, flatten_local_models):
        """
        聚合 HeteroFL 子模型。

        逻辑：
        - 先把每个客户端上传的 flatten buffer 解包回对应 client_model；
        - 得到 local_state_dicts；
        - 调用 aggregate_submodel_states，把不同宽度子模型聚合回 full model。
        """

        local_state_dicts = []

        for client_id, flatten_local_model in flatten_local_models.items():
            client_model = self.client_models[client_id]

            client_state = copy.deepcopy(client_model.state_dict())

            flatten_local_model.unpack(client_state.values())

            client_model.load_state_dict(client_state)

            local_state_dicts.append(client_state)

            rate = float(self.conf.model_rate_list[client_id - 1])

            self.conf.logger.log(
                f"[HETEROFL AGG] "
                f"client_id={client_id}, "
                f"rate={rate}."
            )

        aggregated_state = aggregate_submodel_states(
            self.master_model.state_dict(),
            local_state_dicts,
        )

        aggregated_model = copy.deepcopy(self.master_model)

        aggregated_model.load_state_dict(aggregated_state)

        return {0: aggregated_model}

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        """
        聚合并测试全局模型。
        """

        client_models = self.aggregate(flatten_local_models)

        self.master_model.load_state_dict(
            list(client_models.values())[0].state_dict()
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

        torch.cuda.empty_cache()