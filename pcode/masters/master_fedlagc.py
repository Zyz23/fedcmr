
# -*- coding: utf-8 -*-

import copy

import torch
import torch.distributed as dist

import pcode.master_utils as master_utils

from pcode.masters.master import Master
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.fedlagc import (
    aggregate_overlap_aware_states,
    build_layer_adaptive_masks,
    clone_state_dict_to_cpu,
    first_nonfinite_parameter_name,
    flatten_masks,
    get_parameter_names,
)


class MasterFedLAGC(Master):
    """
    FedLAGC Server.

    优化点：
    1. 每轮只构建 mask，不再为每个 client 构造稀疏完整模型。
    2. 所有 client 接收同一份完整 global model；
       不同 client 再接收自己的 binary mask。
    3. 支持 mask refresh interval：
       interval=1 时，每轮严格重算 mask；
       interval>1 时，连续多轮复用同一组 mask，减少 Top-K 开销。
    """

    def __init__(self, conf):
        super(MasterFedLAGC, self).__init__(conf)

        self.parameter_names = get_parameter_names(
            self.master_model
        )

        self.current_masks = {}
        self.current_reports = {}

        # key: capacity(float)
        # value: {"masks": ..., "report": ...}
        self.mask_cache = {}

        self.conf.logger.log(
            "FedLAGC initialized: "
            "full global model + client-specific sparse masks."
        )

    def _mask_refresh_interval(self):
        return max(
            1,
            int(
                getattr(
                    self.conf,
                    "fedlagc_mask_refresh_interval",
                    1,
                )
            ),
        )

    def _need_refresh_mask(self):
        """
        interval=1:
            每轮重新根据当前 global model 计算 mask。

        interval=5:
            第 1、6、11、16 ... 轮重新计算；
            中间轮次复用旧 mask。
        """
        interval = self._mask_refresh_interval()
        comm_round = int(self.conf.graph.comm_round)

        if len(self.mask_cache) == 0:
            return True

        return (comm_round - 1) % interval == 0

    @staticmethod
    def _capacity_key(capacity):
        return round(float(capacity), 8)

    def send_extra_info_to_selected_clients(
        self,
        selected_client_ids,
    ):
        """
        仅在服务器准备本轮每个 client 对应的 mask。
        不使用 broadcast，避免 collective 死锁。
        """
        global_state_cpu = clone_state_dict_to_cpu(
            self.master_model.state_dict()
        )

        refresh_mask = self._need_refresh_mask()

        if refresh_mask:
            self.mask_cache = {}

        self.current_masks = {}
        self.current_reports = {}

        for client_id in selected_client_ids:
            capacity = float(
                self.conf.model_rate_list[client_id - 1]
            )

            cache_key = self._capacity_key(capacity)

            if cache_key not in self.mask_cache:
                masks, _, report = build_layer_adaptive_masks(
                    state_dict=global_state_cpu,
                    parameter_names=self.parameter_names,
                    capacity=capacity,
                )

                self.mask_cache[cache_key] = {
                    "masks": masks,
                    "report": report,
                }

                reused = False
            else:
                reused = True

            cached = self.mask_cache[cache_key]

            self.current_masks[client_id] = cached["masks"]
            self.current_reports[client_id] = cached["report"]

            report = cached["report"]

            self.conf.logger.log(
                f"[FedLAGC Extract] "
                f"client={client_id}, "
                f"capacity={capacity:.4f}, "
                f"target={report['target_numel']}, "
                f"fixed={report['fixed_numel']}, "
                f"refresh={refresh_mask}, "
                f"reused={reused}, "
                f"layers={report['allocated_counts']}."
            )

    def _send_model_to_selected_clients(
        self,
        selected_client_ids,
    ):
        """
        所有 client 接收相同完整 global model。

        每个 client 额外接收自己的 binary mask。
        这样避免服务器为每一个 client 拷贝和置零 888 万参数。
        """
        global_state_cpu = clone_state_dict_to_cpu(
            self.master_model.state_dict()
        )

        full_model_buffer = TensorBuffer(
            list(global_state_cpu.values())
        ).buffer.cpu().contiguous()

        for worker_rank, client_id in enumerate(
            selected_client_ids,
            start=1,
        ):
            mask_vector = flatten_masks(
                self.current_masks[client_id],
                self.parameter_names,
            ).cpu().contiguous()

            # 1. 完整全局模型。
            dist.send(
                tensor=full_model_buffer,
                dst=worker_rank,
            )

            # 2. 当前 client 的稀疏 mask。
            dist.send(
                tensor=mask_vector,
                dst=worker_rank,
            )

            self.conf.logger.log(
                f"[FedLAGC Send] "
                f"client={client_id}, "
                f"worker={worker_rank}, "
                f"model_numel={full_model_buffer.numel()}, "
                f"mask_numel={mask_vector.numel()}."
            )

        dist.barrier()

    def _receive_models_from_selected_clients(
        self,
        selected_client_ids,
    ):
        """
        接收所有 client 上传的完整 shape 模型。

        虽然模型是完整 shape，但服务器仅根据该 client 本轮 mask
        聚合其真正参与训练的位置。
        """
        self.conf.logger.log(
            "Master waits for FedLAGC local updates."
        )

        dist.barrier()

        master_state_cpu = clone_state_dict_to_cpu(
            self.master_model.state_dict()
        )

        flatten_local_models = {}

        for client_id in selected_client_ids:
            template_state = copy.deepcopy(
                master_state_cpu
            )

            tensor_buffer = TensorBuffer(
                list(template_state.values())
            )

            tensor_buffer.buffer = torch.zeros_like(
                tensor_buffer.buffer,
                device="cpu",
            )

            flatten_local_models[client_id] = tensor_buffer

        requests = []

        for client_id, worker_rank in zip(
            selected_client_ids,
            self.world_ids,
        ):
            request = dist.irecv(
                tensor=flatten_local_models[
                    client_id
                ].buffer,
                src=worker_rank,
            )

            requests.append(request)

        for request in requests:
            request.wait()

        dist.barrier()

        self.conf.logger.log(
            "Master received all FedLAGC local updates."
        )

        return flatten_local_models

    def aggregate(self, flatten_local_models):
        """
        对不同 client 的稀疏坐标做 overlap-aware aggregation。

        某个参数位置只有被选中的 client 才参与平均；
        没有被任何 client 更新的位置保留原 global value。
        """
        global_state_cpu = clone_state_dict_to_cpu(
            self.master_model.state_dict()
        )

        local_states = []
        local_masks = []

        for client_id, tensor_buffer in (
            flatten_local_models.items()
        ):
            local_state = copy.deepcopy(
                global_state_cpu
            )

            tensor_buffer.unpack(
                local_state.values()
            )

            local_states.append(local_state)
            local_masks.append(
                self.current_masks[client_id]
            )

        aggregated_state = aggregate_overlap_aware_states(
            global_state_dict=global_state_cpu,
            local_state_dicts=local_states,
            local_masks=local_masks,
            parameter_names=self.parameter_names,
        )

        candidate_model = copy.deepcopy(
            self.master_model
        )

        candidate_model.load_state_dict(
            aggregated_state,
            strict=True,
        )

        bad_parameter = first_nonfinite_parameter_name(
            candidate_model
        )

        if bad_parameter is not None:
            raise RuntimeError(
                f"FedLAGC aggregated model has NaN/Inf at "
                f"'{bad_parameter}'."
            )

        return {0: candidate_model}

    def _aggregate_model_and_evaluate(
        self,
        flatten_local_models,
        selected_client_ids,
    ):
        client_models = self.aggregate(
            flatten_local_models
        )

        aggregated_model = list(
            client_models.values()
        )[0]

        self.master_model.load_state_dict(
            aggregated_model.state_dict(),
            strict=True,
        )

        # 最终仍然在服务器完整全局模型上统一测试。
        master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label="fedlagc_global_test",
        )

        if self.conf.graph.on_cuda:
            torch.cuda.empty_cache()

