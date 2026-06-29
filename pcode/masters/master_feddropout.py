# -*- coding: utf-8 -*-

import copy

import torch
import torch.distributed as dist

from pcode.masters.master_heterofl import MasterHeteroFL
from pcode.utils.heterofl import aggregate_rolling_submodel_states
from pcode.utils.tensor_buffer import TensorBuffer


def _stable_tag_to_int(tag):
    """
    避免使用 Python hash，因为 hash 在不同进程/运行中可能不稳定。
    """
    value = 0
    for idx, ch in enumerate(str(tag)):
        value += (idx + 1) * ord(ch)
    return value


def _random_indices(global_size, local_size, round_idx, client_id, tag):
    """
    为某一层生成稳定随机通道索引。
    同一 round/client/tag 下索引固定，便于发送和聚合对应。
    """
    if local_size == global_size:
        return torch.arange(global_size, dtype=torch.long)

    seed = (
        int(max(1, round_idx)) * 1000003
        + int(max(1, client_id)) * 9176
        + int(global_size) * 131
        + int(local_size) * 17
        + _stable_tag_to_int(tag) * 97
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.randperm(global_size, generator=generator)[:local_size].sort().values


def _arange(size):
    return torch.arange(size, dtype=torch.long)


def _flatten_channel_indices(channel_indices, spatial_size):
    """
    将卷积通道索引映射到 flatten 后的 fc1 输入索引。

    例如 conv4 输出 shape 是 [B, C, 4, 4]，
    flatten 后每个通道对应连续 16 个位置。
    """
    spatial_indices = torch.arange(spatial_size, dtype=torch.long)
    return (
        channel_indices[:, None] * spatial_size
        + spatial_indices[None, :]
    ).reshape(-1)


def _take_with_indices(value, indices):
    """
    根据 index tuple 从 full tensor 中抽取子 tensor。
    """
    result = value
    for dim, index in enumerate(indices):
        result = result.index_select(dim, index.to(result.device))
    return result


def _mesh_indices(indices, device):
    tensors = [index.to(device) for index in indices]
    if len(tensors) == 0:
        return ()
    return torch.meshgrid(*tensors, indexing="ij")


def _fallback_random_indices_for_param(key, global_value, local_value, round_idx, client_id):
    """
    对未知参数名做兜底处理。
    已知 CNN 层会走显式通道一致逻辑；
    其他层若 shape 不一致，则按维度随机抽取。
    """
    indices = []

    for dim, (global_size, local_size) in enumerate(
        zip(tuple(global_value.shape), tuple(local_value.shape))
    ):
        if global_size == local_size:
            indices.append(_arange(global_size))
        else:
            indices.append(
                _random_indices(
                    global_size,
                    local_size,
                    round_idx,
                    client_id,
                    tag=f"{key}_dim{dim}",
                )
            )

    return tuple(indices)


def build_consistent_feddropout_index_map(
    global_state_dict,
    sub_state_dict,
    round_idx,
    client_id,
):
    """
    为 HeteroCNNCifar 构造层间一致的 FedDropout 随机子网络索引。

    关键约束：
    conv1.out = conv2.in
    conv2.out = conv3.in
    conv3.out = conv4.in
    conv4.out = fc1.input channels
    fc1.out   = classifier.input
    """

    index_map = {}

    # 如果当前模型不是 HeteroCNNCifar 的命名结构，则走兜底随机逻辑。
    required_keys = [
        "conv1.weight",
        "conv2.weight",
        "conv3.weight",
        "conv4.weight",
        "fc1.weight",
        "classifier.weight",
    ]

    is_known_cnn = all(
        key in global_state_dict and key in sub_state_dict
        for key in required_keys
    )

    if not is_known_cnn:
        for key, local_value in sub_state_dict.items():
            global_value = global_state_dict[key]
            index_map[key] = _fallback_random_indices_for_param(
                key,
                global_value,
                local_value,
                round_idx,
                client_id,
            )
        return index_map

    # ---------- 1. 为每一层生成一致的通道索引 ----------
    # conv1: [c1, 3, k, k]
    c1_global = global_state_dict["conv1.weight"].shape[0]
    c1_local = sub_state_dict["conv1.weight"].shape[0]

    # conv2: [c2, c1, k, k]
    c2_global = global_state_dict["conv2.weight"].shape[0]
    c2_local = sub_state_dict["conv2.weight"].shape[0]

    # conv3: [c3, c2, k, k]
    c3_global = global_state_dict["conv3.weight"].shape[0]
    c3_local = sub_state_dict["conv3.weight"].shape[0]

    # conv4: [c4, c3, k, k]
    c4_global = global_state_dict["conv4.weight"].shape[0]
    c4_local = sub_state_dict["conv4.weight"].shape[0]

    # fc1: [h1, c4 * 4 * 4]
    h1_global = global_state_dict["fc1.weight"].shape[0]
    h1_local = sub_state_dict["fc1.weight"].shape[0]

    idx_c1 = _random_indices(c1_global, c1_local, round_idx, client_id, "conv1_out")
    idx_c2 = _random_indices(c2_global, c2_local, round_idx, client_id, "conv2_out")
    idx_c3 = _random_indices(c3_global, c3_local, round_idx, client_id, "conv3_out")
    idx_c4 = _random_indices(c4_global, c4_local, round_idx, client_id, "conv4_out")
    idx_h1 = _random_indices(h1_global, h1_local, round_idx, client_id, "fc1_out")

    # 输入图片 RGB 通道不 dropout
    idx_img = _arange(global_state_dict["conv1.weight"].shape[1])

    # 卷积核空间维度不 dropout
    idx_k1_h = _arange(global_state_dict["conv1.weight"].shape[2])
    idx_k1_w = _arange(global_state_dict["conv1.weight"].shape[3])
    idx_k2_h = _arange(global_state_dict["conv2.weight"].shape[2])
    idx_k2_w = _arange(global_state_dict["conv2.weight"].shape[3])
    idx_k3_h = _arange(global_state_dict["conv3.weight"].shape[2])
    idx_k3_w = _arange(global_state_dict["conv3.weight"].shape[3])
    idx_k4_h = _arange(global_state_dict["conv4.weight"].shape[2])
    idx_k4_w = _arange(global_state_dict["conv4.weight"].shape[3])

    # fc1 输入 flatten 维度：由 conv4 输出通道映射而来
    global_fc1_in = global_state_dict["fc1.weight"].shape[1]
    global_c4 = global_state_dict["conv4.weight"].shape[0]

    if global_fc1_in % global_c4 == 0:
        spatial_size = global_fc1_in // global_c4
    else:
        # 正常 HeteroCNNCifar 应该是 4*4=16。
        # 如果结构变化，则兜底逐维随机。
        spatial_size = None

    if spatial_size is not None:
        idx_fc1_in = _flatten_channel_indices(idx_c4, spatial_size)
    else:
        idx_fc1_in = _random_indices(
            global_fc1_in,
            sub_state_dict["fc1.weight"].shape[1],
            round_idx,
            client_id,
            "fc1_input_fallback",
        )

    # classifier 输出类别不 dropout
    idx_class = _arange(global_state_dict["classifier.weight"].shape[0])

    # ---------- 2. 显式设置每个已知参数的索引 ----------
    index_map["conv1.weight"] = (idx_c1, idx_img, idx_k1_h, idx_k1_w)
    if "conv1.bias" in sub_state_dict:
        index_map["conv1.bias"] = (idx_c1,)

    index_map["conv2.weight"] = (idx_c2, idx_c1, idx_k2_h, idx_k2_w)
    if "conv2.bias" in sub_state_dict:
        index_map["conv2.bias"] = (idx_c2,)

    index_map["conv3.weight"] = (idx_c3, idx_c2, idx_k3_h, idx_k3_w)
    if "conv3.bias" in sub_state_dict:
        index_map["conv3.bias"] = (idx_c3,)

    index_map["conv4.weight"] = (idx_c4, idx_c3, idx_k4_h, idx_k4_w)
    if "conv4.bias" in sub_state_dict:
        index_map["conv4.bias"] = (idx_c4,)

    index_map["fc1.weight"] = (idx_h1, idx_fc1_in)
    if "fc1.bias" in sub_state_dict:
        index_map["fc1.bias"] = (idx_h1,)

    index_map["classifier.weight"] = (idx_class, idx_h1)
    if "classifier.bias" in sub_state_dict:
        index_map["classifier.bias"] = (idx_class,)

    # ---------- 3. 对剩余参数做兜底处理 ----------
    for key, local_value in sub_state_dict.items():
        if key in index_map:
            continue

        global_value = global_state_dict[key]

        if tuple(global_value.shape) == tuple(local_value.shape):
            index_map[key] = tuple(_arange(size) for size in global_value.shape)
        else:
            index_map[key] = _fallback_random_indices_for_param(
                key,
                global_value,
                local_value,
                round_idx,
                client_id,
            )

    return index_map


def extract_consistent_random_submodel_state(
    global_state_dict,
    sub_state_dict,
    round_idx,
    client_id,
):
    """
    使用层间一致的随机索引，从 full model 中抽取 FedDropout 子模型。
    """
    extracted = copy.deepcopy(sub_state_dict)

    index_map = build_consistent_feddropout_index_map(
        global_state_dict,
        sub_state_dict,
        round_idx,
        client_id,
    )

    for key, sub_value in extracted.items():
        global_value = global_state_dict[key]
        indices = index_map[key]

        selected = _take_with_indices(global_value, indices)

        if tuple(selected.shape) != tuple(sub_value.shape):
            raise RuntimeError(
                f"FedDropout extraction shape mismatch for {key}: "
                f"selected={tuple(selected.shape)}, "
                f"expected={tuple(sub_value.shape)}."
            )

        sub_value.copy_(selected.to(sub_value.device, dtype=sub_value.dtype))

    return extracted, index_map


def aggregate_consistent_random_submodel_states(
    global_state_dict,
    local_state_dicts,
    index_maps,
):
    """
    根据 index_map 将 FedDropout 子模型聚合回 full model。

    这里逻辑和 rolling 聚合一致：
    - 被客户端更新过的位置取平均；
    - 没有被更新的位置保留旧全局参数。
    """
    aggregated = copy.deepcopy(global_state_dict)

    for key, global_value in global_state_dict.items():
        if not global_value.dtype.is_floating_point:
            full_values = [
                local_state[key].to(global_value.device)
                for local_state in local_state_dicts
                if key in local_state and tuple(local_state[key].shape) == tuple(global_value.shape)
            ]

            if full_values:
                stacked = torch.stack([value.float() for value in full_values], dim=0)
                aggregated[key] = torch.round(stacked.mean(dim=0)).to(global_value.dtype)

            continue

        sum_value = torch.zeros_like(global_value, dtype=torch.float32)
        count_value = torch.zeros_like(global_value, dtype=torch.float32)

        for local_state, index_map in zip(local_state_dicts, index_maps):
            local_value = local_state[key].to(global_value.device).float()

            if key not in index_map:
                raise RuntimeError(f"Missing index_map for parameter: {key}")

            mesh = _mesh_indices(index_map[key], global_value.device)

            sum_value.index_put_(mesh, local_value, accumulate=True)
            count_value.index_put_(mesh, torch.ones_like(local_value), accumulate=True)

        updated = count_value > 0

        new_value = global_value.detach().clone().float()
        new_value[updated] = sum_value[updated] / count_value[updated]

        aggregated[key] = new_value.to(global_value.dtype)

    return aggregated


class MasterFedDropout(MasterHeteroFL):
    """
    Federated Dropout baseline with consistent random submodel extraction.

    修改点：
    1. 随机子模型保持 CNN 层间通道一致；
    2. 使用容量分层采样；
    3. 增加发送、接收、聚合日志；
    4. 不引入 FedOur / CLIP / anchor / distillation。
    """

    def _random_select_clients(self):
        """
        FedDropout 分层采样。

        每轮尽量从每个 model_rate 组里选一个客户端。
        对 n_clients=20, participation_ratio=0.2, rates=4 档时，
        每轮尽量选到 0.25、0.5、0.75、1.0 各一个。
        """
        model_rate_list = self.conf.model_rate_list

        rate_to_clients = {}

        for client_id in self.client_ids:
            rate = float(model_rate_list[client_id - 1])

            if rate not in rate_to_clients:
                rate_to_clients[rate] = []

            rate_to_clients[rate].append(client_id)

        rates = list(rate_to_clients.keys())
        self.conf.random_state.shuffle(rates)

        selected_client_ids = []

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
            f"FedDropout stratified selected "
            f"{self.conf.n_participated} from {self.conf.n_clients} clients: "
            f"{selected_client_ids}, rates={selected_rates}."
        )

        return selected_client_ids

    def _send_model_to_selected_clients(self, selected_client_ids):
        """
        发送层间通道一致的 FedDropout random submodel。
        """
        self.conf.logger.log("Master sends FedDropout consistent random submodels to workers.")

        global_state = self.master_model.state_dict()

        self.current_index_maps = {}

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model = self.client_models[selected_client_id]
            client_template_state = client_model.state_dict()

            client_state, index_map = extract_consistent_random_submodel_state(
                global_state,
                client_template_state,
                self.conf.graph.comm_round,
                selected_client_id,
            )

            self.current_index_maps[selected_client_id] = index_map

            client_model.load_state_dict(client_state)

            flatten_model = TensorBuffer(
                list(client_model.state_dict().values())
            )

            rate = float(self.conf.model_rate_list[selected_client_id - 1])

            self.conf.logger.log(
                f"[FEDDROPOUT SEND] "
                f"round={self.conf.graph.comm_round}, "
                f"worker_rank={worker_rank}, "
                f"client_id={selected_client_id}, "
                f"rate={rate}, "
                f"numel={flatten_model.buffer.numel()}, "
                f"bytes={flatten_model.buffer.numel() * flatten_model.buffer.element_size()}."
            )

            dist.send(tensor=flatten_model.buffer, dst=worker_rank)

            self.conf.logger.log(
                f"\tMaster sent FedDropout rate={rate} random model "
                f"to client={selected_client_id}, process_id={worker_rank}."
            )

        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        """
        接收客户端训练后的 FedDropout 子模型。
        """
        self.conf.logger.log("Master waits to receive FedDropout local random submodels.")

        dist.barrier()

        flatten_local_models = {}

        for selected_client_id in selected_client_ids:
            client_tb = TensorBuffer(
                list(self.client_models[selected_client_id].state_dict().values())
            )

            client_tb.buffer = torch.zeros_like(client_tb.buffer)

            flatten_local_models[selected_client_id] = client_tb

            rate = float(self.conf.model_rate_list[selected_client_id - 1])

            self.conf.logger.log(
                f"[FEDDROPOUT RECV-BUF] "
                f"client_id={selected_client_id}, "
                f"rate={rate}, "
                f"numel={client_tb.buffer.numel()}, "
                f"bytes={client_tb.buffer.numel() * client_tb.buffer.element_size()}."
            )

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

        self.conf.logger.log("Master received all FedDropout local random submodels.")

        return flatten_local_models

    def aggregate(self, flatten_local_models):
        """
        将 FedDropout 子模型按 index_map 聚合回完整全局模型。
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
                    f"Missing FedDropout index_map for client_id={client_id}. "
                    f"Please check whether _send_model_to_selected_clients() "
                    f"was called before aggregate()."
                )

            index_maps.append(self.current_index_maps[client_id])

            rate = float(self.conf.model_rate_list[client_id - 1])

            self.conf.logger.log(
                f"[FEDDROPOUT AGG] "
                f"client_id={client_id}, "
                f"rate={rate}."
            )

        aggregated_state = aggregate_consistent_random_submodel_states(
            self.master_model.state_dict(),
            local_state_dicts,
            index_maps,
        )

        aggregated_model = copy.deepcopy(self.master_model)
        aggregated_model.load_state_dict(aggregated_state)

        return {0: aggregated_model}