# -*- coding: utf-8 -*-

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
import pcode.create_model as create_model

from pcode.workers.worker import Worker
from pcode.utils.logging import display_training_stat
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.feddse import (
    NEURON_LAYERS,
    build_cnn_index_map,
    clone_state_dict_to_cpu,
    extract_indexed_submodel_state,
    first_nonfinite_state_key,
    get_neuron_counts,
)


class WorkerFedDSE(Worker):
    """
    FedDSE Client.

    每轮流程：
    1. 接收服务器完整模型；
    2. 用少量本地数据统计每层神经元激活；
    3. 根据激活值加权随机采样神经元；
    4. 从完整模型抽取一致的紧凑子模型；
    5. 在紧凑子模型上本地训练；
    6. 上传神经元索引与训练后子模型。
    """

    def extra_init(self):
        """
        base Worker 已创建 self.model：
        它是客户端容量对应的小模型。

        这里额外创建 self.full_model：
        它用于接收服务器完整模型和做激活统计。
        """
        _, self.full_model = create_model.define_model(
            self.conf,
            show_stat=False,
            to_consistent_model=False,
            client_id=0,
        )

        if self.conf.graph.on_cuda:
            self.full_model = self.full_model.to(
                self.device
            )

        self.full_model_state_dict = (
            self.full_model.state_dict()
        )

        self.full_model_tb = TensorBuffer(
            list(self.full_model_state_dict.values())
        )

        self.selected_neurons = None
        self.current_index_map = None

    def recv_extra_info_from_master(self):
        """
        FedDSE 不需要额外服务器广播。
        """
        return

    def _profile_activation_scores(self):
        """
        通过完整模型在本地少量数据上推理，
        计算 conv1~conv4、fc1 每个神经元的平均 ReLU 激活。

        0 表示使用所有本地 batch；
        正数表示最多使用前 profile_batches 个 batch。
        """
        model = self.full_model

        was_training = model.training
        model.eval()

        profile_batches = int(
            getattr(
                self.conf,
                "feddse_profile_batches",
                10,
            )
        )

        layer_sizes = get_neuron_counts(model)

        activation_sums = {
            layer_name: torch.zeros(
                layer_sizes[layer_name],
                dtype=torch.float64,
                device=self.device,
            )
            for layer_name in NEURON_LAYERS
        }

        total_samples = 0

        with torch.no_grad():
            for batch_index, (_input, _target) in enumerate(
                self.train_loader
            ):
                if (
                    profile_batches > 0
                    and batch_index >= profile_batches
                ):
                    break

                data_batch = create_dataset.load_data_batch(
                    self.conf,
                    _input,
                    _target,
                    is_training=False,
                    device=self.device,
                )

                x = data_batch["input"]
                batch_size = x.size(0)

                x = F.relu(
                    model.scaler(model.conv1(x))
                )

                activation_sums["conv1"] += (
                    x.mean(dim=(0, 2, 3))
                    .double()
                    * batch_size
                )

                x = model.pool(x)

                x = F.relu(
                    model.scaler(model.conv2(x))
                )

                activation_sums["conv2"] += (
                    x.mean(dim=(0, 2, 3))
                    .double()
                    * batch_size
                )

                x = model.pool(x)

                x = F.relu(
                    model.scaler(model.conv3(x))
                )

                activation_sums["conv3"] += (
                    x.mean(dim=(0, 2, 3))
                    .double()
                    * batch_size
                )

                x = model.pool(x)

                x = F.relu(
                    model.scaler(model.conv4(x))
                )

                activation_sums["conv4"] += (
                    x.mean(dim=(0, 2, 3))
                    .double()
                    * batch_size
                )

                x = x.flatten(1)

                feature = F.relu(
                    model.scaler(model.fc1(x))
                )

                activation_sums["fc1"] += (
                    feature.mean(dim=0)
                    .double()
                    * batch_size
                )

                total_samples += batch_size

        if was_training:
            model.train()

        if total_samples == 0:
            return {
                layer_name: torch.ones(
                    layer_sizes[layer_name],
                    dtype=torch.float32,
                )
                for layer_name in NEURON_LAYERS
            }

        return {
            layer_name: (
                activation_sums[layer_name]
                / float(total_samples)
            )
            .float()
            .cpu()
            for layer_name in NEURON_LAYERS
        }

    def _sample_neurons(
        self,
        activation_scores,
        num_selected,
        layer_index,
    ):
        """
        根据论文的 weighted sampling：

        p(i) = softmax(h_i / T)

        T 越小越接近 Top-K；
        T 越大越接近均匀随机抽样。
        """
        activation_scores = (
            activation_scores.detach()
            .float()
            .cpu()
        )

        total_neurons = activation_scores.numel()

        if num_selected >= total_neurons:
            return torch.arange(
                total_neurons,
                dtype=torch.long,
            )

        temperature = float(
            getattr(
                self.conf,
                "feddse_temperature",
                1.0,
            )
        )

        if temperature <= 1e-8:
            return torch.topk(
                activation_scores,
                k=num_selected,
                largest=True,
                sorted=True,
            ).indices.sort().values.long()

        activation_scores = torch.nan_to_num(
            activation_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        probs = torch.softmax(
            activation_scores / temperature,
            dim=0,
        )

        if (
            not torch.isfinite(probs).all().item()
            or probs.sum().item() <= 0
        ):
            probs = torch.ones_like(probs)
            probs = probs / probs.sum()

        seed = (
            int(self.conf.manual_seed) * 1000003
            + int(self.conf.graph.comm_round) * 9176
            + int(self.conf.graph.client_id) * 131
            + int(layer_index + 1) * 17
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        selected = torch.multinomial(
            probs,
            num_samples=num_selected,
            replacement=False,
            generator=generator,
        )

        return selected.sort().values.long()

    def _select_neurons_from_activations(self):
        activation_scores = self._profile_activation_scores()

        local_counts = get_neuron_counts(self.model)

        selected_neurons = {}

        for layer_index, layer_name in enumerate(
            NEURON_LAYERS
        ):
            selected_neurons[layer_name] = (
                self._sample_neurons(
                    activation_scores[
                        layer_name
                    ],
                    local_counts[layer_name],
                    layer_index,
                )
            )

        return selected_neurons

    def _recv_model_from_master(self):
        """
        接收完整 global model，完成激活驱动的子模型抽取。
        """
        recv_buffer = torch.zeros_like(
            self.full_model_tb.buffer,
            device="cpu",
        )

        dist.recv(
            tensor=recv_buffer,
            src=0,
        )

        self.full_model_tb.buffer.copy_(
            recv_buffer.to(
                self.full_model_tb.buffer.device
            )
        )

        self.full_model_tb.unpack(
            self.full_model_state_dict.values()
        )

        self.full_model.load_state_dict(
            self.full_model_state_dict,
            strict=True,
        )

        self.selected_neurons = (
            self._select_neurons_from_activations()
        )

        self.current_index_map = build_cnn_index_map(
            global_state_dict=self.full_model.state_dict(),
            local_state_dict=self.model.state_dict(),
            selections=self.selected_neurons,
        )

        extracted_local_state = (
            extract_indexed_submodel_state(
                global_state_dict=self.full_model.state_dict(),
                local_state_dict=self.model.state_dict(),
                index_map=self.current_index_map,
            )
        )

        self.model.load_state_dict(
            extracted_local_state,
            strict=True,
        )

        self.model_state_dict = self.model.state_dict()

        self.model_tb = TensorBuffer(
            list(self.model_state_dict.values())
        )

        self.conf.logger.log(
            f"[FedDSE] Worker="
            f"{self.conf.graph.worker_id}, "
            f"client={self.conf.graph.client_id}, "
            f"selected="
            f"{ {k: int(v.numel()) for k, v in self.selected_neurons.items()} }."
        )

        dist.barrier()

    def _restore_round_start_model(
        self,
        round_start_state,
    ):
        self.model.load_state_dict(
            round_start_state,
            strict=True,
        )

        self.optimizer.zero_grad(set_to_none=True)

    def _first_nonfinite_gradient(self):
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(
                    parameter.grad
                ).all().item():
                    return name

        return None

    def _train(self):
        """
        与普通 Worker 训练逻辑一致，
        仅添加 NaN/Inf 防护与每个 epoch 一次日志。
        """
        self.prepare_train()

        round_start_state = clone_state_dict_to_cpu(
            self.model.state_dict()
        )

        while True:
            for _input, _target in self.train_loader:
                with self.timer(
                    "load_data",
                    epoch=self.scheduler.epoch_,
                ):
                    data_batch = create_dataset.load_data_batch(
                        self.conf,
                        _input,
                        _target,
                        is_training=True,
                        device=self.device,
                    )

                with self.timer(
                    "forward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    (
                        loss,
                        performance,
                        feature,
                        output,
                    ) = self._inference(data_batch)

                with self.timer(
                    "extra_forward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    loss = self.local_training_with_extra_calculate(
                        loss,
                        output,
                        data_batch,
                        feature,
                    )

                if not torch.isfinite(loss).all().item():
                    self.conf.logger.log(
                        f"[FedDSE Abort] client="
                        f"{self.conf.graph.client_id}, "
                        f"round={self.conf.graph.comm_round}, "
                        f"reason=nonfinite_loss."
                    )

                    self._restore_round_start_model(
                        round_start_state
                    )

                    self._terminate_comm_round()
                    return

                with self.timer(
                    "backward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                    loss.backward()

                    bad_gradient = (
                        self._first_nonfinite_gradient()
                    )

                    if bad_gradient is not None:
                        self.conf.logger.log(
                            f"[FedDSE Abort] client="
                            f"{self.conf.graph.client_id}, "
                            f"round={self.conf.graph.comm_round}, "
                            f"gradient={bad_gradient}."
                        )

                        self._restore_round_start_model(
                            round_start_state
                        )

                        self._terminate_comm_round()
                        return

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=10.0,
                    )

                    if not torch.isfinite(
                        grad_norm
                    ).all().item():
                        self.conf.logger.log(
                            f"[FedDSE Abort] client="
                            f"{self.conf.graph.client_id}, "
                            f"round={self.conf.graph.comm_round}, "
                            f"reason=nonfinite_grad_norm."
                        )

                        self._restore_round_start_model(
                            round_start_state
                        )

                        self._terminate_comm_round()
                        return

                    self.optimizer.step()

                    bad_key = first_nonfinite_state_key(
                        self.model.state_dict()
                    )

                    if bad_key is not None:
                        self.conf.logger.log(
                            f"[FedDSE Abort] client="
                            f"{self.conf.graph.client_id}, "
                            f"round={self.conf.graph.comm_round}, "
                            f"parameter={bad_key}."
                        )

                        self._restore_round_start_model(
                            round_start_state
                        )

                        self._terminate_comm_round()
                        return

                    self.scheduler.step()

                if self._is_finished_one_comm_round():
                    timer_summary = self.timer.summary()

                    if (
                        isinstance(timer_summary, str)
                        and len(timer_summary) > 0
                    ):
                        self.conf.logger.log(
                            timer_summary
                        )

                    self._terminate_comm_round()
                    return

                batches_per_epoch = max(
                    1,
                    int(
                        getattr(
                            self.conf,
                            "num_batches_per_device_per_epoch",
                            1,
                        )
                    ),
                )

                if (
                    self.scheduler.local_index
                    % batches_per_epoch
                    == 0
                ):
                    display_training_stat(
                        self.conf,
                        self.scheduler,
                        self.tracker,
                    )

                    self.tracker.reset()

                if self.conf.logger.meet_cache_limit():
                    self.conf.logger.save_json()

    def send_extra_info_to_master(self):
        """
        先上传五组神经元索引。

        Master 收到这些索引后，才知道客户端子模型
        对应 global model 的哪些通道。
        """
        if self.selected_neurons is None:
            raise RuntimeError(
                "FedDSE neuron selection is missing."
            )

        for layer_name in NEURON_LAYERS:
            index_tensor = (
                self.selected_neurons[layer_name]
                .long()
                .cpu()
                .contiguous()
            )

            dist.send(
                tensor=index_tensor,
                dst=0,
            )

        dist.barrier()

    def _terminate_comm_round(self):
        """
        释放用于 activation profiling 的 full model 显存。
        """
        if hasattr(self, "full_model"):
            self.full_model = self.full_model.cpu()

        super(WorkerFedDSE, self)._terminate_comm_round()