# -*- coding: utf-8 -*-

import math
import torch
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
from pcode.utils.logging import display_training_stat
from pcode.workers.worker import Worker


class WorkerFjORD(Worker):
    """
    FjORD client with Ordered Dropout and finite-value protection.

    核心策略：
    1. 每个客户端最多训练其 p_max 对应的子模型；
    2. 每个 batch 随机采样 p <= p_max 的 Ordered Dropout 子模型；
    3. p_max 模型承担 teacher CE 训练；
    4. 小子模型通过 KL 学习 teacher 的软输出；
    5. 任意 loss / gradient / parameter 出现 NaN 或 Inf 时：
       恢复本轮下发的初始模型，再上传，避免污染服务器。
    """

    def extra_init(self):
        self.fjord_rates = self._available_fjord_rates()

        self.fjord_max_rate = float(
            self.conf.model_rate_list[self.conf.graph.client_id - 1]
        )

        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} "
            f"(client-{self.conf.graph.client_id}) uses FjORD "
            f"p_max={self.fjord_max_rate}, "
            f"sampled rates={self.fjord_rates}."
        )

    @staticmethod
    def _tensor_is_finite(tensor):
        return tensor is not None and torch.isfinite(tensor).all().item()

    def _model_is_finite(self):
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                if not torch.isfinite(value).all().item():
                    return False, name

        return True, None

    def _first_nonfinite_gradient(self):
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all().item():
                    return name

        return None

    def _snapshot_round_start_model(self):
        """
        保存服务器刚下发、且已经成功加载的本轮初始模型。
        异常时恢复它，保证上传结果一定是有限的。
        """
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _abort_round_and_restore(
        self,
        round_start_state,
        stage,
        sampled_rate,
        parameter_name=None,
    ):
        message = (
            f"[FjORD ABORT] worker={self.conf.graph.worker_id}, "
            f"client={self.conf.graph.client_id}, "
            f"round={self.conf.graph.comm_round}, "
            f"p_max={self.fjord_max_rate}, "
            f"sampled_rate={sampled_rate}, "
            f"stage={stage}"
        )

        if parameter_name is not None:
            message += f", parameter={parameter_name}"

        message += ". Restore the round-start model before upload."

        self.conf.logger.log(message)

        self.model.load_state_dict(
            round_start_state,
            strict=True,
        )

        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad()

        # 与普通 Worker 的正常结束方式一致。
        # 后续 run() 会执行 _send_model_to_master()，
        # 因此上传的是恢复后的有限模型。
        self._terminate_comm_round()

    def _available_fjord_rates(self):
        p_max = float(
            self.conf.model_rate_list[self.conf.graph.client_id - 1]
        )

        rates_text = getattr(self.conf, "fjord_sample_rates", "")

        if rates_text is None or len(str(rates_text).strip()) == 0:
            rates_text = self.conf.hetero_model_rates

        rates = sorted(
            {
                float(rate.strip())
                for rate in str(rates_text).split(",")
                if len(rate.strip()) > 0
                and float(rate.strip()) <= p_max + 1e-12
            }
        )

        if p_max not in rates:
            rates.append(p_max)

        rates.sort()
        return rates

    def _sample_fjord_rate(self):
        return float(self.conf.random_state.choice(self.fjord_rates))

    def _active_channels(self, size, sampled_rate):
        ratio = min(
            1.0,
            float(sampled_rate) / max(self.fjord_max_rate, 1e-12),
        )

        return max(1, min(size, int(math.ceil(size * ratio))))

    def _scale(self, x, sampled_rate):
        if self.model.training and getattr(
            self.conf,
            "fjord_use_scaler",
            False,
        ):
            return x / sampled_rate

        return x

    def _fjord_forward_cnn(self, inputs, sampled_rate):
        """
        Ordered Dropout 前缀子网络前向传播。

        保持通道一致：
        conv1 输出 -> conv2 输入
        conv2 输出 -> conv3 输入
        conv3 输出 -> conv4 输入
        conv4 输出 -> fc1 输入
        fc1 输出 -> classifier 输入
        """
        model = self.model

        required = [
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "fc1",
            "classifier",
        ]

        if not all(hasattr(model, name) for name in required):
            raise NotImplementedError(
                "FjORD Ordered Dropout is implemented for "
                "federated_averaging_cnn/HeteroCNNCifar. "
                "Use --arch federated_averaging_cnn."
            )

        c1 = self._active_channels(
            model.conv1.out_channels,
            sampled_rate,
        )
        c2 = self._active_channels(
            model.conv2.out_channels,
            sampled_rate,
        )
        c3 = self._active_channels(
            model.conv3.out_channels,
            sampled_rate,
        )
        c4 = self._active_channels(
            model.conv4.out_channels,
            sampled_rate,
        )
        h1 = self._active_channels(
            model.fc1.out_features,
            sampled_rate,
        )

        x = F.conv2d(
            inputs,
            model.conv1.weight[:c1],
            None if model.conv1.bias is None else model.conv1.bias[:c1],
            stride=model.conv1.stride,
            padding=model.conv1.padding,
        )
        x = model.pool(F.relu(self._scale(x, sampled_rate)))

        x = F.conv2d(
            x,
            model.conv2.weight[:c2, :c1],
            None if model.conv2.bias is None else model.conv2.bias[:c2],
            stride=model.conv2.stride,
            padding=model.conv2.padding,
        )
        x = model.pool(F.relu(self._scale(x, sampled_rate)))

        x = F.conv2d(
            x,
            model.conv3.weight[:c3, :c2],
            None if model.conv3.bias is None else model.conv3.bias[:c3],
            stride=model.conv3.stride,
            padding=model.conv3.padding,
        )
        x = model.pool(F.relu(self._scale(x, sampled_rate)))

        x = F.conv2d(
            x,
            model.conv4.weight[:c4, :c3],
            None if model.conv4.bias is None else model.conv4.bias[:c4],
            stride=model.conv4.stride,
            padding=model.conv4.padding,
        )
        x = F.relu(self._scale(x, sampled_rate))

        x = x.reshape(x.size(0), -1)

        active_in_features = c4 * 4 * 4

        feature = F.linear(
            x[:, :active_in_features],
            model.fc1.weight[:h1, :active_in_features],
            None if model.fc1.bias is None else model.fc1.bias[:h1],
        )
        feature = F.relu(self._scale(feature, sampled_rate))

        logits = F.linear(
            feature,
            model.classifier.weight[:, :h1],
            model.classifier.bias,
        )

        return feature, logits

    def _train(self):
        self.prepare_train()

        # 本轮开始前保存服务器下发模型，用于 NaN 时恢复。
        round_start_state = self._snapshot_round_start_model()

        use_kd = bool(getattr(self.conf, "fjord_kd", True))
        kd_weight = float(getattr(self.conf, "fjord_kd_weight", 1.0))
        temperature = float(getattr(self.conf, "fjord_temperature", 1.0))

        if temperature <= 0:
            temperature = 1.0
            self.conf.logger.log(
                "[FjORD] Invalid fjord_temperature detected. "
                "Fallback to temperature=1.0."
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

                sampled_rate = self._sample_fjord_rate()
                target = data_batch["target"]

                with self.timer(
                    "forward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    # 只有 student 比 p_max 小时，才计算 teacher 和 KD。
                    # 避免 p=p_max 时重复进行两次完全相同的 forward。
                    if use_kd and sampled_rate < self.fjord_max_rate:
                        _, sampled_logits = self._fjord_forward_cnn(
                            data_batch["input"],
                            sampled_rate,
                        )

                        if not self._tensor_is_finite(sampled_logits):
                            self._abort_round_and_restore(
                                round_start_state,
                                stage="sampled_logits",
                                sampled_rate=sampled_rate,
                            )
                            return

                        sampled_loss = self.criterion(
                            sampled_logits,
                            target,
                        )

                        if not self._tensor_is_finite(sampled_loss):
                            self._abort_round_and_restore(
                                round_start_state,
                                stage="sampled_loss",
                                sampled_rate=sampled_rate,
                            )
                            return

                        _, teacher_logits = self._fjord_forward_cnn(
                            data_batch["input"],
                            self.fjord_max_rate,
                        )

                        if not self._tensor_is_finite(teacher_logits):
                            self._abort_round_and_restore(
                                round_start_state,
                                stage="teacher_logits",
                                sampled_rate=sampled_rate,
                            )
                            return

                        teacher_loss = self.criterion(
                            teacher_logits,
                            target,
                        )

                        if not self._tensor_is_finite(teacher_loss):
                            self._abort_round_and_restore(
                                round_start_state,
                                stage="teacher_loss",
                                sampled_rate=sampled_rate,
                            )
                            return

                        # teacher 仍然通过 teacher_loss 更新；
                        # KD 项仅约束 student，避免 KL 梯度反向冲击 teacher。
                        teacher_probs = F.softmax(
                            teacher_logits.detach() / temperature,
                            dim=1,
                        )

                        kd_loss = F.kl_div(
                            F.log_softmax(
                                sampled_logits / temperature,
                                dim=1,
                            ),
                            teacher_probs,
                            reduction="batchmean",
                        ) * (temperature ** 2)

                        if not self._tensor_is_finite(kd_loss):
                            self._abort_round_and_restore(
                                round_start_state,
                                stage="kd_loss",
                                sampled_rate=sampled_rate,
                            )
                            return

                        loss = teacher_loss + kd_weight * kd_loss
                        metric_logits = sampled_logits
                        metric_loss = sampled_loss

                    else:
                        # 没有 KD，或 sampled_rate == p_max：
                        # 只进行一次完整前向。
                        _, metric_logits = self._fjord_forward_cnn(
                            data_batch["input"],
                            sampled_rate,
                        )

                        if not self._tensor_is_finite(metric_logits):
                            self._abort_round_and_restore(
                                round_start_state,
                                stage="single_logits",
                                sampled_rate=sampled_rate,
                            )
                            return

                        metric_loss = self.criterion(
                            metric_logits,
                            target,
                        )

                        if not self._tensor_is_finite(metric_loss):
                            self._abort_round_and_restore(
                                round_start_state,
                                stage="single_loss",
                                sampled_rate=sampled_rate,
                            )
                            return

                        loss = metric_loss

                    if not self._tensor_is_finite(loss):
                        self._abort_round_and_restore(
                            round_start_state,
                            stage="total_loss",
                            sampled_rate=sampled_rate,
                        )
                        return

                    if loss.detach().item() > 1e4:
                        self._abort_round_and_restore(
                            round_start_state,
                            stage="loss_too_large",
                            sampled_rate=sampled_rate,
                        )
                        return

                    performance = self.metrics.evaluate(
                        metric_loss,
                        metric_logits,
                        target,
                    )

                if self.tracker is not None:
                    batch_size = target.size(0)

                    self.tracker.update_local_metrics(
                        loss.detach().item(),
                        0,
                        n_samples=batch_size,
                    )

                    for idx in range(1, 1 + len(performance)):
                        self.tracker.update_local_metrics(
                            performance[idx - 1],
                            idx,
                            n_samples=batch_size,
                        )

                with self.timer(
                    "backward_pass",
                    epoch=self.scheduler.epoch_,
                ):
                    self.optimizer.zero_grad()
                    loss.backward()

                    bad_grad_name = self._first_nonfinite_gradient()

                    if bad_grad_name is not None:
                        self._abort_round_and_restore(
                            round_start_state,
                            stage="gradient",
                            sampled_rate=sampled_rate,
                            parameter_name=bad_grad_name,
                        )
                        return

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=10.0,
                    )

                    if not self._tensor_is_finite(grad_norm):
                        self._abort_round_and_restore(
                            round_start_state,
                            stage="gradient_norm",
                            sampled_rate=sampled_rate,
                        )
                        return

                    self.optimizer.step()

                    model_is_finite, bad_parameter_name = (
                        self._model_is_finite()
                    )

                    if not model_is_finite:
                        self._abort_round_and_restore(
                            round_start_state,
                            stage="parameter_after_step",
                            sampled_rate=sampled_rate,
                            parameter_name=bad_parameter_name,
                        )
                        return

                    self.scheduler.step()

                if self._is_finished_one_comm_round():
                    self.conf.logger.log(self.timer.summary())
                    self._terminate_comm_round()
                    return

                num_batches = self.conf.num_batches_per_device_per_epoch

                if self.scheduler.local_index % num_batches == 0:
                    display_training_stat(
                        self.conf,
                        self.scheduler,
                        self.tracker,
                    )
                    self.tracker.reset()

                if self.conf.logger.meet_cache_limit():
                    self.conf.logger.save_json()