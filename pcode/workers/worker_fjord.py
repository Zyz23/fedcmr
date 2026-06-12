# -*- coding: utf-8 -*-
import math

import torch
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
from pcode.utils.logging import display_training_stat
from pcode.workers.worker import Worker


class WorkerFjORD(Worker):
    """FjORD client with Ordered Dropout and self-distillation.

    Each client receives its maximal submodel p_max from the server. During
    local training it samples an ordered submodel p <= p_max for every batch.
    When KD is enabled, the p_max submodel is the teacher and participates in
    backpropagation together with the sampled student submodel.
    """

    def extra_init(self):
        self.fjord_rates = self._available_fjord_rates()
        self.fjord_max_rate = float(self.conf.model_rate_list[self.conf.graph.client_id - 1])
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) "
            f"uses FjORD p_max={self.fjord_max_rate}, sampled rates={self.fjord_rates}."
        )

    def _available_fjord_rates(self):
        p_max = float(self.conf.model_rate_list[self.conf.graph.client_id - 1])
        rates_text = getattr(self.conf, "fjord_sample_rates", "")
        if rates_text is None or len(str(rates_text).strip()) == 0:
            rates_text = self.conf.hetero_model_rates

        rates = sorted(
            {
                float(rate.strip())
                for rate in str(rates_text).split(",")
                if len(rate.strip()) > 0 and float(rate.strip()) <= p_max + 1e-12
            }
        )
        if p_max not in rates:
            rates.append(p_max)
            rates.sort()
        return rates

    def _sample_fjord_rate(self):
        return float(self.conf.random_state.choice(self.fjord_rates))

    def _active_channels(self, size, sampled_rate):
        ratio = min(1.0, float(sampled_rate) / max(self.fjord_max_rate, 1e-12))
        return max(1, min(size, int(math.ceil(size * ratio))))

    def _scale(self, x, sampled_rate):
        if self.model.training and getattr(self.conf, "fjord_use_scaler", False):
            return x / sampled_rate
        return x

    def _fjord_forward_cnn(self, inputs, sampled_rate):
        model = self.model
        required = ["conv1", "conv2", "conv3", "conv4", "fc1", "classifier"]
        if not all(hasattr(model, name) for name in required):
            raise NotImplementedError(
                "FjORD Ordered Dropout is implemented for federated_averaging_cnn/"
                "HeteroCNNCifar in this codebase. Use --arch federated_averaging_cnn."
            )

        c1 = self._active_channels(model.conv1.out_channels, sampled_rate)
        c2 = self._active_channels(model.conv2.out_channels, sampled_rate)
        c3 = self._active_channels(model.conv3.out_channels, sampled_rate)
        c4 = self._active_channels(model.conv4.out_channels, sampled_rate)
        h1 = self._active_channels(model.fc1.out_features, sampled_rate)

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

        while True:
            for _input, _target in self.train_loader:
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True, device=self.device
                    )

                sampled_rate = self._sample_fjord_rate()
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    sampled_feature, sampled_logits = self._fjord_forward_cnn(
                        data_batch["input"], sampled_rate
                    )
                    sampled_loss = self.criterion(sampled_logits, data_batch["target"])
                    performance = self.metrics.evaluate(
                        sampled_loss, sampled_logits, data_batch["target"]
                    )

                    if getattr(self.conf, "fjord_kd", True):
                        _, teacher_logits = self._fjord_forward_cnn(
                            data_batch["input"], self.fjord_max_rate
                        )
                        teacher_loss = self.criterion(
                            teacher_logits, data_batch["target"]
                        )
                        if sampled_rate < self.fjord_max_rate:
                            temperature = self.conf.fjord_temperature
                            kd_loss = F.kl_div(
                                F.log_softmax(sampled_logits / temperature, dim=1),
                                F.softmax(teacher_logits / temperature, dim=1),
                                reduction="batchmean",
                            ) * (temperature ** 2)
                            loss = teacher_loss + self.conf.fjord_kd_weight * kd_loss
                        else:
                            loss = teacher_loss
                    else:
                        loss = sampled_loss

                if self.tracker is not None:
                    bsz = data_batch["target"].size(0)
                    self.tracker.update_local_metrics(loss.item(), 0, n_samples=bsz)
                    for idx in range(1, 1 + len(performance)):
                        self.tracker.update_local_metrics(
                            performance[idx - 1], idx, n_samples=bsz
                        )

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(), max_norm=10
                    )
                    self.optimizer.step()
                    self.scheduler.step()

                if loss > 1e4 or torch.isnan(loss):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )
                    self._terminate_comm_round()
                    return

                if self._is_finished_one_comm_round():
                    self.conf.logger.log(self.timer.summary())
                    self._terminate_comm_round()
                    return

            display_training_stat(self.conf, self.scheduler, self.tracker)
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()