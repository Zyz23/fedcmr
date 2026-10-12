# -*- coding: utf-8 -*-
"""FedRE client for model-heterogeneous federated learning."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import pcode.create_dataset as create_dataset
import pcode.datasets.mixup_data as mixup
from pcode.utils.fedre import (
    average_pool_representation,
    pack_fedre_payload,
    random_average_prototype,
)
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.workers.worker import Worker


class WorkerFedRE(Worker):
    """Train a private local extractor and upload one FedRE RAP pair.

    A logical client can be assigned to a different Gloo worker process in a
    later round.  Its private model is therefore persisted under the shared
    experiment checkpoint directory, keyed by logical client id.  Only the
    common classifier and the entangled representation-label pair traverse
    the distributed communication channel.
    """

    def extra_init(self):
        feature_dim = int(self.conf.fedre_feature_dim)
        num_classes = int(self.conf.num_classes)
        if feature_dim <= 0:
            raise ValueError("--fedre_feature_dim must be positive.")

        self.model.fedre_head = nn.Linear(feature_dim, num_classes).to(
            self.device
        )
        self._load_private_client_state()

        if self.conf.partitioned_by_user:
            self.fedre_test_loader, _ = create_dataset.define_data_loader(
                self.conf,
                dataset=self.dataset["test"],
                localdata_id=self.conf.graph.client_id - 1,
                is_train=False,
                shuffle=False,
            )
        else:
            # The repository normally exposes one global vision test set.
            # Split it deterministically across logical clients so cached
            # client metrics cover the test set once rather than evaluating
            # the full test set independently on every client each round.
            client_offset = int(self.conf.graph.client_id) - 1
            indices = list(
                range(
                    client_offset,
                    len(self.dataset["test"]),
                    int(self.conf.n_clients),
                )
            )
            loader_kwargs = {
                "batch_size": self.conf.batch_size,
                "shuffle": False,
                "num_workers": self.conf.num_workers,
                "pin_memory": self.conf.pin_memory,
                "drop_last": False,
            }
            if int(self.conf.num_workers) > 0:
                loader_kwargs["multiprocessing_context"] = "fork"
            self.fedre_test_loader = DataLoader(
                Subset(self.dataset["test"], indices),
                **loader_kwargs
            )
            self.conf.logger.log(
                "FedRE assigned {} deterministic test samples to client-{}."
                .format(len(indices), self.conf.graph.client_id)
            )

    def _client_state_path(self):
        state_dir = os.path.join(
            self.conf.checkpoint_root,
            "fedre_private_clients",
        )
        if not os.path.exists(state_dir):
            try:
                os.makedirs(state_dir)
            except OSError:
                if not os.path.isdir(state_dir):
                    raise
        return os.path.join(
            state_dir,
            "client_{}.pt".format(int(self.conf.graph.client_id)),
        )

    def _load_private_client_state(self):
        state_path = self._client_state_path()
        if not os.path.exists(state_path):
            self.conf.logger.log(
                "FedRE initializes private state for client-{}.".format(
                    self.conf.graph.client_id
                )
            )
            return

        state_dict = torch.load(state_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.conf.logger.log(
            "FedRE restored private state for client-{}.".format(
                self.conf.graph.client_id
            )
        )

    def _save_private_client_state(self):
        state_path = self._client_state_path()
        temporary_path = "{}.rank-{}.tmp".format(state_path, self.rank)
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }
        torch.save(state_dict, temporary_path)
        os.replace(temporary_path, state_path)

    def _recv_model_from_master(self):
        """Receive only the homogeneous global classifier."""

        head_state = self.model.fedre_head.state_dict()
        head_buffer = TensorBuffer(list(head_state.values()))
        received = torch.zeros_like(head_buffer.buffer, device="cpu")
        dist.recv(received, src=0)
        head_buffer.buffer.copy_(received.to(head_buffer.buffer.device))
        head_buffer.unpack(head_state.values())
        self.model.fedre_head.load_state_dict(head_state)

        self.conf.logger.log(
            "Worker-{} (client-{}) received the FedRE global classifier "
            "({} scalars).".format(
                self.rank,
                self.conf.graph.client_id,
                received.numel(),
            )
        )
        dist.barrier()

    def _mapped_feature(self, inputs):
        representation, _ = self.model(inputs)
        return average_pool_representation(
            representation,
            self.conf.fedre_feature_dim,
        )

    def _inference(self, data_batch):
        feature = self._mapped_feature(data_batch["input"])
        output = self.model.fedre_head(feature)

        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )
            performance_a = self.metrics.evaluate(
                loss, output, data_batch["target_a"]
            )
            performance_b = self.metrics.evaluate(
                loss, output, data_batch["target_b"]
            )
            performance = [
                data_batch["mixup_lambda"] * first
                + (1 - data_batch["mixup_lambda"]) * second
                for first, second in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(
                loss,
                output,
                data_batch["target"],
            )

        if self.tracker is not None:
            batch_size = data_batch["target"].size(0)
            self.tracker.update_local_metrics(
                loss.item(),
                0,
                n_samples=batch_size,
            )
            for index, value in enumerate(performance, start=1):
                self.tracker.update_local_metrics(
                    value,
                    index,
                    n_samples=batch_size,
                )
        return loss, performance, feature, output

    def _collect_class_statistics(self):
        feature_dim = int(self.conf.fedre_feature_dim)
        num_classes = int(self.conf.num_classes)
        class_sums = torch.zeros(
            num_classes,
            feature_dim,
            device=self.device,
        )
        class_counts = torch.zeros(num_classes, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for batch_index, (_input, _target) in enumerate(self.train_loader):
                if (
                    self.conf.fedre_max_proto_batches > 0
                    and batch_index >= self.conf.fedre_max_proto_batches
                ):
                    break

                data_batch = create_dataset.load_data_batch(
                    self.conf,
                    _input,
                    _target,
                    is_training=False,
                    device=self.device,
                )
                mapped = self._mapped_feature(data_batch["input"])
                target = data_batch["target"].long()
                class_sums.index_add_(0, target, mapped)
                class_counts.index_add_(
                    0,
                    target,
                    torch.ones_like(target, dtype=class_counts.dtype),
                )

        generator = torch.Generator()
        generator.manual_seed(
            int(self.conf.fedre_seed)
            + 1000003 * int(self.conf.graph.comm_round)
            + 9176 * int(self.conf.graph.client_id)
        )
        return random_average_prototype(
            class_sums,
            class_counts,
            generator=generator,
        )

    def _evaluate_local_model(self):
        if not self.conf.fedre_eval_local:
            return 0.0, 0.0, 0.0, 0.0

        loss_sum = 0.0
        top1_correct = 0.0
        top5_correct = 0.0
        num_samples = 0.0
        topk = min(5, int(self.conf.num_classes))

        self.model.eval()
        with torch.no_grad():
            for batch_index, (_input, _target) in enumerate(
                self.fedre_test_loader
            ):
                if (
                    self.conf.fedre_eval_max_batches > 0
                    and batch_index >= self.conf.fedre_eval_max_batches
                ):
                    break

                data_batch = create_dataset.load_data_batch(
                    self.conf,
                    _input,
                    _target,
                    is_training=False,
                    device=self.device,
                )
                target = data_batch["target"].long()
                feature = self._mapped_feature(data_batch["input"])
                logits = self.model.fedre_head(feature)

                loss_sum += F.cross_entropy(
                    logits,
                    target,
                    reduction="sum",
                ).item()
                predictions = logits.topk(topk, dim=1).indices
                matches = predictions.eq(target.unsqueeze(1))
                top1_correct += matches[:, :1].any(dim=1).sum().item()
                top5_correct += matches.any(dim=1).sum().item()
                num_samples += float(target.numel())

        return loss_sum, top1_correct, top5_correct, num_samples

    def _send_model_to_master(self):
        """Upload one entangled representation-label pair, never the model."""

        self.model = self.model.to(self.device)
        representation, soft_label = self._collect_class_statistics()
        metrics = self._evaluate_local_model()

        self.model = self.model.cpu()
        self._save_private_client_state()

        payload = pack_fedre_payload(
            representation.detach().cpu(),
            soft_label.detach().cpu(),
            *metrics
        )

        dist.barrier()
        dist.send(payload, dst=0)
        dist.barrier()

        self.conf.logger.log(
            "Worker-{} (client-{}) uploaded one FedRE RAP pair: "
            "algorithm_scalars={}, payload_scalars={}.".format(
                self.rank,
                self.conf.graph.client_id,
                int(self.conf.fedre_feature_dim) + int(self.conf.num_classes),
                payload.numel(),
            )
        )