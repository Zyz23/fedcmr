# -*- coding: utf-8 -*-
"""FedRE server implementation (CVPR 2026)."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn

import pcode.master_utils as master_utils
from pcode.masters.master import Master, get_n_local_epoch
from pcode.utils.fedre import (
    fedre_payload_size,
    soft_cross_entropy,
    unpack_fedre_payload,
)
from pcode.utils.tensor_buffer import TensorBuffer


class MasterFedRE(Master):
    """Train and broadcast the common FedRE classifier.

    Model parameters are never aggregated.  Every participating client sends
    one Random Average Prototype (RAP) representation and its entangled-label
    encoding; the server uses those pairs to optimize the homogeneous head.
    """

    def __init__(self, conf):
        super(MasterFedRE, self).__init__(conf)

        if int(conf.fedre_feature_dim) <= 0:
            raise ValueError("--fedre_feature_dim must be positive.")
        if int(conf.fedre_server_epochs) <= 0:
            raise ValueError("--fedre_server_epochs must be positive.")
        if int(conf.fedre_server_batch_size) <= 0:
            raise ValueError("--fedre_server_batch_size must be positive.")
        if (
            not conf.fedre_eval_local
            and (
                conf.target_perf is not None
                or (
                    conf.early_stopping_rounds is not None
                    and conf.early_stopping_rounds > 0
                )
            )
        ):
            raise ValueError(
                "FedRE local evaluation must remain enabled when target or "
                "early stopping is configured."
            )

        self.global_head = nn.Linear(
            int(conf.fedre_feature_dim),
            int(conf.num_classes),
        )
        self.client_metric_cache = {}
        self.conf.logger.log(
            "FedRE initialized a shared Linear({}, {}) classifier; "
            "client representation mapping=adaptive average pooling, "
            "entanglement=RAP.".format(
                int(conf.fedre_feature_dim),
                int(conf.num_classes),
            )
        )

    def run(self):
        """Use the repository's activation protocol with FedRE payloads."""

        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                "Master starts FedRE round {}.".format(comm_round)
            )

            local_epochs = get_n_local_epoch(
                conf=self.conf,
                n_participated=self.conf.n_participated,
            )
            self.list_of_local_n_epochs = local_epochs
            selected_client_ids = self._random_select_clients()
            self._check_early_stopping()
            self.activate_selected_clients(
                selected_client_ids,
                self.conf.graph.comm_round,
                local_epochs,
            )

            if self.conf.is_finished:
                # Matches Worker._terminate_by_early_stopping().
                dist.barrier()
                self._save_global_head(comm_round - 1)
                self.conf.logger.log(
                    "Master stopped FedRE early at round {}.".format(
                        comm_round
                    )
                )
                return

            self._send_model_to_selected_clients(selected_client_ids)
            payloads = self._receive_models_from_selected_clients(
                selected_client_ids
            )
            self._aggregate_model_and_evaluate(
                payloads,
                selected_client_ids,
            )
            self.conf.logger.log(
                "Master finished FedRE round {}.\n".format(comm_round)
            )

        self._save_global_head(self.conf.n_comm_rounds)
        dist.barrier()
        self._finishing()

    def _save_global_head(self, comm_round):
        torch.save(
            {
                "current_comm_round": int(comm_round),
                "feature_dim": int(self.conf.fedre_feature_dim),
                "num_classes": int(self.conf.num_classes),
                "state_dict": self.global_head.state_dict(),
            },
            os.path.join(self.conf.checkpoint_root, "fedre_global_head.pt"),
        )

    def _send_model_to_selected_clients(self, selected_client_ids):
        """Broadcast only the common classifier to participating workers."""

        head_buffer = TensorBuffer(
            list(self.global_head.state_dict().values())
        )
        for worker_rank, selected_client_id in enumerate(
            selected_client_ids,
            start=1,
        ):
            dist.send(head_buffer.buffer, dst=worker_rank)
            self.conf.logger.log(
                "FedRE sent global classifier to worker-{} for client-{} "
                "({} scalars).".format(
                    worker_rank,
                    selected_client_id,
                    head_buffer.buffer.numel(),
                )
            )
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        """Receive one fixed-size entangled pair from each selected client."""

        dist.barrier()
        payload_size = fedre_payload_size(
            self.conf.fedre_feature_dim,
            self.conf.num_classes,
        )
        payloads = {
            client_id: torch.zeros(payload_size, dtype=torch.float32)
            for client_id in selected_client_ids
        }

        requests = []
        for client_id, world_id in zip(
            selected_client_ids,
            self.world_ids,
        ):
            requests.append(
                dist.irecv(payloads[client_id], src=world_id)
            )
        for request in requests:
            request.wait()
        dist.barrier()

        self.conf.logger.log(
            "FedRE received {} RAP pairs ({} algorithm scalars/client; "
            "4 simulator metric scalars/client).".format(
                len(payloads),
                int(self.conf.fedre_feature_dim)
                + int(self.conf.num_classes),
            )
        )
        return payloads

    def _decode_payloads(self, payloads):
        representations = []
        soft_labels = []
        metric_rows = []

        for client_id in sorted(payloads.keys()):
            representation, soft_label, metrics = unpack_fedre_payload(
                payloads[client_id],
                self.conf.fedre_feature_dim,
                self.conf.num_classes,
            )
            if not bool(torch.isfinite(representation).all().item()):
                raise ValueError(
                    "FedRE received non-finite representation from client-{}."
                    .format(client_id)
                )
            if not bool(torch.isfinite(soft_label).all().item()):
                raise ValueError(
                    "FedRE received non-finite soft label from client-{}."
                    .format(client_id)
                )
            if abs(float(soft_label.sum().item()) - 1.0) > 1e-4:
                raise ValueError(
                    "FedRE soft label from client-{} does not sum to one."
                    .format(client_id)
                )

            representations.append(representation)
            soft_labels.append(soft_label)
            metric_rows.append(metrics)

        return (
            torch.stack(representations, dim=0),
            torch.stack(soft_labels, dim=0),
            torch.stack(metric_rows, dim=0),
        )

    def _train_global_head(self, representations, soft_labels):
        device = torch.device(
            "cuda" if self.conf.graph.on_cuda else "cpu"
        )
        head = self.global_head.to(device)
        representations = representations.to(device)
        soft_labels = soft_labels.to(device)
        optimizer = torch.optim.SGD(
            head.parameters(),
            lr=float(self.conf.fedre_server_lr),
            momentum=float(self.conf.fedre_server_momentum),
        )

        generator = torch.Generator()
        generator.manual_seed(
            int(self.conf.fedre_seed)
            + 104729 * int(self.conf.graph.comm_round)
        )
        total_loss = 0.0
        total_steps = 0
        batch_size = int(self.conf.fedre_server_batch_size)

        head.train()
        for _ in range(int(self.conf.fedre_server_epochs)):
            order = torch.randperm(
                representations.size(0),
                generator=generator,
            )
            for start in range(0, order.numel(), batch_size):
                indices = order[start : start + batch_size].to(device)
                logits = head(representations[indices])
                loss = soft_cross_entropy(logits, soft_labels[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                total_steps += 1

        self.global_head = head.cpu()
        self.conf.logger.log(
            "FedRE server classifier update: epochs={}, batch_size={}, "
            "lr={}, mean_soft_ce={:.6f}.".format(
                int(self.conf.fedre_server_epochs),
                batch_size,
                float(self.conf.fedre_server_lr),
                total_loss / max(1, total_steps),
            )
        )
        torch.cuda.empty_cache()

    def _payload_performance(self):
        if len(self.client_metric_cache) == 0:
            return None

        totals = torch.stack(
            list(self.client_metric_cache.values()),
            dim=0,
        ).sum(dim=0)
        num_samples = float(totals[3].item())
        if num_samples <= 0:
            return None

        performance = {
            "loss": float(totals[0].item()) / num_samples,
            "top1": 100.0 * float(totals[1].item()) / num_samples,
            "loss2": 0.0,
        }
        if "top5" in self.metrics.metric_names:
            performance["top5"] = (
                100.0 * float(totals[2].item()) / num_samples
            )
        return performance

    def _aggregate_model_and_evaluate(self, payloads, selected_client_ids):
        representations, soft_labels, metric_rows = self._decode_payloads(
            payloads
        )
        self._train_global_head(representations, soft_labels)

        for client_id, metrics in zip(
            sorted(payloads.keys()),
            metric_rows,
        ):
            if float(metrics[3].item()) > 0:
                self.client_metric_cache[client_id] = metrics.detach().cpu()

        performance = self._payload_performance()
        if performance is None:
            self.conf.logger.log(
                "FedRE local evaluation is disabled; no coordinator metric "
                "was updated this round."
            )
            return

        self.conf.logger.log(
            "FedRE client-model average: seen_clients={}/{}, "
            "updated_clients={}, performance={}.".format(
                len(self.client_metric_cache),
                self.conf.n_clients,
                selected_client_ids,
                performance,
            )
        )
        master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.global_head,
            self.criterion,
            self.metrics,
            self.test_loaders,
            performance=performance,
            label="fedre_client_average",
        )