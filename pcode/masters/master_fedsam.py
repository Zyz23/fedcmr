# -*- coding: utf-8 -*-
import copy

import torch

import pcode.create_dataset as create_dataset
import pcode.create_model as create_model
import pcode.master_utils as master_utils
from pcode.masters.master import Master


class MasterFedSam(Master):
    """FedSam server: cluster sampling and memory aggregation.

    FedSam keeps homogeneous client models. The server merges clients selected
    by the previous round's virtual-data evaluation with newly random clients,
    then updates the global model with a memory blend of the previous global
    model and the current local-model aggregation.
    """

    def __init__(self, conf):
        super(MasterFedSam, self).__init__(conf)
        self.similar_client_ids = []
        self.virtual_loader = self._define_virtual_loader()
        conf.logger.log(
            "FedSam initialized with server-side virtual data, cluster sampling, "
            "memory aggregation, and client-side DP upload perturbation."
        )

    def _define_virtual_loader(self):
        if self.dataset.get("agg") is not None:
            loader, _ = create_dataset.define_data_loader(
                self.conf, self.dataset["agg"], is_train=False, shuffle=False
            )
            self.conf.logger.log("FedSam uses dataset['agg'] as virtual dataset D_v.")
            return loader
        if self.val_loader is not None:
            self.conf.logger.log("FedSam uses validation split as virtual dataset D_v.")
            return self.val_loader
        if getattr(self.conf, "fedsam_allow_test_virtual", False):
            self.conf.logger.log(
                "FedSam uses test loader as D_v because --fedsam_allow_test_virtual=True."
            )
            return self.test_loaders[0]
        raise ValueError(
            "FedSam requires a server-side virtual dataset D_v. Set --val_data_ratio > 0 "
            "or --agg_data_ratio > 0. Avoid test data unless explicitly allowed."
        )

    def _random_select_clients(self):
        n_selected = self.conf.n_participated
        keep_similar = min(len(self.similar_client_ids), n_selected)
        selected = list(self.similar_client_ids[:keep_similar])

        remaining_pool = [client_id for client_id in self.client_ids if client_id not in selected]
        n_random = n_selected - len(selected)
        if n_random > 0:
            random_clients = self.conf.random_state.choice(
                remaining_pool, n_random, replace=False
            ).tolist()
            selected.extend(random_clients)

        selected.sort()
        self.conf.logger.log(
            f"FedSam selected clients: similar_previous={self.similar_client_ids[:keep_similar]}, "
            f"final={selected}."
        )
        return selected

    def _memory_aggregate_state(self, previous_state, current_state):
        memory_lambda = float(self.conf.fedsam_memory_lambda)
        if memory_lambda < 0 or memory_lambda > 1:
            raise ValueError("--fedsam_memory_lambda must be in [0, 1].")

        blended = copy.deepcopy(current_state)
        for key, current_value in current_state.items():
            previous_value = previous_state[key]
            if current_value.dtype.is_floating_point:
                blended[key] = (
                    (1.0 - memory_lambda) * previous_value.float()
                    + memory_lambda * current_value.float()
                ).to(current_value.dtype)
            else:
                blended[key] = current_value
        return blended

    def _load_local_models(self, flatten_local_models):
        local_models = {}
        for client_id, flatten_local_model in flatten_local_models.items():
            arch = self.clientid2arch[client_id]
            model_state_dict = copy.deepcopy(self.client_models[arch].state_dict())
            flatten_local_model.unpack(model_state_dict.values())
            _, local_model = create_model.define_model(
                self.conf, to_consistent_model=False, arch=arch
            )
            local_model.load_state_dict(model_state_dict)
            local_models[client_id] = local_model
        return local_models

    def _evaluate_model_on_virtual_data(self, model):
        device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")
        model = model.to(device)
        model.eval()

        correct = 0.0
        total = 0
        with torch.no_grad():
            for batch_idx, (_input, _target) in enumerate(self.virtual_loader):
                if (
                    self.conf.fedsam_virtual_max_batches > 0
                    and batch_idx >= self.conf.fedsam_virtual_max_batches
                ):
                    break
                data_batch = create_dataset.load_data_batch(
                    self.conf, _input, _target, is_training=False, device=device
                )
                _, logits = model(data_batch["input"])
                pred = logits.argmax(dim=1)
                correct += pred.eq(data_batch["target"]).float().sum().item()
                total += data_batch["target"].numel()

        model.cpu()
        return correct / max(1, total)

    def _update_similar_clients(self, flatten_local_models, selected_client_ids):
        if self.conf.fedsam_similar_ratio <= 0:
            self.similar_client_ids = []
            return

        local_models = self._load_local_models(flatten_local_models)
        scores = []
        for client_id in selected_client_ids:
            score = self._evaluate_model_on_virtual_data(local_models[client_id])
            scores.append((client_id, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        n_similar = max(
            1,
            min(
                len(scores),
                int(round(self.conf.n_participated * self.conf.fedsam_similar_ratio)),
            ),
        )
        self.similar_client_ids = sorted([client_id for client_id, _ in scores[:n_similar]])
        self.conf.logger.log(
            "FedSam virtual-data client scores: "
            + str({client_id: float(score) for client_id, score in scores})
        )
        self.conf.logger.log(
            f"FedSam selected similar clients for next round: {self.similar_client_ids}."
        )

        for model in local_models.values():
            model.cpu()
        torch.cuda.empty_cache()

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        previous_state = copy.deepcopy(self.master_model.state_dict())

        current_model = self._fedavg(flatten_local_models)
        blended_state = self._memory_aggregate_state(
            previous_state, current_model.state_dict()
        )
        self.master_model.load_state_dict(blended_state)

        for arch in self.used_client_archs:
            self.client_models[arch].load_state_dict(blended_state)

        master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label="aggregated_test_loader_0",
        )

        self._update_similar_clients(flatten_local_models, selected_client_ids)
        torch.cuda.empty_cache()