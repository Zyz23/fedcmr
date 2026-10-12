# -*- coding: utf-8 -*-
"""Synchronous FedGMR server with a complete, directly evaluated global model."""

import copy
import os

import torch
import torch.distributed as dist

import pcode.master_utils as master_utils

from pcode.masters.master import Master
from pcode.utils.fedgmr import (
    FedGMRRestorationController,
    aggregate_masked_model_states,
    build_nested_importance_masks,
    clone_state_dict_to_cpu,
    first_nonfinite_tensor_name,
    flatten_mask_dict,
    get_prunable_parameter_names,
    mask_state_dict,
)
from pcode.utils.tensor_buffer import TensorBuffer


class MasterFedGMR(Master):
    """FedGMR-Sync: GMR + synchronous mask-aware aggregation.

    The paper's production protocol uses a virtual-time semi-asynchronous
    simulator.  This repository has a round-synchronous Gloo runtime, so this
    class implements the synchronous specialization used by the paper's
    analysis and ``w/o Asyn`` ablation.  The server model is never pruned.
    """

    def __init__(self, conf):
        # FedGMR capacity is a parameter mask density, not physical width.
        conf.model_rate = 1.0
        super(MasterFedGMR, self).__init__(conf)

        master_arch = self.conf.arch_info["master"]
        incompatible_archs = {
            arch
            for arch in self.clientid2arch.values()
            if arch != master_arch
        }
        if incompatible_archs:
            raise ValueError(
                "FedGMR uses masked sub-models of one global architecture; "
                f"master={master_arch}, incompatible workers="
                f"{sorted(incompatible_archs)}."
            )

        initial_densities = getattr(conf, "model_rate_list", None)
        if initial_densities is None:
            initial_densities = [1.0 for _ in self.client_ids]

        self.prunable_parameter_names = get_prunable_parameter_names(
            self.master_model
        )
        if len(self.prunable_parameter_names) == 0:
            raise RuntimeError(
                "FedGMR requires at least one trainable matrix/tensor weight."
            )

        self.restoration = FedGMRRestorationController(
            initial_densities=initial_densities,
            density_ladder=getattr(conf, "fedgmr_density_ladder", ""),
            patience=getattr(conf, "fedgmr_patience", 5),
            min_delta=getattr(conf, "fedgmr_min_delta", 0.0),
            check_interval=getattr(conf, "fedgmr_check_interval", 5),
        )

        self.mask_refresh_interval = max(
            1,
            int(getattr(conf, "fedgmr_mask_refresh_interval", 1)),
        )
        self.previous_global_state = {
            name: torch.zeros_like(value, device="cpu")
            for name, value in self.master_model.state_dict().items()
        }
        self.cached_masks_by_density = {}
        self.cached_density_signature = None
        self.last_mask_refresh_round = None
        self.current_client_masks = {}

        self.conf.logger.log(
            "FedGMR-Sync initialized with a complete server model: "
            f"initial_densities={self.restoration.client_densities}, "
            f"density_ladder={self.restoration.density_ladder}, "
            f"patience={self.restoration.patience}, "
            f"check_interval={self.restoration.check_interval}, "
            f"mask_refresh_interval={self.mask_refresh_interval}."
        )

    def _random_select_clients(self):
        """Stratify by current density so each round covers capacity groups."""
        density_to_clients = {}
        for client_id, density in zip(
            self.client_ids,
            self.restoration.client_densities,
        ):
            density_to_clients.setdefault(float(density), []).append(client_id)

        densities = list(density_to_clients.keys())
        self.conf.random_state.shuffle(densities)
        selected = []

        for density in densities:
            if len(selected) >= self.conf.n_participated:
                break
            selected.append(
                int(
                    self.conf.random_state.choice(
                        density_to_clients[density],
                        size=1,
                        replace=False,
                    )[0]
                )
            )

        if len(selected) < self.conf.n_participated:
            remaining = [
                client_id
                for client_id in self.client_ids
                if client_id not in selected
            ]
            extra = self.conf.random_state.choice(
                remaining,
                size=self.conf.n_participated - len(selected),
                replace=False,
            ).tolist()
            selected.extend(int(client_id) for client_id in extra)

        selected.sort()
        selected_densities = [
            self.restoration.client_densities[client_id - 1]
            for client_id in selected
        ]
        self.conf.logger.log(
            f"FedGMR selected clients={selected}, "
            f"densities={selected_densities}."
        )
        return selected

    def _refresh_masks_if_needed(self):
        round_idx = int(self.conf.graph.comm_round)
        density_signature = tuple(self.restoration.client_densities)
        density_changed = density_signature != self.cached_density_signature
        interval_elapsed = (
            self.last_mask_refresh_round is None
            or round_idx - self.last_mask_refresh_round
            >= self.mask_refresh_interval
        )

        if not density_changed and not interval_elapsed:
            return

        unique_densities = sorted(set(density_signature))
        masks = build_nested_importance_masks(
            global_state_dict=self.master_model.state_dict(),
            previous_global_state_dict=self.previous_global_state,
            parameter_names=self.prunable_parameter_names,
            densities=unique_densities,
        )
        self.cached_masks_by_density = dict(zip(unique_densities, masks))
        self.cached_density_signature = density_signature
        self.last_mask_refresh_round = round_idx

        self.conf.logger.log(
            f"FedGMR refreshed nested importance masks at round={round_idx}, "
            f"densities={unique_densities}."
        )

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        """Send each worker its density and exact fixed mask for this round."""
        self._refresh_masks_if_needed()
        self.current_client_masks = {}

        for worker_rank, client_id in enumerate(selected_client_ids, start=1):
            density = float(
                self.restoration.client_densities[client_id - 1]
            )
            mask_dict = self.cached_masks_by_density[density]
            self.current_client_masks[client_id] = mask_dict

            flat_mask = flatten_mask_dict(
                mask_dict,
                self.prunable_parameter_names,
                dtype=torch.float32,
            )
            payload = torch.cat(
                [torch.tensor([density], dtype=torch.float32), flat_mask],
                dim=0,
            )
            dist.send(payload, dst=worker_rank)

            self.conf.logger.log(
                f"[FEDGMR MASK] worker_rank={worker_rank}, "
                f"client_id={client_id}, density={density:.6f}, "
                f"active={int(flat_mask.sum().item())}/{flat_mask.numel()}."
            )

    def _send_model_to_selected_clients(self, selected_client_ids):
        """Extract and send masked sub-models from the complete global model."""
        self.conf.logger.log("Master sends FedGMR masked sub-models.")
        global_state = self.master_model.state_dict()

        for worker_rank, client_id in enumerate(selected_client_ids, start=1):
            if client_id not in self.current_client_masks:
                raise RuntimeError(
                    f"Missing FedGMR mask for selected client {client_id}."
                )

            client_state = mask_state_dict(
                global_state,
                self.current_client_masks[client_id],
                to_cpu=True,
            )
            flatten_model = TensorBuffer(
                list(client_state.values()),
                use_cuda=False,
            )
            dist.send(flatten_model.buffer, dst=worker_rank)

            density = self.restoration.client_densities[client_id - 1]
            self.conf.logger.log(
                f"[FEDGMR SEND] worker_rank={worker_rank}, "
                f"client_id={client_id}, density={density:.6f}, "
                f"dense_transport_numel={flatten_model.buffer.numel()}."
            )

        dist.barrier()

    def aggregate(self, flatten_local_models):
        local_states_and_masks = []

        for client_id, flatten_local_model in flatten_local_models.items():
            arch = self.clientid2arch[client_id]
            local_state = copy.deepcopy(
                self.client_models[arch].state_dict()
            )
            flatten_local_model.unpack(local_state.values())

            if client_id not in self.current_client_masks:
                raise RuntimeError(
                    f"Missing FedGMR aggregation mask for client {client_id}."
                )

            local_states_and_masks.append(
                (local_state, self.current_client_masks[client_id])
            )

        aggregated_state = aggregate_masked_model_states(
            global_state_dict=self.master_model.state_dict(),
            local_states_and_masks=local_states_and_masks,
            parameter_names=self.prunable_parameter_names,
        )
        aggregated_model = copy.deepcopy(self.master_model)
        aggregated_model.load_state_dict(aggregated_state, strict=True)
        return {0: aggregated_model}

    def _restoration_metric(self, test_performance):
        metric_name = self.metrics.metric_names[0]

        if self.val_loader is None:
            metric = float(test_performance[metric_name])
            self.conf.logger.log(
                "FedGMR has no validation split; restoration uses the "
                f"current full-global test {metric_name}={metric:.6f}."
            )
            return metric

        validation_performance = master_utils.validate(
            conf=self.conf,
            coordinator=self.coordinator,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            data_loader=self.val_loader,
            label=None,
            display=False,
        )
        metric = float(validation_performance[metric_name])
        self.conf.logger.log(
            f"FedGMR restoration validation at round="
            f"{self.conf.graph.comm_round}: {metric_name}={metric:.6f}."
        )
        return metric

    def _aggregate_model_and_evaluate(
        self,
        flatten_local_models,
        selected_client_ids,
    ):
        round_start_state = clone_state_dict_to_cpu(
            self.master_model.state_dict()
        )

        try:
            candidate_model = list(
                self.aggregate(flatten_local_models).values()
            )[0]
            bad_name = first_nonfinite_tensor_name(
                candidate_model.state_dict()
            )
            if bad_name is not None:
                raise FloatingPointError(
                    f"aggregated model is non-finite at '{bad_name}'"
                )
            self.master_model.load_state_dict(
                candidate_model.state_dict(),
                strict=True,
            )
        except FloatingPointError as error:
            self.conf.logger.log(
                f"[FEDGMR SERVER] Reject round "
                f"{self.conf.graph.comm_round}: {error}."
            )

        # The next mask refresh estimates gradients with this global change.
        self.previous_global_state = round_start_state

        test_performance = master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label="fedgmr_full_global_test",
        )

        if self.coordinator.key_metric.is_best:
            self._save_global_checkpoint("fedgmr_global_best.pt")

        if self.restoration.should_check(self.conf.graph.comm_round):
            metric = self._restoration_metric(test_performance)
            event = self.restoration.observe(
                metric,
                self.conf.graph.comm_round,
            )
            if event is not None:
                self.conf.logger.log(
                    "[FEDGMR RESTORE] "
                    f"round={event['round']}, metric={event['metric']:.6f}, "
                    f"floor={event['old_floor']:.6f}->"
                    f"{event['new_floor']:.6f}, "
                    f"densities={event['new_densities']}."
                )

        if self.conf.graph.on_cuda:
            torch.cuda.empty_cache()

    def _save_global_checkpoint(self, filename):
        os.makedirs(self.conf.checkpoint_root, exist_ok=True)
        destination = os.path.join(self.conf.checkpoint_root, filename)
        temporary = destination + ".tmp"
        payload = {
            "method": "fedgmr_sync",
            "arch": self.conf.arch,
            "comm_round": int(getattr(self.conf.graph, "comm_round", 0)),
            "best_global_top1": self.coordinator.key_metric.best_perf,
            "model_state_dict": clone_state_dict_to_cpu(
                self.master_model.state_dict()
            ),
            "restoration_state": self.restoration.state_dict(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        self.conf.logger.log(
            f"FedGMR saved complete global model to {destination}."
        )

    def _finishing(self, _comm_round=None):
        self._save_global_checkpoint("fedgmr_global_final.pt")
        super(MasterFedGMR, self)._finishing(_comm_round)