# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.master_utils as master_utils
from pcode.masters.master_heterofl import MasterHeteroFL


class MasterFedDF(MasterHeteroFL):
    """FedDF baseline: heterogeneous client ensemble distills a global model."""

    def _get_distill_loader(self):
        if self.dataset.get("agg") is not None:
            loader, _ = create_dataset.define_data_loader(
                self.conf, self.dataset["agg"], is_train=False, shuffle=True
            )
            return loader
        if self.val_loader is not None:
            return self.val_loader
        return self.test_loaders[0]

    def _load_local_models(self, flatten_local_models):
        local_models = []
        for client_id, flatten_local_model in flatten_local_models.items():
            client_model = self.client_models[client_id]
            client_state = copy.deepcopy(client_model.state_dict())
            flatten_local_model.unpack(client_state.values())
            client_model.load_state_dict(client_state)
            local_models.append(client_model)
        return local_models

    def _distill_global_model(self, teacher_models):
        if len(teacher_models) == 0:
            return

        device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")
        distill_loader = self._get_distill_loader()
        temperature = self.conf.feddf_temperature
        hard_weight = self.conf.feddf_hard_label_weight
        lr = self.conf.lr if self.conf.feddf_lr is None else self.conf.feddf_lr

        student = self.master_model.to(device)
        student.train()

        teachers = [model.to(device).eval() for model in teacher_models]
        optimizer = create_optimizer.define_optimizer(
            self.conf, student, self.conf.optimizer, lr=lr
        )

        self.conf.logger.log(
            f"FedDF distillation starts: teachers={len(teachers)}, epochs={self.conf.feddf_epochs}, "
            f"temperature={temperature}, lr={lr}."
        )

        for _ in range(self.conf.feddf_epochs):
            for batch_idx, (_input, _target) in enumerate(distill_loader):
                if self.conf.feddf_max_batches > 0 and batch_idx >= self.conf.feddf_max_batches:
                    break

                data_batch = create_dataset.load_data_batch(
                    self.conf, _input, _target, is_training=False, device=device
                )
                inputs = data_batch["input"]
                targets = data_batch["target"]

                with torch.no_grad():
                    teacher_probs = []
                    for teacher in teachers:
                        _, teacher_logits = teacher(inputs)
                        teacher_probs.append(F.softmax(teacher_logits / temperature, dim=1))
                    soft_targets = torch.stack(teacher_probs, dim=0).mean(dim=0)

                _, student_logits = student(inputs)
                log_probs = F.log_softmax(student_logits / temperature, dim=1)
                kd_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean") * (
                    temperature ** 2
                )

                if hard_weight > 0:
                    hard_loss = F.cross_entropy(student_logits, targets)
                    loss = (1.0 - hard_weight) * kd_loss + hard_weight * hard_loss
                else:
                    loss = kd_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10)
                optimizer.step()

        for teacher in teachers:
            teacher.cpu()
        self.master_model = student.cpu()
        torch.cuda.empty_cache()

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        teacher_models = self._load_local_models(flatten_local_models)
        fused_models = self.aggregate(flatten_local_models)
        self.master_model.load_state_dict(list(fused_models.values())[0].state_dict())
        self._distill_global_model(teacher_models)

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