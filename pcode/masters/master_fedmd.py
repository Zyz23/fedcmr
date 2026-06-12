# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.master_utils as master_utils
from pcode.masters.master_feddf import MasterFedDF


class MasterFedMD(MasterFedDF):
    """FedMD-style public-logit consensus distillation for heterogeneous clients.

    FedMD forms a consensus prediction on public data from heterogeneous local
    models. This server-global variant distills that consensus into the full
    server model so the final metric remains the aggregated global model.
    """

    def _distill_global_model(self, teacher_models):
        if len(teacher_models) == 0:
            return

        device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")
        distill_loader = self._get_distill_loader()
        temperature = self.conf.fedmd_temperature
        hard_weight = self.conf.fedmd_hard_label_weight
        lr = self.conf.lr if self.conf.fedmd_lr is None else self.conf.fedmd_lr

        student = self.master_model.to(device)
        student.train()

        teachers = [model.to(device).eval() for model in teacher_models]
        optimizer = create_optimizer.define_optimizer(
            self.conf, student, self.conf.optimizer, lr=lr
        )

        self.conf.logger.log(
            f"FedMD consensus distillation starts: teachers={len(teachers)}, epochs={self.conf.fedmd_epochs}, "
            f"temperature={temperature}, loss={self.conf.fedmd_loss}, lr={lr}."
        )

        for _ in range(self.conf.fedmd_epochs):
            for batch_idx, (_input, _target) in enumerate(distill_loader):
                if self.conf.fedmd_max_batches > 0 and batch_idx >= self.conf.fedmd_max_batches:
                    break

                data_batch = create_dataset.load_data_batch(
                    self.conf, _input, _target, is_training=False, device=device
                )
                inputs = data_batch["input"]
                targets = data_batch["target"]

                with torch.no_grad():
                    teacher_logits = []
                    for teacher in teachers:
                        _, logits = teacher(inputs)
                        teacher_logits.append(logits / temperature)
                    consensus_logits = torch.stack(teacher_logits, dim=0).mean(dim=0)

                _, student_logits = student(inputs)
                scaled_student_logits = student_logits / temperature

                if self.conf.fedmd_loss == "kl":
                    consensus_probs = F.softmax(consensus_logits, dim=1)
                    student_log_probs = F.log_softmax(scaled_student_logits, dim=1)
                    distill_loss = F.kl_div(
                        student_log_probs, consensus_probs, reduction="batchmean"
                    ) * (temperature ** 2)
                else:
                    distill_loss = F.mse_loss(scaled_student_logits, consensus_logits)

                if hard_weight > 0:
                    hard_loss = F.cross_entropy(student_logits, targets)
                    loss = (1.0 - hard_weight) * distill_loss + hard_weight * hard_loss
                else:
                    loss = distill_loss

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