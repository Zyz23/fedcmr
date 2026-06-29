# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import pcode.create_dataset as create_dataset
import pcode.master_utils as master_utils
from pcode.masters.master_heterofl import MasterHeteroFL


class MasterFjORD(MasterHeteroFL):
    """
    FjORD server with finite-value protection.

    不修改底层 HeteroFL 的切片与聚合代码：
    - 仍使用 MasterHeteroFL 的完整模型、子模型下发和重叠聚合；
    - 聚合后检查全局模型是否含 NaN/Inf；
    - 若异常，则回退到上一轮有限的全局模型；
    - 重写 t-SNE，遇到 NaN/Inf 时跳过，而不是让 Master 进程崩溃。
    """

    def __init__(self, conf):
        super(MasterFjORD, self).__init__(conf)

        # 保存最近一次“正常有限”的全局模型状态。
        self.last_finite_master_state = self._clone_state_to_cpu(
            self.master_model.state_dict()
        )

        self.conf.logger.log(
            "MasterFjORD initialized with finite-state rollback protection."
        )

    @staticmethod
    def _clone_state_to_cpu(state_dict):
        return {
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
        }

    @staticmethod
    def _find_nonfinite_key(state_dict):
        """
        返回第一个包含 NaN/Inf 的浮点参数名；
        若全部正常则返回 None。
        """
        for key, value in state_dict.items():
            if value.is_floating_point():
                if not torch.isfinite(value).all().item():
                    return key
        return None

    def _restore_last_finite_master(self, reason):
        self.conf.logger.log(
            f"[FjORD SERVER ROLLBACK] {reason}. "
            f"Restore the last finite global model."
        )

        self.master_model.load_state_dict(
            self.last_finite_master_state,
            strict=True,
        )

        if self.conf.graph.on_cuda:
            torch.cuda.empty_cache()

    def _send_model_to_selected_clients(self, selected_client_ids):
        """
        下发前再检查一次服务器模型。
        防止任何意外 NaN 状态继续下发给 worker。
        """
        bad_key = self._find_nonfinite_key(self.master_model.state_dict())

        if bad_key is not None:
            self._restore_last_finite_master(
                reason=f"global model already contains NaN/Inf at {bad_key} before send"
            )

        super(MasterFjORD, self)._send_model_to_selected_clients(
            selected_client_ids
        )

    def _aggregate_model_and_evaluate(
        self,
        flatten_local_models,
        selected_client_ids,
    ):
        """
        先执行原 HeteroFL 聚合，再检查聚合结果。
        如果任何客户端上传 NaN 模型导致聚合结果异常，
        本轮不更新全局模型，直接恢复到最近正常状态。
        """
        try:
            client_models = self.aggregate(flatten_local_models)
            candidate_model = list(client_models.values())[0]
        except Exception as error:
            self._restore_last_finite_master(
                reason=f"aggregation failed: {repr(error)}"
            )
            return

        bad_key = self._find_nonfinite_key(candidate_model.state_dict())

        if bad_key is not None:
            self._restore_last_finite_master(
                reason=(
                    f"aggregated global model contains NaN/Inf "
                    f"at parameter '{bad_key}'"
                )
            )
            return

        # 本轮聚合正常，正式更新服务器全局模型。
        self.master_model.load_state_dict(
            candidate_model.state_dict(),
            strict=True,
        )

        # 更新最近正常全局状态，用于后续回滚。
        self.last_finite_master_state = self._clone_state_to_cpu(
            self.master_model.state_dict()
        )

        master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label="aggregated_test_loader_0",
        )

        if self.conf.graph.on_cuda:
            torch.cuda.empty_cache()

    def visualize_global_features(self):
        """
        覆盖 Master 中默认的 t-SNE。
        特征或输出出现 NaN/Inf 时记录日志并跳过，
        不让 Master 崩溃，从而避免 worker 连带出现 Gloo barrier 错误。
        """
        bad_key = self._find_nonfinite_key(self.master_model.state_dict())

        if bad_key is not None:
            self.conf.logger.log(
                f"[FjORD t-SNE] Skip visualization: global model contains "
                f"NaN/Inf at '{bad_key}'."
            )
            return

        self.conf.logger.log(
            "Extracting finite global features for t-SNE visualization..."
        )

        self.master_model.eval()
        moved_to_cuda = False

        if self.conf.graph.on_cuda:
            self.master_model = self.master_model.cuda()
            moved_to_cuda = True

        all_features = []
        all_targets = []

        try:
            with torch.no_grad():
                for data_loader in self.test_loaders:
                    for _input, _target in data_loader:
                        data_batch = create_dataset.load_data_batch(
                            self.conf,
                            _input,
                            _target,
                            is_training=False,
                            device=next(self.master_model.parameters()).device,
                        )

                        feature, output = self.master_model(
                            data_batch["input"]
                        )

                        if not torch.isfinite(feature).all().item():
                            self.conf.logger.log(
                                "[FjORD t-SNE] Skip visualization: "
                                "feature contains NaN/Inf."
                            )
                            return

                        if not torch.isfinite(output).all().item():
                            self.conf.logger.log(
                                "[FjORD t-SNE] Skip visualization: "
                                "logits contain NaN/Inf."
                            )
                            return

                        all_features.append(feature.detach().cpu().numpy())
                        all_targets.append(
                            data_batch["target"].detach().cpu().numpy()
                        )

            if len(all_features) == 0:
                self.conf.logger.log(
                    "[FjORD t-SNE] Skip visualization: no features extracted."
                )
                return

            all_features = np.concatenate(all_features, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            if not np.isfinite(all_features).all():
                self.conf.logger.log(
                    "[FjORD t-SNE] Skip visualization: concatenated features "
                    "contain NaN/Inf."
                )
                return

            self.conf.logger.log(
                f"Extracted features shape: {all_features.shape}, "
                "running t-SNE..."
            )

            tsne = TSNE(
                n_components=2,
                random_state=42,
                init="pca",
                learning_rate="auto",
            )

            features_2d = tsne.fit_transform(all_features)

            plt.figure(figsize=(10, 8))

            num_classes = len(np.unique(all_targets))
            palette = (
                sns.color_palette("tab10", num_classes)
                if num_classes <= 10
                else sns.color_palette("husl", num_classes)
            )

            sns.scatterplot(
                x=features_2d[:, 0],
                y=features_2d[:, 1],
                hue=all_targets,
                palette=palette,
                legend="full",
                alpha=0.7,
                s=20,
            )

            plt.title(
                f"t-SNE visualization of FjORD Global Features "
                f"(Round {self.conf.graph.comm_round})"
            )
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend(
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.0,
            )
            plt.tight_layout()

            save_path = os.path.join(
                self.conf.checkpoint_root,
                "fjord_global_model_tsne_final_round.png",
            )

            plt.savefig(save_path, dpi=300)
            plt.close()

            self.conf.logger.log(
                f"FjORD t-SNE visualization saved to {save_path}."
            )

        except Exception as error:
            self.conf.logger.log(
                f"[FjORD t-SNE] Skip visualization due to error: {repr(error)}"
            )

        finally:
            if moved_to_cuda:
                self.master_model = self.master_model.cpu()
                torch.cuda.empty_cache()