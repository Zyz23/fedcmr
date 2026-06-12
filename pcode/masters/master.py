# # -*- coding: utf-8 -*-
# import os
# import copy

# import numpy as np
# import torch
# import torch.distributed as dist
# from pcode.utils.module_state import ModuleState
# import pcode.master_utils as master_utils
# import pcode.create_coordinator as create_coordinator
# import pcode.create_dataset as create_dataset
# import pcode.create_metrics as create_metrics
# import pcode.create_model as create_model
# import pcode.utils.checkpoint as checkpoint
# from pcode.utils.tensor_buffer import TensorBuffer
# import pcode.utils.cross_entropy as cross_entropy
# from pcode.utils.early_stopping import EarlyStoppingTracker


# class Master(object):
#     def __init__(self, conf):
#         self.conf = conf

#         # some initializations.
#         self.client_ids = list(range(1, 1 + conf.n_clients)) #总的客户端id列表
#         self.world_ids = list(range(1, 1 + conf.n_participated))  # 总的参与客户端id列表 

#         # create model as well as their corresponding state_dicts.
#         # 创建服务器模型 不需要arch
#         _, self.master_model = create_model.define_model( # 获取模型
#             conf, to_consistent_model=False
#         )

#         # 所有客户端的模型架构 
#         self.used_client_archs = set( # set去重，打印所有客户端架构
#             [
#                 create_model.determine_arch(conf, client_id, use_complex_arch=True)
#                 for client_id in range(1, 1 + conf.n_clients)
#             ]
#         )
#         self.conf.used_client_archs = self.used_client_archs
#         # 打印日志 客户端使用的模型架构 和服务器已经为客户端创建模型架构
#         conf.logger.log(f"The client will use archs={self.used_client_archs}.")
#         conf.logger.log("Master created model templates for client models.")

#         #创建客户端模型字典 便于异构聚合 包含了模型参数
#         self.client_models = dict( # 获取客户端模型信息
#             create_model.define_model(conf, to_consistent_model=False, arch=arch)
#             for arch in self.used_client_archs
#         )
#         # 确定每一个客户的具体使用的什么模型 只有名称 没有返回模型参数

#         self.clientid2arch = dict( # 获取所有客户端的模型名称
#             (
#                 client_id,
#                 create_model.determine_arch(
#                     conf, client_id=client_id, use_complex_arch=True
#                 ),
#             )
#             for client_id in range(1, 1 + conf.n_clients)
#         )
#         self.conf.clientid2arch = self.clientid2arch

#         conf.logger.log(
#             f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
#         )

#         # create dataset (as well as the potential data_partitioner) for training.
#         dist.barrier()
#         self.dataset = create_dataset.define_dataset(conf, data=conf.data, agg_data_ratio=conf.agg_data_ratio)
#         _, self.data_partitioner = create_dataset.define_data_loader(
#             self.conf,
#             dataset=self.dataset["train"],
#             localdata_id=0,  # random id here.
#             is_train=True,
#             data_partitioner=None,
#         ) # 虽然划分了数据但是全丢了，只保留数据划分器



#         conf.logger.log(f"Master initialized the local training data with workers.")

#         # create val loader.
#         # right now we just ignore the case of partitioned_by_user.
#         if self.dataset["val"] is not None:
#             assert not conf.partitioned_by_user
#             self.val_loader, _ = create_dataset.define_data_loader(
#                 conf, self.dataset["val"], is_train=False
#             )
#             conf.logger.log(f"Master initialized val data.")
#         else:
#             self.val_loader = None

#         # create test loaders.
#         # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
#         if conf.partitioned_by_user:
#             self.test_loaders = []
#             for localdata_id in self.client_ids:
#                 test_loader, _ = create_dataset.define_data_loader(
#                     conf,
#                     self.dataset["test"],
#                     localdata_id=localdata_id - 1,
#                     is_train=False,
#                     shuffle=False,
#                 )
#                 self.test_loaders.append(copy.deepcopy(test_loader))
#         else:
#             test_loader, _ = create_dataset.define_data_loader(
#                 conf, self.dataset["test"], is_train=False
#             )
#             self.test_loaders = [test_loader] # 这里的test_loader和上面的val_loader都是总的，也就是说未平分给客户端的

#         # define the criterion and metrics.
#         self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")   # 模型的损失函数，这里传入mean表示计算的损失会求平均
#         self.metrics = create_metrics.Metrics(self.master_model, task="classification")  # 模型的评估，计算准确率之类
#         conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

#         self.coordinator = create_coordinator.Coordinator(conf, self.metrics)   # 记录最佳性能
#         if not self.conf.train_fast:
#             self.local_coordinator = [create_coordinator.Coordinator(conf, self.metrics) for _ in
#                                       range(self.conf.n_clients)]

#         conf.logger.log(f"Master initialized the coordinator.\n")

#         # define early_stopping_tracker.
#         self.early_stopping_tracker = EarlyStoppingTracker(
#             patience=conf.early_stopping_rounds
#         )   # 早停装置，为0时关掉早停功能

#         # save arguments to disk.
#         conf.is_finished = False    # 用于控制全局训练是否已经结束
#         checkpoint.save_arguments(conf) # 将超参数保存在json文件中

#     def run(self):
#         for comm_round in range(1, 1 + self.conf.n_comm_rounds):
#             self.conf.graph.comm_round = comm_round
#             self.conf.logger.log(
#                 f"Master starting one round of federated learning: (comm_round={comm_round})."
#             )

#             # get random n_local_epochs.
#             list_of_local_n_epochs = get_n_local_epoch(
#                 conf=self.conf, n_participated=self.conf.n_participated
#             )   # 生成参与数长的列表，每个值为每个客户端的训练轮数
#             self.list_of_local_n_epochs = list_of_local_n_epochs

#             # random select clients from a pool.
#             selected_client_ids = self._random_select_clients() # 随机选择客户端

#             # detect early stopping.
#             self._check_early_stopping() # 检查是否满足早停，可能未开启早停

#             # init the activation tensor and broadcast to all clients (either start or stop).
#             self.activate_selected_clients(
#                 selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
#             )

#             # will decide to send the model or stop the training.
#             if not self.conf.is_finished:
#                 self.send_extra_info_to_selected_clients(selected_client_ids)

#                 # broadcast the model to activated clients.
#                 self._send_model_to_selected_clients(selected_client_ids)

#             else:
#                 dist.barrier()
#                 self.conf.logger.log(
#                     f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
#                 )
#                 return

#             self.receive_extra_info_from_selected_clients(selected_client_ids)

#             # wait to receive the local models.
#             flatten_local_models = self._receive_models_from_selected_clients(
#                 selected_client_ids
#             ) # 接受模型参数

#             # aggregate the local models and evaluate on the validation dataset.
#             self._aggregate_model_and_evaluate(flatten_local_models, selected_client_ids)   # 聚合模型，评估和存档

#             # evaluate the aggregated model.
#             self.conf.logger.log(f"Master finished one round of federated learning.\n")

#         # formally stop the training (the master has finished all communication rounds).
#         dist.barrier()
#         self._finishing()

#     def receive_extra_info_from_selected_clients(self, selected_client_ids):
#         pass


#     def _random_select_clients(self):
#         selected_client_ids = self.conf.random_state.choice(
#             self.client_ids, self.conf.n_participated, replace=False
#         ).tolist()  # 随机选择客户端
#         selected_client_ids.sort()
#         self.conf.logger.log(
#             f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
#         )
#         return selected_client_ids

#     def activate_selected_clients(
#             self, selected_client_ids, comm_round, list_of_local_n_epochs, to_send_history=False
#     ):
#         # Activate the selected clients:
#         # the first row indicates the client id,
#         # the second row indicates the current_comm_round,
#         # the third row indicates the expected local_n_epochs
#         selected_client_ids = np.array(selected_client_ids)
#         msg_len = 3 # 消息行数

#         activation_msg = torch.zeros((msg_len, len(selected_client_ids)))
#         activation_msg[0, :] = torch.Tensor(selected_client_ids) # 第一行装客户端id
#         activation_msg[1, :] = comm_round                       # 第二行装联邦轮次
#         activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)     # 第三行装每个客户端训练轮数

#         dist.broadcast(tensor=activation_msg, src=0)
#         self.conf.logger.log(f"Master activated the selected clients.")
#         dist.barrier()

#     def _send_model_to_selected_clients(self, selected_client_ids):
#         # the master_model can be large; the client_models can be small and different.
#         self.conf.logger.log(f"Master send the models to workers.")

#         for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
#             arch = self.clientid2arch[selected_client_id] # 获取当前循环模型名称
#             client_model_state_dict = self.client_models[arch].state_dict()

#             flatten_model = TensorBuffer(list(client_model_state_dict.values())) # 打包模型参数
#             dist.send(tensor=flatten_model.buffer, dst=worker_rank) # 发送参数到对应模型
#             self.conf.logger.log(
#                 f"\tMaster send the current model={arch} to process_id={worker_rank}."
#             )

#         dist.barrier()

#     def send_extra_info_to_selected_clients(self, selected_client_ids):
#         pass

#     def _receive_models_from_selected_clients(self, selected_client_ids):
#         self.conf.logger.log(f"Master waits to receive the local models.")
#         dist.barrier()

#         # init the placeholders to recv the local models from workers.
#         flatten_local_models = dict()
#         for selected_client_id in selected_client_ids: # 为接受数据准备容器，这也可以解释为什么要在服务器创建模型
#             arch = self.clientid2arch[selected_client_id]
#             client_tb = TensorBuffer(
#                 list(self.client_models[arch].state_dict().values())
#             )
#             client_tb.buffer = torch.zeros_like(client_tb.buffer)
#             flatten_local_models[selected_client_id] = client_tb

#         # async to receive model from clients.
#         reqs = []
#         for client_id, world_id in zip(selected_client_ids, self.world_ids):
#             req = dist.irecv(
#                 tensor=flatten_local_models[client_id].buffer, src=world_id
#             )   # 这里并不会阻塞，相当于告诉客户端数据放在哪里
#             reqs.append(req)

#         for req in reqs:
#             req.wait()  # 这里会阻塞等待数据

#         dist.barrier()
#         self.conf.logger.log(f"Master received all local models.")
#         return flatten_local_models

#     def _fedavg(self, flatten_local_models, weights=None):
#         n_selected_clients = len(flatten_local_models)

#         if weights == None:
#             weights = [
#                 torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
#             ]

#         # NOTE: the arch for different local models needs to be the same as the master model.
#         # retrieve the local models.
#         local_models = {}
#         for client_idx, flatten_local_model in flatten_local_models.items():
#             _arch = self.clientid2arch[client_idx]
#             _model = copy.deepcopy(self.client_models[_arch])
#             _model_state_dict = self.client_models[_arch].state_dict()
#             flatten_local_model.unpack(_model_state_dict.values())
#             _model.load_state_dict(_model_state_dict)
#             local_models[client_idx] = _model

#         # uniformly average the local models.
#         # assume we use the runtime stat from the last model.
#         _model = copy.deepcopy(_model)
#         local_states = [
#             ModuleState(copy.deepcopy(local_model.state_dict()))
#             for _, local_model in local_models.items()
#         ]
#         model_state = local_states[0] * weights[0]
#         for idx in range(1, len(local_states)):
#             model_state += local_states[idx] * weights[idx]
#         model_state.copy_to_module(_model)
#         return _model

#     def _avg_over_archs(self, flatten_local_models):
#         # get unique arch from this comm. round.
#         archs = set(
#             [
#                 self.clientid2arch[client_idx]
#                 for client_idx in flatten_local_models.keys()
#             ]
#         )

#         # average for each arch.
#         archs_fedavg_models = {}
#         for arch in archs:
#             # extract local_models from flatten_local_models.
#             _flatten_local_models = {}
#             for client_idx, flatten_local_model in flatten_local_models.items():
#                 if self.clientid2arch[client_idx] == arch:
#                     _flatten_local_models[client_idx] = flatten_local_model

#             # average corresponding local models.
#             self.conf.logger.log(
#                 f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
#             )

#             fedavg_model = self._fedavg(_flatten_local_models) # 平均算法
#             archs_fedavg_models[arch] = fedavg_model
#         return archs_fedavg_models

#     def aggregate(self, flatten_local_models):
#         # uniformly average local models with the same architecture.
#         fedavg_models = self._avg_over_archs(flatten_local_models) 
#         return fedavg_models

#     def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
#         # aggregate the local models.
#         client_models = self.aggregate(
#             flatten_local_models
#         ) # 计算平均后的模型

#         self.master_model.load_state_dict(
#             list(client_models.values())[0].state_dict()
#         ) # 更新全局模型
#         for arch, _client_model in client_models.items():
#             self.client_models[arch].load_state_dict(_client_model.state_dict())

#         # evaluate the aggregated model on the test data.
#         master_utils.do_validation( # 最终评估
#             self.conf,
#             self.coordinator,
#             self.master_model,
#             self.criterion,
#             self.metrics,
#             # self.test_loaders,# cifar数据集使用测试集
#             [self.val_loader], #针对tiny-imagenet

#             label=f"aggregated_test_loader_0",
#         )

#         if not self.conf.train_fast:  # test all the selected_clients
#             for client_idx, flatten_local_model in flatten_local_models.items():
#                 _arch = self.clientid2arch[client_idx]
#                 _model_state_dict = copy.deepcopy(self.client_models[_arch].state_dict())
#                 flatten_local_model.unpack(_model_state_dict.values())
#                 _, test_model = create_model.define_model(self.conf, to_consistent_model=False, arch=_arch)
#                 test_model.load_state_dict(_model_state_dict)

#                 master_utils.do_validation(
#                     conf=self.conf,
#                     coordinator=self.local_coordinator[client_idx - 1],
#                     model=test_model,
#                     criterion=self.criterion,
#                     metrics=self.metrics,
#                     data_loaders=self.test_loaders,
#                     label=f"aggregated_test_loader_{client_idx}",
#                 )
#             self.additional_validation()


#         torch.cuda.empty_cache()
#     def additional_validation(self):
#         pass

#     def _check_early_stopping(self):
#         meet_flag = False

#         # consider both of target_perf and early_stopping
#         if self.conf.target_perf is not None:   # 是否设置目标性能
#             assert 100 >= self.conf.target_perf > 0

#             # meet the target perf.
#             if (
#                     self.coordinator.key_metric.cur_perf is not None
#                     and self.coordinator.key_metric.cur_perf > self.conf.target_perf
#             ):
#                 self.conf.logger.log("Master early stopping: meet target perf.")
#                 self.conf.meet_target = True
#                 meet_flag = True
#             # or has no progress and early stop it.
#             elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
#                 self.conf.logger.log(
#                     "Master early stopping: not meet target perf but has no patience."
#                 )
#                 meet_flag = True
#         # only consider the early stopping.
#         else:
#             if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
#                 meet_flag = True

#         if meet_flag:
#             # we perform the early-stopping check:
#             # (1) before the local training and (2) after the update of the comm_round.
#             _comm_round = self.conf.graph.comm_round - 1
#             self.conf.graph.comm_round = -1
#             self._finishing(_comm_round)

#     def _finishing(self, _comm_round=None):
#         self.conf.logger.save_json()
#         self.conf.logger.log(f"Master finished the federated learning.")
#         self.conf.is_finished = True
#         self.conf.finished_comm = _comm_round
#         checkpoint.save_arguments(self.conf)
#         os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


# def get_n_local_epoch(conf, n_participated):
#     if conf.min_local_epochs is None:
#         return [conf.local_n_epochs] * n_participated
#     else:
#         # here we only consider to (uniformly) randomly sample the local epochs.
#         assert conf.min_local_epochs > 1.0
#         random_local_n_epochs = conf.random_state.uniform(
#             low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
#         )
#         return random_local_n_epochs


# -*- coding: utf-8 -*-
import os
import copy

import numpy as np
import torch
import torch.distributed as dist
from pcode.utils.module_state import ModuleState
import pcode.master_utils as master_utils
import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.cross_entropy as cross_entropy
from pcode.utils.early_stopping import EarlyStoppingTracker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pcode.create_dataset as create_dataset # 确保引入了这个，用于加载 batch

class Master(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))

        # create model as well as their corresponding state_dicts.
        _, self.master_model = create_model.define_model( # 获取模型
            conf, to_consistent_model=False
        )
        self.used_client_archs = set( # set去重，打印所有客户端架构
            [
                create_model.determine_arch(conf, client_id, use_complex_arch=True)
                for client_id in range(1, 1 + conf.n_clients)
            ]
        )
        self.conf.used_client_archs = self.used_client_archs

        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")

        self.client_models = dict( # 获取客户端模型信息
            create_model.define_model(conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )

        self.clientid2arch = dict( # 获取所有客户端的模型名称
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch

        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data, agg_data_ratio=conf.agg_data_ratio)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        ) # 虽然划分了数据但是全丢了，只保留数据划分器



        conf.logger.log(f"Master initialized the local training data with workers.")

        # create val loader.
        # right now we just ignore the case of partitioned_by_user.
        if self.dataset["val"] is not None:
            assert not conf.partitioned_by_user
            self.val_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["val"], is_train=False
            )
            conf.logger.log(f"Master initialized val data.")
        else:
            self.val_loader = None

        # create test loaders.
        # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
        if conf.partitioned_by_user:
            self.test_loaders = []
            for localdata_id in self.client_ids:
                test_loader, _ = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=localdata_id - 1,
                    is_train=False,
                    shuffle=False,
                )
                self.test_loaders.append(copy.deepcopy(test_loader))
        else:
            test_loader, _ = create_dataset.define_data_loader(
                conf, self.dataset["test"], is_train=False
            )
            self.test_loaders = [test_loader] # 这里的test_loader和上面的val_loader都是总的，也就是说未平分给客户端的

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")   # 模型的损失函数，这里传入mean表示计算的损失会求平均
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")  # 模型的评估，计算准确率之类
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)   # 记录最佳性能
        if not self.conf.train_fast:
            self.local_coordinator = [create_coordinator.Coordinator(conf, self.metrics) for _ in
                                      range(self.conf.n_clients)]

        conf.logger.log(f"Master initialized the coordinator.\n")

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )   # 早停装置，为0时关掉早停功能

        # save arguments to disk.
        conf.is_finished = False    # 用于控制全局训练是否已经结束
        checkpoint.save_arguments(conf) # 将超参数保存在json文件中

    def run(self):

        # [新增 1] 确定起始轮次并尝试加载断点
        # ==========================================
        start_round = 1
        # 假设将断点保存在 checkpoint 目录下
        experiment_dir = os.path.dirname(self.conf.checkpoint_root)
        checkpoint_path = os.path.join(experiment_dir, "master_checkpoint.pt")
        # checkpoint_path = os.path.join(self.conf.checkpoint_root, "master_checkpoint.pt")
        
        if os.path.exists(checkpoint_path):
            self.conf.logger.log(f"发现断点文件 {checkpoint_path}，尝试恢复训练...")
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                start_round = ckpt['comm_round'] + 1
                self.master_model.load_state_dict(ckpt['master_model_state'])
                self.conf.logger.log(f"成功恢复！将从第 {start_round} 轮继续训练。")
                # ==========================================
                # [新增修复 1]：在下发给 Worker 炸毁前，先评估一次当前的真实实力
                # ==========================================
                self.conf.logger.log(">>> [断点检查] 正在评估恢复的初始全局模型...")
                master_utils.do_validation(
                    self.conf,
                    self.coordinator,
                    self.master_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label="Resume_Validation",
                )
                # ==========================================
            except Exception as e:
                self.conf.logger.log(f"加载断点失败: {e}，将从头开始训练。")


        for comm_round in range(start_round, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )   # 生成参与数长的列表，每个值为每个客户端的训练轮数
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.
            selected_client_ids = self._random_select_clients() # 随机选择客户端

            # detect early stopping.
            self._check_early_stopping() # 检查是否满足早停，可能未开启早停

            # init the activation tensor and broadcast to all clients (either start or stop).
            self.activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
            )

            # will decide to send the model or stop the training.
            if not self.conf.is_finished:
                self.send_extra_info_to_selected_clients(selected_client_ids)

                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids)

            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return

            self.receive_extra_info_from_selected_clients(selected_client_ids)

            # wait to receive the local models.
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            ) # 接受模型参数

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model_and_evaluate(flatten_local_models, selected_client_ids)   # 聚合模型，评估和存档

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

            if comm_round == self.conf.n_comm_rounds:
                # if comm_round == 5:
                self.visualize_global_features()
        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()
        self._finishing()

    def visualize_global_features(self):
        """
        在最后一轮提取全局模型在测试集上的特征，并使用 t-SNE 绘制特征分布散点图。
        """
        self.conf.logger.log("Extracting features for t-SNE visualization...")
        
        # 1. 准备模型
        self.master_model.eval()
        if self.conf.graph.on_cuda:
            self.master_model = self.master_model.cuda()

        all_features = []
        all_targets = []

        # 2. 遍历测试集提取特征
        with torch.no_grad():
            # 假设 self.test_loaders 是一个列表，合并所有 loader 的特征
            for data_loader in self.test_loaders:
                for _input, _target in data_loader:
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=False
                    )
                    
                    # 根据你 worker 的代码，模型的 forward 返回 feature 和 output
                    # 如果你的 master_model 返回格式不同，请在这里相应修改
                    feature, output = self.master_model(data_batch["input"])
                    
                    # 收集特征和标签
                    all_features.append(feature.cpu().numpy())
                    all_targets.append(data_batch["target"].cpu().numpy())

        # 3. 拼接所有 Batch 的数据
        all_features = np.concatenate(all_features, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        self.conf.logger.log(f"Extracted features shape: {all_features.shape}, running t-SNE...")

        # 4. 使用 t-SNE 降维到 2D
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        features_2d = tsne.fit_transform(all_features)

        # 5. 绘制散点图
        plt.figure(figsize=(10, 8))
        num_classes = len(np.unique(all_targets))
        
        sns.scatterplot(
            x=features_2d[:, 0], 
            y=features_2d[:, 1],
            hue=all_targets,
            palette=sns.color_palette("tab10", num_classes) if num_classes <= 10 else sns.color_palette("husl", num_classes),
            legend="full",
            alpha=0.7,
            s=20 # 点的大小
        )
        
        plt.title(f"t-SNE visualization of Global Model Features (Round {self.conf.graph.comm_round})")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        
        # 6. 保存图片到 checkpoint 目录
        save_path = os.path.join(self.conf.checkpoint_root, "global_model_tsne_final_round.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        self.conf.logger.log(f"t-SNE feature visualization saved to {save_path}")
        
        if self.conf.graph.on_cuda:
            self.master_model = self.master_model.cpu()


    def receive_extra_info_from_selected_clients(self, selected_client_ids):
        pass


    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()  # 随机选择客户端
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def activate_selected_clients(
            self, selected_client_ids, comm_round, list_of_local_n_epochs, to_send_history=False
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        msg_len = 3 # 消息行数
       

        activation_msg = torch.zeros((msg_len, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids) # 第一行装客户端id
        activation_msg[1, :] = comm_round                       # 第二行装联邦轮次
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)     # 第三行装每个客户端训练轮数

        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")

        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id] # 获取当前循环模型名称
            client_model_state_dict = self.client_models[arch].state_dict()

            flatten_model = TensorBuffer(list(client_model_state_dict.values())) # 打包模型参数
            
            # # 【关键修改】：确保发送的 buffer 在 GPU 上
            # device = torch.device(f"cuda:{self.conf.graph.rank % torch.cuda.device_count()}")
            # flatten_model.buffer = flatten_model.buffer.to(device)

            dist.send(tensor=flatten_model.buffer, dst=worker_rank) # 发送参数到对应模型
            self.conf.logger.log(
                f"\tMaster send the current model={arch} to process_id={worker_rank}."
            )

        dist.barrier()

    def send_extra_info_to_selected_clients(self, selected_client_ids):
        pass

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        # device = torch.device(f"cuda:{self.conf.graph.rank % torch.cuda.device_count()}")
        for selected_client_id in selected_client_ids: # 为接受数据准备容器，这也可以解释为什么要在服务器创建模型
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[arch].state_dict().values())
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )   # 这里并不会阻塞，相当于告诉客户端数据放在哪里
            reqs.append(req)

        for req in reqs:
            req.wait()  # 这里会阻塞等待数据

        dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models

    # def _fedavg(self, flatten_local_models, weights=None):
    #     n_selected_clients = len(flatten_local_models)

    #     if weights == None:
    #         weights = [
    #             torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
    #         ]

    #     # NOTE: the arch for different local models needs to be the same as the master model.
    #     # retrieve the local models.
    #     local_models = {}
    #     for client_idx, flatten_local_model in flatten_local_models.items():
    #         _arch = self.clientid2arch[client_idx]
    #         _model = copy.deepcopy(self.client_models[_arch])
    #         _model_state_dict = self.client_models[_arch].state_dict()
    #         flatten_local_model.unpack(_model_state_dict.values())
    #         _model.load_state_dict(_model_state_dict)
    #         local_models[client_idx] = _model

    #     # uniformly average the local models.
    #     # assume we use the runtime stat from the last model.
    #     _model = copy.deepcopy(_model)
    #     local_states = [
    #         ModuleState(copy.deepcopy(local_model.state_dict()))
    #         for _, local_model in local_models.items()
    #     ]
    #     model_state = local_states[0] * weights[0]
    #     for idx in range(1, len(local_states)):
    #         model_state += local_states[idx] * weights[idx]
    #     model_state.copy_to_module(_model)
    #     return _model
    def _fedavg(self, flatten_local_models, weights=None):
            n_selected_clients = len(flatten_local_models)
            if weights is None:
                weights = [1.0 / n_selected_clients for _ in range(n_selected_clients)]

            client_indices = list(flatten_local_models.keys())
            _arch = self.clientid2arch[client_indices[0]]
            _model = copy.deepcopy(self.client_models[_arch])
            
            # 1. 直接在 GPU 上初始化累加器
            device = flatten_local_models[client_indices[0]].buffer.device
            avg_buffer = torch.zeros_like(flatten_local_models[client_indices[0]].buffer, device=device)

            # 2. 纯 GPU 向量乘加运算 (极其快速，0 CPU 占用)
            for idx, client_idx in enumerate(client_indices):
                avg_buffer += flatten_local_models[client_idx].buffer * weights[idx]

            # 3. 算完之后，只把最终结果拉回 CPU 并解包更新到模型中
            _model_state_dict = _model.state_dict()
            avg_tb = TensorBuffer(list(_model_state_dict.values()))
            avg_tb.buffer = avg_buffer.cpu() # 唯一一次回传 CPU
            avg_tb.unpack(_model_state_dict.values())
            _model.load_state_dict(_model_state_dict)

            return _model

    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )

            fedavg_model = self._fedavg(flatten_local_models) # 平均算法
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def aggregate(self, flatten_local_models):
        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        return fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models, selected_client_ids):
        # aggregate the local models.
        client_models = self.aggregate(
            flatten_local_models
        ) # 计算平均后的模型

        self.master_model.load_state_dict(
            list(client_models.values())[0].state_dict()
        ) # 更新全局模型
        for arch, _client_model in client_models.items():
            self.client_models[arch].load_state_dict(_client_model.state_dict())

        # evaluate the aggregated model on the test data.
        master_utils.do_validation( # 最终评估
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.test_loaders,
            label=f"aggregated_test_loader_0",
        )

        if not self.conf.train_fast:  # test all the selected_clients
            for client_idx, flatten_local_model in flatten_local_models.items():
                _arch = self.clientid2arch[client_idx]
                _model_state_dict = copy.deepcopy(self.client_models[_arch].state_dict())
                flatten_local_model.unpack(_model_state_dict.values())
                _, test_model = create_model.define_model(self.conf, to_consistent_model=False, arch=_arch)
                test_model.load_state_dict(_model_state_dict)

                master_utils.do_validation(
                    conf=self.conf,
                    coordinator=self.local_coordinator[client_idx - 1],
                    model=test_model,
                    criterion=self.criterion,
                    metrics=self.metrics,
                    data_loaders=self.test_loaders,
                    label=f"aggregated_test_loader_{client_idx}",
                )
            self.additional_validation()


        torch.cuda.empty_cache()
    def additional_validation(self):
        pass

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:   # 是否设置目标性能
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
