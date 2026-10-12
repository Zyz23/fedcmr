# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import pcode.local_training.random_reinit as random_reinit
import pcode.datasets.mixup_data as mixup
import pcode.create_model as create_model
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.create_metrics as create_metrics
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat
from pcode.utils.timer import Timer
from pcode.utils.stat_tracker import RuntimeTracker
import  time
import copy
import clip
import os
import math

from pcode.utils.efficiency_profiler import (
    begin_efficiency_profile,
    finish_efficiency_profile,
)

class WorkerFedOur(object):
    def __init__(self, conf):
        
        self.conf = conf
        # some initializations.
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        self.device = torch.device("cuda" if self.conf.graph.on_cuda else "cpu")
        conf.device = self.device
        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=conf.logger.log_metric,
        )
        
        # create dataset (as well as the potential data_partitioner) for training.
        self.anchor_weight = torch.tensor([0, 0.50, 0.75, 1.0], device=self.device)
        self.text_model, _ = self.load_clip_text_model() # 加载文本编码器
        self.anchor = self.generate_text_anchors() #并生成锚点
        # Text encoder is only used for one-time anchor construction.
        # The local training phase uses cached self.anchor tensors.
        del self.text_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.output_dim = self.anchor[0].shape[-1] 
        
        print(f"Anchors generated. Count: {len(self.anchor)}, Dim: {self.output_dim}")
        conf.output_dim = self.output_dim 
        self.conf = conf

        
        print("anchor:{self.anchor}")

        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data, agg_data_ratio=conf.agg_data_ratio)
        _, self.data_partitioner = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
        )

        conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} initialized the local training data with Master."
        )

        # define the criterion.
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} initialized dataset/criterion.\n"
        )   # 打印当前的进程编号


    def load_clip_text_model(self, model_name="ViT-B/32", save_dir="./model_state/"):
        """
        仅加载 CLIP 的文本提取部分。
        
        1. 自动检查本地/下载 (逻辑不变)。
        2. 使用 jit=False 加载，以便我们可以修改模型结构。
        3. 删除 visual 部分以节省显存。
        """
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Loading CLIP Text Backbone: {model_name}...")
        
        # [关键修改 1] jit=False: 允许我们将模型作为普通的 PyTorch nn.Module 加载，
        # 这样我们才能执行 del model.visual 操作。
        # model 的 preprocess (针对图像) 我们直接忽略，用不到。
        model, _ = clip.load(model_name, device=device, download_root=save_dir, jit=False)
        # [关键修改 2] 删除视觉编码器部分
        # 这会释放掉 ViT 或 ResNet 部分占用的显存
        if hasattr(model, 'visual'):
            # 1. 先保存当前的数据类型 (通常是 float16 或 float32)
            # 注意：如果 visual 已经被删了，这里会报错，所以要小心
            try:
                stored_dtype = model.visual.conv1.weight.dtype
            except:
                stored_dtype = torch.float16 # 默认备选
                
            # 2. 删除真正的视觉编码器 (释放显存)
            del model.visual
            if device == "cuda":
                torch.cuda.empty_cache()
                
            # 3. [核心] 塞入一个假的 visual，只为了让 model.dtype 属性不报错
            model.visual = DummyVisual(stored_dtype)
                
        print("Text-only model loaded. Visual encoder removed.")
        
        # 将模型设为评估模式 (对于文本部分这通常不影响，但好习惯)
        model.eval()
        model.to("cuda")

        return model, device
    

    def generate_text_anchors(self):
        """
        根据数据集名称生成文本锚点 (Text Anchors)。
        支持 CIFAR 和动态加载的 Tiny-ImageNet。
        """
        device = next(self.text_model.parameters()).device
        dataset_name = self.conf.data.lower() # 确保大小写匹配
        
        # === 1. 获取类别名称列表 ===
        classes = []
        
        if "tiny" in dataset_name:
            # ---------------------------------------------------------
            # 针对 Tiny-ImageNet 动态计算锚点 (防错位核心逻辑)
            # ---------------------------------------------------------
            # 假设 conf.data_dir 是 './data'，conf.data 是 'imagenet_tiny'
            tiny_root = os.path.join(self.conf.data_dir, self.conf.data) 
            words_path = os.path.join(tiny_root, "words.txt")
            train_dir = os.path.join(tiny_root, "train")
            
            if not os.path.exists(words_path) or not os.path.exists(train_dir):
                raise FileNotFoundError(f"找不到 Tiny-ImageNet 的 words.txt 或 train 目录，请检查路径: {tiny_root}")
                
            # A. 解析 words.txt 映射字典 (只取逗号前的核心名词，提升 CLIP 识别率)
            # 例如: "n01443537 \t goldfish, Carassius auratus" -> {"n01443537": "goldfish"}
            wnid_to_word = {}
            with open(words_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        wnid_to_word[parts[0]] = parts[1].split(',')[0].strip()
                        
            # B. 严格按照字母排序读取 train 目录下的文件夹名 (底层对齐 ImageFolder 的 0-199 索引)
            wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            
            # C. 映射出最终的 200 个英文标签
            classes = [wnid_to_word.get(wnid, "object") for wnid in wnids]
            print(f"✅ 成功提取 Tiny-ImageNet 类别，共 {len(classes)} 类。前5类为: {classes[:5]}")
            
        elif dataset_name == "cifar10":
            classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        elif dataset_name == "cifar100":
            classes = [
                'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]
        else:
            raise ValueError(f"Dataset '{dataset_name}' 不在支持的锚点生成列表中。")


        # === 2. 构造提示词 (Prompt Template) ===
        # CLIP 官方推荐模板
        prompts = [f"a photo of a {c}" for c in classes]
        
        # === 3. Tokenize ===
        text_inputs = clip.tokenize(prompts).to(device)
        
        # === 4. 提取 CLIP 文本特征并截取多层锚点 ===
        self.text_model.eval()
        with torch.no_grad():
            x = self.text_model.token_embedding(text_inputs).type(self.text_model.dtype)
            x = x + self.text_model.positional_embedding.type(self.text_model.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            layers = self.text_model.transformer.resblocks
            total_layers = len(layers)
            
            # 定义要截取的层索引
            indices = [
                int(total_layers * 1 / 4) - 1,
                int(total_layers * 2 / 4) - 1,
                int(total_layers * 3 / 4) - 1,
                total_layers - 1
            ]
            
            anchors_list = []
            
            for i, layer in enumerate(layers):
                x = layer(x)
                
                if i in indices:
                    x_temp = x.permute(1, 0, 2)
                    # 提取 [EOS] token
                    sent_emb = x_temp[torch.arange(x_temp.shape[0]), text_inputs.argmax(dim=-1)]
                    
                    if i == total_layers - 1:
                        sent_emb = self.text_model.ln_final(sent_emb).type(self.text_model.dtype)
                        sent_emb = sent_emb @ self.text_model.text_projection
                    
                    # 保留原始文本表征
                    raw_anchor= sent_emb.float()
                    # L2 归一化 (核心步骤)
                    norm_anchor = F.normalize(raw_anchor, p=2, dim=-1)

                    anchors_list.append(norm_anchor)

            return anchors_list # 返回 4 个层级的锚点 Tensor 列表
        

    def recv_extra_info_from_master(self):
        pass

    def send_extra_info_to_master(self):
        pass

    def get_params(self, model):
        weight_collector = None
        for param in model.parameters():
            if not isinstance(weight_collector, torch.Tensor):
                weight_collector = param.reshape(-1)
            else:
                weight_collector = torch.cat((weight_collector, param.reshape(-1)), 0)
        return weight_collector

    def run(self):
        while True:
            self.listen_to_master() # 每次可能分配不同的客户端

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self.recv_extra_info_from_master()

            self._recv_model_from_master() # 接受模型参数
            self._train()

            self.send_extra_info_to_master()

            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return
    def extra_init(self):
        # pass

        # 将第4层（最后一层）的文本锚点传给模型
            # self.model.set_text_anchors(self.anchor[-1].clone().detach().to(self.device))
        """
    原型锚点模型需要将文本锚点写入模型，
    传统分类头模型不需要该步骤。
    """
        if hasattr(self.model, "set_text_anchors"):
            self.model.set_text_anchors(
                self.anchor[-1].clone().detach().to(self.device)
            )    

    def listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((3, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)

        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs = (
            msg[:3, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        ) # 取出属于自己的消息

        # Master 传来的 client_id 是 0-based: 0 ~ n_clients-1
        client_id0 = int(self.conf.graph.client_id)

        # create_model 里 client_id 是按 1-based 设计的，所以这里转回 1-based
        client_id_for_model = client_id0 + 1
        
        if self.conf.rank_list is not None:
            # self.ratio_LR = self.conf.rank_list[self.conf.graph.client_id - 1]  # 取出属于自己的秩
            self.ratio_LR = self.conf.rank_list[client_id0]
            self.conf.ratio_LR = self.ratio_LR


        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=client_id_for_model
        ) # 获取模型

        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values())) # 将参数打包为一维参数，方便传输

        self.conf.logger.log(
        f"[WORKER BUFFER] worker_rank={self.conf.graph.rank}, "
        f"client_id0={client_id0}, "
        f"client_id_for_model={client_id_for_model}, "
        f"rank={self.ratio_LR}, "
        f"numel={self.model_tb.buffer.numel()}, "
        f"bytes={self.model_tb.buffer.numel() * self.model_tb.buffer.element_size()}"
        )
        self.extra_init()

        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()

        self.train_loader, _ = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset["train"],
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            localdata_id=client_id0,
            is_train=True,
            data_partitioner=self.data_partitioner,
        )   # 这里的data_partitioner已经统一,这里传入的参时真正的客户端id

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values()) # 将接受的参数填入模型字典

        # if not self.conf.freeze_bn:
        self.model.load_state_dict(self.model_state_dict)   # 更新模型参数
        random_reinit.random_reinit_model(self.conf, self.model)    # 若为true，这里可能会随机丢弃参数
        self.init_model = self._turn_off_grad(copy.deepcopy(self.model).to(self.device)) # 获得一个关闭了梯度的一摸一样的模型，常用于蒸馏

        # self.aggregation = Aggregation(self.model.classifier.in_features).cuda()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the model ({self.arch}) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )

        dist.barrier()

    def _prepare_train(self):
        self.model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)
            # self.text_model = self.text_model.to(self.device)

        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )   # 建立优化器，同时加入L2正则
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)

        # # ==========================================
        # # [新增修复 2]：防止学习率重置导致模型炸毁！
        # # ==========================================
        # current_round = self.conf.graph.comm_round
        
        # # 获取第一次衰减的节点 (你的配置里是 150)
        # decay_epoch = getattr(self.conf, "lr_warmup_epochs_upper_bound", 150)
        
        # # 如果当前轮次已经超过了衰减节点，强制衰减学习率
        # if current_round >= decay_epoch:
        #     # 计算衰减次数 (例如 164 // 150 = 1)
        #     decay_times = current_round // decay_epoch
        #     actual_lr = self.conf.lr * (self.conf.lr_decay ** decay_times)
            
        #     # 强行覆盖优化器中的学习率
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = actual_lr
                
        #     self.conf.logger.log(f"Worker 保护机制激活：检测到当前为第 {current_round} 轮，已将学习率强行衰减为 {actual_lr}")
        # # ==========================================

        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names) # 创建追踪器，记录损失和准确率
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # efficient local training.
        if hasattr(self, "model_compression_fn"):
            self.model_compression_fn.compress_model(
                param_groups=self.optimizer.param_groups
            )

    def prepare_train(self):
        self._prepare_train()

    def local_training_with_extra_calculate(self, loss, output, data_batch, feature=None, target=None,global_feature=None):
        # 1. 基础损失
        # total_loss = loss + self.conf.meta_L2 * self.model.get_decomposed_loss()
        # total_loss = 2.0*loss + self.conf.meta_L2 * self.model.L2_decay()
        total_loss = 2.0*loss + self.conf.meta_L2 * self.model.L2_decay()

        # l2_loss = self.conf.meta_L2 * self.model.L2_decay()

        # 2. 计算层级锚点损失
        # 检查是否开启对齐 (可以通过 output 里的 stage_features 是否为空，或者 check use_align)
        # 注意：这里需要确保你已经正确解包了 output，例如 x, logits = output
        # 或者你现在的 output 就是 logits，而 stage_features 存在 self.model.stage_features 里
        if "resnet" in self.conf.arch:
            current_epoch = self.conf.graph.comm_round
            max_epochs = self.conf.n_comm_rounds
            # 获取联邦轮次信息用于动态权重
            
            # 余弦退火动态权重 (5.0 -> 0.5)
            dynamic_weight = 0.5 + 8.5 * (1 + math.cos(math.pi * current_epoch / max_epochs)) / 2
            if self.model.use_align and hasattr(self, 'anchor'):

                """计算MSE 四层对齐损失 权重([0.125, 0.25, 0.5, 1.0],  """
                
                # print("四级对齐") 
                # [关键步骤 A] 获取当前 Batch 的标签
                if "target" in data_batch:
                    current_target = data_batch["target"]
                elif target is not None:
                    current_target = target.to(self.device)
                else:
                    raise ValueError("local_training_with_extra_calculate") # 没标签没法算，直接返回

           
                current_batch_anchors = []
                for layer_idx in range(4):
                    # 利用高级索引，选出当前 target 对应的锚点行
                    # layer_anchor shape: [Batch_Size, Dim]
                    layer_anchor = self.anchor[layer_idx][current_target]
                    current_batch_anchors.append(layer_anchor)

                # [关键步骤 C] 传入处理好的 Batch 锚点
                # 返回的是 shape 为 [4] 的 tensor，包含 4 个阶段的 loss
                loss_vec = self.model.calculate_stage_anchor_loss(current_batch_anchors)
                
                # [关键步骤 D] 求和并加权
                # 因为 loss_vec 是 [L1, L2, L3, L4]，你需要把它变成一个标量才能加到 total_loss
                weight_tensor = torch.tensor(self.anchor_weight, device=loss_vec.device, dtype=loss_vec.dtype)
                anchor_loss_scalar = torch.sum(loss_vec * weight_tensor)
                
                total_loss += self.conf.anchor_loss * anchor_loss_scalar
                # total_loss += dynamic_weight * anchor_loss_scalar


            

            if feature is not None:
                # ------------------------------------------------------
                # [模块 0]: 全局特征蒸馏 (FedGKD 思想)
                # ------------------------------------------------------
                if current_epoch > 20 and global_feature is not None:
                    # 计算本地特征与全局特征的余弦相似度
                    cosine_sim = torch.sum(feature * global_feature, dim=1)
                    loss_global_distill = (1.0 - cosine_sim).mean()
                    total_loss += 0 * loss_global_distill
                
                # ------------------------------------------------------
                # [模块 A]: 语义感知蒸馏 (辅助勾勒特征空间)
                # ------------------------------------------------------
                # weight_idea_A = 3
                weight_idea_A = 0
                if weight_idea_A > 0:
                    tau_sem = 0.1  
                    tau_model = 0.15 
                    
                    # 懒加载：只计算一次全局锚点的语义分布矩阵
                    if getattr(self, "semantic_target_probs", None) is None:
                        anchors = self.anchor[-1].to(self.device)
                        self.cached_anchors = anchors
                        semantic_matrix = torch.matmul(anchors, anchors.T)
                        self.semantic_target_probs = F.softmax(semantic_matrix / tau_sem, dim=1).detach()
                    
                    # 取出当前 target 的绝对语义概率分布
                    target_probs = self.semantic_target_probs[current_target]
                    # 计算当前特征在锚点空间的预测分布
                    model_logits = torch.matmul(feature, self.cached_anchors.T) / tau_model
                    model_log_probs = F.log_softmax(model_logits, dim=1)
                    
                    loss_semantic_kd = F.kl_div(model_log_probs, target_probs, reduction='batchmean')
                    total_loss += weight_idea_A * loss_semantic_kd

                # ------------------------------------------------------
                # [模块 B]: 特征级幻觉 Mixup (防数据缺失坍缩)
                # ------------------------------------------------------
                # weight_idea_B = 0.3
                weight_idea_B = 0.0
                if weight_idea_B > 0:
                    batch_size = feature.size(0)
                    indices = torch.randperm(batch_size).to(self.device)
                    feature_shuffled = feature[indices]
                    target_shuffled = current_target[indices]

                    # 确保 cached_anchors 存在
                    if getattr(self, "cached_anchors", None) is None:
                        self.cached_anchors = self.anchor[-1].to(self.device)

                    # Beta 分布采样插值系数
                    lam = np.random.beta(0.2, 0.2)
                    
                    # 混合特征 & 混合锚点
                    virtual_features = F.normalize(lam * feature + (1 - lam) * feature_shuffled, p=2, dim=1)
                    virtual_anchors = F.normalize(lam * self.cached_anchors[current_target] + (1 - lam) * self.cached_anchors[target_shuffled], p=2, dim=1)
                    
                    # 让混合后的虚拟特征对齐到混合后的虚拟锚点
                    loss_virtual_mixup = (1.0 - torch.sum(virtual_features * virtual_anchors, dim=1)).mean()
                    total_loss += weight_idea_B * loss_virtual_mixup


        elif "cnn" in self.conf.arch:
            if "target" in data_batch:
                current_target = data_batch["target"]
            elif target is not None:
                current_target = target.to(self.device)
            else:
                raise ValueError("local_training_with_extra_calculate")
            sleleted_anchor = self.anchor[-1][current_target]

            current_epoch = self.conf.graph.comm_round
            max_epochs = self.conf.n_comm_rounds

            

            # # ==========================================================
            # # 1. MSE 稳定拉力 
            # # ==========================================================
            # print("计算MSE拉力")
            dynamic_weight = 0.5 + 8.5 * (1 + math.cos(math.pi * current_epoch / max_epochs)) / 2
            # 针对tiny数据集 正向增长
            # dynamic_weight = 0.5 + 8.5 * (1 - math.cos(math.pi * current_epoch / max_epochs)) / 2
            
            mse_loss = F.mse_loss(feature, sleleted_anchor)
            
            cosine_loss = (
                1.0 - torch.sum(feature * sleleted_anchor, dim=1)
            ).mean()
            # anchor_term = self.conf.anchor_loss * anchor_align_loss


    #         self.conf.logger.log(
    #     f"[RAW-MSE CHECK] "
    # f"ce={loss.item():.6f}, "
    # f"raw_mse={anchor_align_loss.item():.8f}, "
    # f"lambda={self.conf.anchor_loss:.4f}, "
    # f"mse_term={anchor_term.item():.6f}, "
    # f"||h||={raw_feature_norm.item():.6f}, "
    # f"||a_raw||={raw_anchor_norm.item():.6f},"
    # f"||a_||={anchor_norm.item():.6f},"
    # )

            total_loss = total_loss + dynamic_weight * mse_loss #传入的基础 loss + l2_loss + MSE拉力

            if current_epoch > 20:
                # # 计算本地特征与全局模型提取的特征的余弦相似度 (逐样本计算)
                cosine_sim = torch.sum(feature * global_feature, dim=1)
                # 目标是最大化相似度，所以 loss 为 (1 - similarity)
                loss_global_distill = (1.0 - cosine_sim).mean()
                 # # 4. 汇总 Total Loss
                total_loss = total_loss + 0 * loss_global_distill 
            mu = 5.0 # 蒸馏权重

            
           
            '''
            语义感知蒸馏 和虚假特征排斥
            '''

            # ==========================================================
            # [实验模块 A]: 语义感知蒸馏 (辅助勾勒特征空间)
            # ==========================================================
            # 【实验开关】：先设为 0，确认基本盘正常后，再设为 0.3 或 0.5
            weight_idea_A = 1.5
            
            if weight_idea_A > 0:
                # tau_sem = 0.05
                # tau_model = 0.07
                tau_sem = 0.1  # 从 0.05 提高到 0.1
                tau_model = 0.15 # 从 0.07 提高到 0.15
                # 懒加载语义矩阵
                if getattr(self, "semantic_target_probs", None) is None:
                    anchors = self.anchor[-1].to(self.device)
                    self.cached_anchors = anchors
                    semantic_matrix = torch.matmul(anchors, anchors.T)
                    self.semantic_target_probs = F.softmax(semantic_matrix / tau_sem, dim=1).detach()
                
                target_probs = self.semantic_target_probs[current_target]
                model_logits = torch.matmul(feature, self.cached_anchors.T) / tau_model
                model_log_probs = F.log_softmax(model_logits, dim=1)
                
                loss_semantic_kd = F.kl_div(model_log_probs, target_probs, reduction='batchmean')
                total_loss += weight_idea_A * loss_semantic_kd

            # ==========================================================
            # [实验模块 B]: 特征级幻觉 Mixup (防数据缺失坍缩)
            # ==========================================================
            # 【实验开关】：必须等 Idea A 跑出效果后，再尝试设为 0.1 (千万不要太大)
            weight_idea_B = 0.3
            
            if weight_idea_B > 0:
                batch_size = feature.size(0)
                indices = torch.randperm(batch_size).to(self.device)
                feature_shuffled = feature[indices]
                target_shuffled = current_target[indices]

                # 确保 cached_anchors 存在
                if not hasattr(self, "cached_anchors"):
                    self.cached_anchors = self.anchor[-1].to(self.device)

                lam = np.random.beta(0.2, 0.2)
                virtual_features = F.normalize(lam * feature + (1 - lam) * feature_shuffled, p=2, dim=1)
                virtual_anchors = F.normalize(lam * self.cached_anchors[current_target] + (1 - lam) * self.cached_anchors[target_shuffled], p=2, dim=1)
                
                loss_virtual_mixup = (1.0 - torch.sum(virtual_features * virtual_anchors, dim=1)).mean()
                total_loss += weight_idea_B * loss_virtual_mixup
        #     self.conf.logger.log(
        #     f"[RAW-MSE CHECK] "
        # f"ce={loss.item():.6f}, "
        # f"raw_mse={mse_loss.item():.8f}, "
        # # f"lambda={loss_global_distill.item():.4f}, "
        # f"mse_term={loss_virtual_mixup.item():.6f}, "
        # f"||h||={loss_semantic_kd.item():.6f}, "
        # # f"||a_raw||={raw_anchor_norm.item():.6f},"
        # # f"||a_||={anchor_norm.item():.6f},"
        # )

         
        return total_loss

    def add_grad(self):
        pass

    def _train(self):
        self.prepare_train() # 初始化指标追踪器，优化器等
        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:

            for _input, _target in self.train_loader:

                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_): # 数据统计
                    data_batch = create_dataset.load_data_batch(    # 这里主要是将数据搬到cuda上
                        self.conf, _input, _target, is_training=True, device=self.device
                    )
                    begin_efficiency_profile(self)

                # inference and get current performance.
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    loss, performance, output, feature  = self._inference(data_batch)    # 前向传播

                # 同时效仿moon 提取全局刚下发的特征 也就是全局特征 
                # # ==========================================
                # 【新增】：利用初始全局模型提取 global_feature
                # ==========================================
                with torch.no_grad():
                    # self.init_model 是从 Master 接收并冻结了梯度的全局模型
                    global_feature, _ = self.init_model(data_batch["input"])  

                    


                                       
                with self.timer("extra_forward_pass", epoch=self.scheduler.epoch_):
                    loss = self.local_training_with_extra_calculate(loss, output, data_batch, feature=feature,target = _target,global_feature=global_feature)  # 计算L2损失
                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10) # 梯度裁剪，max_norm为允许的最大梯度

                    self.optimizer.step()
                    finish_efficiency_profile(self, data_batch)
                    self.scheduler.step()

                # check divergence.检查loss是否过大，防止将有毒参数上传
                if loss > 1e4 or torch.isnan(loss):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )   # 报告哪个客户端出错

                    self._terminate_comm_round()    # 终止训练
                    return

                # check stopping condition.
                if self._is_finished_one_comm_round(): # 检查轮数，如果够20轮直接返回

                    self.conf.logger.log(self.timer.summary())

                    self._terminate_comm_round()
                    return

            # # display tracking time.
            # if (
            #         self.conf.display_tracked_time
            #         and self.scheduler.local_index % self.conf.summary_freq == 0
            # ):


            # display the logging info.
            display_training_stat(self.conf, self.scheduler, self.tracker) # 打印这一轮的成功
            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        feature,output = self.model(data_batch["input"]) 

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            bsz = data_batch["target"].size(0)
            self.tracker.update_local_metrics(
                loss.item(), 0, n_samples=bsz
            )
            for idx in range(1, 1 + len(performance)):
                self.tracker.update_local_metrics(
                    performance[idx - 1], idx, n_samples=bsz
                )
        return loss, performance, output, feature

    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        return model

    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) sending the model ({self.arch}) back to Master."
        )
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))    # 打包数据
        dist.send(tensor=flatten_model.buffer, dst=0)   # 发送
        dist.barrier()      # 等待其他客户端

    def _terminate_comm_round(self):
        self.model = self.model.cpu()

        self.scheduler.clean()
        self.conf.logger.save_json()
        torch.cuda.empty_cache()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

    def _terminate_by_early_stopping(self):
        if self.conf.graph.comm_round == -1:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning by early-stopping."
            )
            return True
        else:
            return False

    def _terminate_by_complete_training(self):
        if self.conf.graph.comm_round == self.conf.n_comm_rounds:
            dist.barrier()
            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} finished the federated learning: (total comm_rounds={self.conf.graph.comm_round})."
            )
            return True
        else:
            return False

    def _is_finished_one_comm_round(self):
        return True if self.conf.epoch_ >= self.conf.local_n_epochs else False


class DummyLayer:
    def __init__(self, dtype):
        self.weight = type('DummyTensor', (), {'dtype': dtype})()

class DummyVisual:
    def __init__(self, dtype):
        self.conv1 = DummyLayer(dtype)