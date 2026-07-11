# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torchvision
import copy
import numpy as np
import string

__all__ = ["simple_cnn","MetaCNN"]


def _decide_num_classes(dataset):
    if dataset == "cifar10" or dataset == "svhn":
        return 10
    elif dataset == "cifar100":
        return 100
    elif "tiny" in dataset:
        return 200
    elif "imagenet" in dataset:
        return 1000
    elif "mnist" == dataset:
        return 10
    elif "femnist" == dataset:
        return 62
    else:
        raise NotImplementedError(f"this dataset ({dataset}) is not supported yet.")


class CNNMnist(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True):
        super(CNNMnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=w_conv_bias)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=w_conv_bias)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50, bias=w_fc_bias)
        self.classifier = nn.Linear(50, self.num_classes, bias=w_fc_bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.classifier(x)
        return x


class CNNfemnist(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNfemnist, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        # define layers.
        self.conv1 = nn.Conv2d(1, 32, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(1024, 2048, bias=w_fc_bias)
        self.classifier = nn.Linear(2048, self.num_classes, bias=w_fc_bias)

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        activation1 = self.conv1(x)
        x = self.pool(F.relu(activation1))

        activation2 = self.conv2(x)

        x = self.pool(F.relu(activation2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2]
        return x


class CNNCifar(nn.Module):
    def __init__(
        self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True
    ):
        super(CNNCifar, self).__init__()

        # decide the num of classes.
        self.num_classes = _decide_num_classes(dataset)

        # define layers.
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 64)
        self.classifier = nn.Linear(64, self.num_classes, bias=w_fc_bias)

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)

        x = self.pool(F.relu(x))
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, 1024)
        feat = F.relu(self.fc1(x))
        x = self.classifier(feat)

        return feat, x
    
# 新增5层卷积

class CNNCifar5Layer(nn.Module):
    def __init__(self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True):
        super(CNNCifar5Layer, self).__init__()

        # 确定类别数
        self.num_classes = _decide_num_classes(dataset)
        
        # 注意：如果使用 BatchNorm，卷积层的 bias 通常设为 False，因为 BN 自带 bias shift
        # 但为了兼容您的 conf配置，这里保留逻辑，但在下面 layer 定义时强制 bias=False (推荐)
        use_bias = False 

        # === 5层卷积结构 (VGG Style Block: Conv-BN-ReLU) ===
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=0, stride=1, bias=True)
        # self.bn1 = nn.GroupNorm(32, 32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        # self.bn2 = nn.GroupNorm(64, 64)
        
        # # Layer 3: 64 -> 128
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=use_bias)
        # self.bn3 = nn.GroupNorm(128, 128)
        
        # # Layer 4: 128 -> 256
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=use_bias)
        # self.bn4 = nn.GroupNorm(256, 256)
        
        # # Layer 5: 256 -> 256 (保持通道数，提取高层语义)
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias)
        # self.bn5 = nn.GroupNorm(256, 256)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
        # Dropout (防止过拟合)
        # self.dropout = nn.Dropout(p=0.5)

        # 计算全连接层输入维度
        # 32x32 -> (Pool) -> 16x16 -> (Pool) -> 8x8 -> (Pool) -> 4x4
        # 最终特征图大小: 256 channels * 4 * 4 = 4096
        self.fc1 = nn.Linear(1600, 512, bias=w_fc_bias)

        self.classifier = nn.Linear(512, self.num_classes, bias=w_fc_bias)

        self.save_activations = save_activations
        self.activations = None

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )
        # 分类头
        self.head =self.classifier


        


    def forward(self, x,start_layer_idx=0):
        if start_layer_idx < 0:
            
            logits = self.head(x)
            return x, logits
        # Block 1
        # x = self.conv1(x)
        # x = F.relu(self.bn1(x))
        # x = self.pool(x)        # 32 -> 16
        
        # # Block 2
        # x = self.conv2(x)
        # x = F.relu(self.bn2(x)) 
        # # 这里不加池化，保持特征更丰富，或者隔层池化
        
        # # Block 3
        # x = self.conv3(x)
        # x = F.relu(self.bn3(x))
        # x = self.pool(x)        # 16 -> 8
        
        # # Block 4
        # x = self.conv4(x)
        # x = F.relu(self.bn4(x))
        # # 不加池化
        
        # # Block 5
        # x = self.conv5(x)
        # x = F.relu(self.bn5(x))
        # x = self.pool(x)        # 8 -> 4

        # # Flatten
        # x = x.view(-1, 256 * 4 * 4)
        
        # # FC Block
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)     # 关键：训练时随机丢弃神经元
        
        # logits = self.classifier(x)
        x = self.base(x)
        logits = self.head(x)

        return x, logits # 返回 feat, logits 保持与之前接口一致

   # 定义分解的cnn网络
class MetaCNN(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True, rank_rate=1.0):
        super(MetaCNN, self).__init__()
        
        self.num_classes = _decide_num_classes(dataset)
        self.rank_rate = rank_rate # 秩率
        self.w_conv_bias = w_conv_bias # 记录配置以便恢复时使用
        self.w_fc_bias = w_fc_bias

        # BN层通常不需要卷积偏置
        use_conv_bias = False 

        # === 前4层卷积 (保持固定，不进行分解) ===
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=0, stride=1, bias=True)
   
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        

        # 池化与Dropout
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        # # 计算Flatten后的维度: 256 * 4 * 4 = 4096
        # self.flat_dim = 256 * 4 * 4
        
        # === FC1 层 (可分解目标) ===
        # 输入: 4096, 输出: 512
        if rank_rate >= 1.0:
            self.fc1 = nn.Linear(1600, 512)
        else:
            self.fc1 = FactorizedLinear(1600, out_features=512, 
                                        rank_rate=rank_rate, bias=w_fc_bias)

        # === Classifier (保持固定) ===
        # self.classifier = nn.Linear(512, self.num_classes)

         # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )
        # 分类头
        # self.head =self.classifier

        # [ADD] 新增全局共享、可训练的 CLIP 适配层
        # 输入是 fc1 输出的 512 维特征，输出是对齐 CLIP 的 512 维特征
        self.clip_adapter = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512)
                )
    def set_text_anchors(self, anchors):
        """设置文本锚点特征 (CLIP 生成的文本特征)"""
        anchors = F.normalize(anchors,p=2, dim=-1) # L2 归一化

        self.register_buffer('text_anchors', anchors, persistent=False) # 不更新也不打包传播

    def forward(self, x):
      
        x = self.base(x)
        # logits = self.head(x)

        # 基于原型推理
        aligned_feature = self.clip_adapter(x)
        aligned_feature = F.normalize(aligned_feature, p=2, dim=-1)

        if hasattr(self, 'text_anchors') and self.text_anchors is not None:
            # 计算与文本锚点的相似度 (点积) 并应用温度缩放
            # logits = self.logit_scale.exp() * aligned_feature @ self.text_anchors.T
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            logits = logit_scale * aligned_feature @ self.text_anchors.T
        else:
            print("Warning: Text anchors not set. Returning unscaled features as logits.")


        return aligned_feature, logits

    # === 模型分解与恢复方法 ===

    def decom_model(self, rank_rate):
        """将 fc1 分解为低秩层"""
        if rank_rate >= 1.0:
            return
        
        print(f"Executing Decomposition (rank_rate={rank_rate})...")
        
        # 分解 Conv5
        # if isinstance(self.conv5, nn.Conv2d):
        #     # 调用你提供的 Decom_COV
        #     self.conv5 = Decom_COV(self.conv5, rank_rate)
        #     print(" -> conv5 decomposed.")

        # 分解 FC1
        if isinstance(self.fc1, nn.Linear):
            # 调用你提供的 Decom_LINEAR
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
            print(" -> fc1 decomposed.")
        
        self._rebuid_base() # 分解完成后重构基础组件
            
        self.rank_rate = rank_rate

    def _rebuid_base(self):
        """重建基础层的 Sequential 模块 (在分解或恢复后调用)"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

    def recover_model(self):
        """将 conv5 和 fc1 恢复为完整层"""
        if self.rank_rate >= 1.0:
            return

        print("Executing Recovery to Full Rank...")

        # 恢复 Conv5
        # if isinstance(self.conv5, FactorizedConv):
        #     # 调用你提供的 Recover_COV
        #     self.conv5 = Recover_COV(self.conv5)
        #     print(" -> conv5 recovered.")

        # 恢复 FC1
        if isinstance(self.fc1, FactorizedLinear):
            # 调用你提供的 Recover_LINEAR
            self.fc1 = Recover_LINEAR(self.fc1)
            print(" -> fc1 recovered.")
        self._rebuid_base()
        self.rank_rate = 1.0

    # === 正则化 Loss 计算 (用于低秩训练阶段) ===
    
    def get_decomposed_loss(self, type='frobenius'):
        """计算分解层的正则化损失"""
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        
        if self.rank_rate >= 1.0:
            return loss

        # 获取 Conv5 的 loss
        # if isinstance(self.conv5, FactorizedConv):
        #     if type == 'frobenius': loss += self.conv5.frobenius_loss()
        #     elif type == 'kronecker': loss += self.conv5.kronecker_loss()
        #     elif type == 'l2': loss += self.conv5.L2_loss()

        # 获取 FC1 的 loss
        if isinstance(self.fc1, FactorizedLinear):
            if type == 'frobenius': loss += self.fc1.frobenius_loss()
            elif type == 'kronecker': loss += self.fc1.kronecker_loss()
            elif type == 'l2': loss += self.fc1.L2_loss()
            
        return loss

class MetaCNN3(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True, rank_rate=1.0):
        super(MetaCNN3, self).__init__()
        
        self.num_classes = _decide_num_classes(dataset)
        self.rank_rate = rank_rate # 秩率
        self.w_conv_bias = w_conv_bias # 记录配置以便恢复时使用
        self.w_fc_bias = w_fc_bias

        # BN层通常不需要卷积偏置
        use_conv_bias = False 

        # === 前4层卷积 (保持固定，不进行分解) ===
        # 注意：这里改用了 kernel_size=3, padding=1，这在4层CNN中更常见，
        # 能避免特征图缩小过快，从而严格对齐你注释中的 256 * 4 * 4 = 4096 维度 (假设输入为32x32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True)

        # 池化与激活
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
        # 计算Flatten后的维度: 假设经过3次池化 (32->16->8->4)，256 * 4 * 4 = 4096
        # 如果你的输入图片尺寸不是 32x32，这里需要相应调整
        self.flat_dim = 4096
        
        # === FC1 层 (可分解目标) ===
        # 输入: 4096 (基于上方计算), 输出: 512
        if rank_rate >= 1.0:
            self.fc1 = nn.Linear(self.flat_dim, 512)
        else:
            self.fc1 = FactorizedLinear(self.flat_dim, out_features=512, 
                                        rank_rate=rank_rate, bias=w_fc_bias)

        # === Classifier (保持固定) ===
        # self.classifier = nn.Linear(512, self.num_classes)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 构建共享基础部分 (加入 conv3 和 conv4 及其对应的池化/激活)
        self.base = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu, self.pool,
            self.conv4, self.relu, # 这里假设第4层后不接池化，或者可以根据需要补上 self.pool
            self.flatten,
            self.fc1,
            self.relu
        )
        
        # 分类头
        # self.head =self.classifier

        # [ADD] 新增全局共享、可训练的 CLIP 适配层
        self.clip_adapter = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                # nn.Linear(512, 512)
                )

    def set_text_anchors(self, anchors):
        """设置文本锚点特征 (CLIP 生成的文本特征)"""
        anchors = F.normalize(anchors, p=2, dim=-1) # L2 归一化
        self.register_buffer('text_anchors', anchors, persistent=False) # 不更新也不打包传播

    def forward(self, x):
        # 卷积池化基础层
        x = self.base(x)
        # logits = self.head(x)
        
        # 基于原型推理
        aligned_feature = self.clip_adapter(x)
        aligned_feature = F.normalize(aligned_feature, p=2, dim=-1)

        if hasattr(self, 'text_anchors') and self.text_anchors is not None:
            # 计算与文本锚点的相似度 (点积) 并应用温度缩放
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            logits = logit_scale * aligned_feature @ self.text_anchors.T
        else:
            print("Warning: Text anchors not set. Returning unscaled features as logits.")
            logits = aligned_feature @ self.text_anchors.T if hasattr(self, 'text_anchors') else None

        return aligned_feature, logits

    # === 模型分解与恢复方法 ===

    def decom_model(self, rank_rate):
        """将 fc1 分解为低秩层"""
        if rank_rate >= 1.0:
            return
        
        print(f"Executing Decomposition (rank_rate={rank_rate})...")

        # 分解 FC1
        if isinstance(self.fc1, nn.Linear):
            # 调用你提供的 Decom_LINEAR
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
            print(" -> fc1 decomposed.")
        
        self._rebuid_base() # 分解完成后重构基础组件
            
        self.rank_rate = rank_rate

    def _rebuid_base(self):
        """重建基础层的 Sequential 模块 (在分解或恢复后调用)"""
        self.base = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu, self.pool,
            self.conv4, self.relu,
            self.flatten,
            self.fc1,
            self.relu
        )

    def recover_model(self):
        """将 fc1 恢复为完整层"""
        if self.rank_rate >= 1.0:
            return

        print("Executing Recovery to Full Rank...")

        # 恢复 FC1
        if isinstance(self.fc1, FactorizedLinear):
            # 调用你提供的 Recover_LINEAR
            self.fc1 = Recover_LINEAR(self.fc1)
            print(" -> fc1 recovered.")
            
        self._rebuid_base()
        self.rank_rate = 1.0

    # === 正则化 Loss 计算 (用于低秩训练阶段) ===
    
    def L2_decay(self, type='frobenius'):
        """计算分解层的正则化损失"""
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        
        if self.rank_rate >= 1.0:
            return loss

        # 获取 FC1 的 loss
        if isinstance(self.fc1, FactorizedLinear):
            if type == 'frobenius': loss += self.fc1.frobenius_loss()
            elif type == 'kronecker': loss += self.fc1.kronecker_loss()
            elif type == 'l2': loss += self.fc1.L2_loss()
            
        return loss

class CNNCifar5Layer3(nn.Module):
    def __init__(self, dataset, w_conv_bias=True, w_fc_bias=True, save_activations=True):
        super(CNNCifar5Layer3, self).__init__()

        # 确定类别数
        self.num_classes = _decide_num_classes(dataset)
        
        # === 4层卷积结构 (保持与 MetaCNN 一致的对齐结构) ===
        # 统一使用 kernel_size=3, padding=1，防止特征图过早缩小到无法计算
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)

        # 池化与激活
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
        # === 第5层提取层 (FC1) ===
        # 计算全连接层输入维度:
        # 输入 32x32 -> (Pool) 16x16 -> (Pool) 8x8 -> (Pool) 4x4
        # 最终特征图大小: 256 channels * 4 * 4 = 4096
        self.flat_dim = 4096
        self.fc1 = nn.Linear(self.flat_dim, 512, bias=w_fc_bias)

        # === 分类器 ===
        self.head = nn.Linear(512, self.num_classes, bias=w_fc_bias)

        self.save_activations = save_activations
        self.activations = None

        # # 构建共享基础部分 (4 Conv + 1 FC = 5 提取层)
        # self.base = nn.Sequential(
        #     self.conv1, self.relu, self.pool,
        #     self.conv2, self.relu, self.pool,
        #     self.conv3, self.relu, self.pool,
        #     self.conv4, self.relu, # 第4层卷积后不接池化，以对齐 4096 的展平维度
        #     self.flatten,
        #     self.fc1,
        #     self.relu
        # )
        
        # 分类头
        # self.head = self.classifier

    def forward(self, x, start_layer_idx=0):
        # 兼容你原有的分层截断前向逻辑
        if start_layer_idx < 0:
            logits = self.head(x)
            return x, logits
            
        # 基础特征提取
        # x = self.base(x)
        # 基础特征提取 (手动执行，替代 Sequential)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x)) # 第4层卷积后不接池化
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        # 获取预测 logits
        logits = self.head(x)

        return x, logits # 返回 feat, logits





"""
针对femnist设计的分解低质网络
"""
class MetaCNNFemnist(nn.Module):
    def __init__(self, dataset, w_conv_bias=True, w_fc_bias=True, rank_rate=1.0):
        super(MetaCNNFemnist, self).__init__()
        
        # 决定分类数 (FEMNIST 通常为 62 类)
        self.num_classes = _decide_num_classes(dataset)
        self.rank_rate = rank_rate # 秩率
        self.w_conv_bias = w_conv_bias # 记录配置以便恢复时使用
        self.w_fc_bias = w_fc_bias

        # === 前2层卷积 (保持固定，不进行分解) ===
        # FEMNIST 输入是 1通道 (28x28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=0, stride=1, bias=w_conv_bias)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=w_conv_bias)
        
        # 池化与激活
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
        # 计算FEMNIST Flatten后的维度: 64通道 * 4 * 4 = 1024
        self.flat_dim = 1024
        # 根据标准 CNNfemnist，FC1 的输出维度为 2048
        self.fc1_out_dim = 2048
        
        # === FC1 层 (可分解目标) ===
        # 输入: 1024, 输出: 2048
        if rank_rate >= 1.0:
            self.fc1 = nn.Linear(self.flat_dim, self.fc1_out_dim, bias=w_fc_bias)
        else:
            self.fc1 = FactorizedLinear(self.flat_dim, out_features=self.fc1_out_dim, 
                                        rank_rate=rank_rate, bias=w_fc_bias)

        # 构建共享基础部分 (和标准 CNNfemnist 一致的提取流程)
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

        # [ADD] 新增全局共享、可训练的 CLIP 适配层
        # 输入是 fc1 输出的 2048 维特征，输出是对齐 CLIP 的 512 维特征
        self.clip_adapter = nn.Sequential(
                nn.Linear(self.fc1_out_dim, 512),
                nn.ReLU(inplace=True),
        )
        
        # CLIP 的温度缩放参数 (初始化为 CLIP 的默认值)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_text_anchors(self, anchors):
        """设置文本锚点特征 (CLIP 生成的文本特征)"""
        anchors = F.normalize(anchors, p=2, dim=-1) # L2 归一化
        self.register_buffer('text_anchors', anchors, persistent=False) # 不更新也不打包传播

    def forward(self, x):
        # 基础特征提取
        x = self.base(x)

        # 基于原型推理
        aligned_feature = self.clip_adapter(x)
        aligned_feature = F.normalize(aligned_feature, p=2, dim=-1)

        if hasattr(self, 'text_anchors') and self.text_anchors is not None:
            # 计算与文本锚点的相似度 (点积) 并应用温度缩放
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            logits = logit_scale * aligned_feature @ self.text_anchors.T
        else:
            print("Warning: Text anchors not set. Returning unscaled features as logits.")
            logits = aligned_feature # 或者您可以自定义 fallback 逻辑

        return aligned_feature, logits

    # === 模型分解与恢复方法 ===

    def decom_model(self, rank_rate):
        """将 fc1 分解为低秩层"""
        if rank_rate >= 1.0:
            return
        
        print(f"Executing Decomposition (rank_rate={rank_rate})...")
        
        # 分解 FC1
        if isinstance(self.fc1, nn.Linear):
            # 调用你提供的 Decom_LINEAR
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
            print(" -> fc1 decomposed.")
        
        self._rebuid_base() # 分解完成后重构基础组件
        self.rank_rate = rank_rate

    def _rebuid_base(self):
        """重建基础层的 Sequential 模块 (在分解或恢复后调用)"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

    def recover_model(self):
        """将 fc1 恢复为完整层"""
        if self.rank_rate >= 1.0:
            return

        print("Executing Recovery to Full Rank...")

        # 恢复 FC1
        if isinstance(self.fc1, FactorizedLinear):
            # 调用你提供的 Recover_LINEAR
            self.fc1 = Recover_LINEAR(self.fc1)
            print(" -> fc1 recovered.")
            
        self._rebuid_base()
        self.rank_rate = 1.0

    # === 正则化 Loss 计算 (用于低秩训练阶段) ===
    
    def get_decomposed_loss(self, type='frobenius'):
        """计算分解层的正则化损失"""
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        
        if self.rank_rate >= 1.0:
            return loss

        # 获取 FC1 的 loss
        if isinstance(self.fc1, FactorizedLinear):
            if type == 'frobenius': loss += self.fc1.frobenius_loss()
            elif type == 'kronecker': loss += self.fc1.kronecker_loss()
            elif type == 'l2': loss += self.fc1.L2_loss()
            
        return loss
       
"""
tiny数据集模型
"""
class CNNTiny(nn.Module):
    def __init__(self, dataset="imagenet_tiny", w_conv_bias=True, w_fc_bias=True, save_activations=True):
        super(CNNTiny, self).__init__()
        self.num_classes = 200
        
        # 去除了 GroupNorm，直接保留卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
        # 针对 64x64 输入，经过 3 次 2x2 池化后尺寸为 8x8
        # 256 * 8 * 8 = 16384
        self.flat_dim = 16384
        self.fc1 = nn.Linear(self.flat_dim, 512, bias=w_fc_bias)
        self.head = nn.Linear(512, self.num_classes, bias=w_fc_bias)

        self.save_activations = save_activations
        self.activations = None

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            return x, self.head(x)
            
        # 前向传播时去除 GN 的调用
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x)) # 第4层按原逻辑不接池化
        
        x = self.flatten(x)
        # x = self.dropout(x) # 依然保留了注释状态，如果你后续在 Non-IID 下训练遇到严重过拟合，可以把这行解开
        x = self.relu(self.fc1(x))
        
        logits = self.head(x)
        return x, logits    

class MetaCNN3_tiny(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True, rank_rate=1.0):
        super(MetaCNN3_tiny, self).__init__()
       
        self.num_classes = 200
        self.rank_rate = rank_rate # 秩率
        self.w_conv_bias = w_conv_bias # 记录配置以便恢复时使用
        self.w_fc_bias = w_fc_bias

        # BN层通常不需要卷积偏置
        use_conv_bias = False

        # === 前4层卷积 (保持固定，不进行分解) ===
        # 注意：这里改用了 kernel_size=3, padding=1，这在4层CNN中更常见，
        # 能避免特征图缩小过快，从而严格对齐你注释中的 256 * 4 * 4 = 4096 维度 (假设输入为32x32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True)

        # 池化与激活
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        # 计算Flatten后的维度: 假设经过3次池化 (32->16->8->4)，256 * 4 * 4 = 4096
        # 如果你的输入图片尺寸不是 32x32，这里需要相应调整
        self.flat_dim = 256 * 8 * 8    
        # === FC1 层 (可分解目标) ===
        # 输入: 4096 (基于上方计算), 输出: 512
        if rank_rate >= 1.0:
            self.fc1 = nn.Linear(self.flat_dim, 512)
        else:
            self.fc1 = FactorizedLinear(self.flat_dim, out_features=512,
                                        rank_rate=rank_rate, bias=w_fc_bias)

        # === Classifier (保持固定) ===
        # self.classifier = nn.Linear(512, self.num_classes)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 构建共享基础部分 (加入 conv3 和 conv4 及其对应的池化/激活)
        self.base = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu, self.pool,
            self.conv4, self.relu, # 这里假设第4层后不接池化，或者可以根据需要补上 self.pool
            self.flatten,
            self.fc1,
            self.relu
        )

        # [ADD] 新增全局共享、可训练的 CLIP 适配层
        self.clip_adapter = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                # nn.Linear(512, 512)

                )

    def set_text_anchors(self, anchors):
        """设置文本锚点特征 (CLIP 生成的文本特征)"""
        anchors = F.normalize(anchors, p=2, dim=-1) # L2 归一化
        self.register_buffer('text_anchors', anchors, persistent=False) # 不更新也不打包传播

    def forward(self, x):
        # 卷积池化基础层
        x = self.base(x)
       
        # 基于原型推理
        aligned_feature = self.clip_adapter(x)
        aligned_feature = F.normalize(aligned_feature, p=2, dim=-1)

        if hasattr(self, 'text_anchors') and self.text_anchors is not None:
            # 计算与文本锚点的相似度 (点积) 并应用温度缩放
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
            logits = logit_scale * aligned_feature @ self.text_anchors.T
        else:
            print("Warning: Text anchors not set. Returning unscaled features as logits.")
            logits = aligned_feature @ self.text_anchors.T if hasattr(self, 'text_anchors') else None

        return aligned_feature, logits



    # === 模型分解与恢复方法 ===

    def decom_model(self, rank_rate):
        """将 fc1 分解为低秩层"""
        if rank_rate >= 1.0:
            return
        print(f"Executing Decomposition (rank_rate={rank_rate})...")

        # 分解 FC1

        if isinstance(self.fc1, nn.Linear):

            # 调用你提供的 Decom_LINEAR

            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
            print(" -> fc1 decomposed.")
    
        self._rebuid_base() # 分解完成后重构基础组件
        self.rank_rate = rank_rate



    def _rebuid_base(self):

        """重建基础层的 Sequential 模块 (在分解或恢复后调用)"""
        self.base = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu, self.pool,
            self.conv4, self.relu,
            self.flatten,
            self.fc1,
            self.relu

        )

    def recover_model(self):
        """将 fc1 恢复为完整层"""
        if self.rank_rate >= 1.0:
            return

        print("Executing Recovery to Full Rank...")

       # 恢复 FC1
        if isinstance(self.fc1, FactorizedLinear):
            # 调用你提供的 Recover_LINEAR
            self.fc1 = Recover_LINEAR(self.fc1)
            print(" -> fc1 recovered.")
        
        self._rebuid_base()
        self.rank_rate = 1.0

    # === 正则化 Loss 计算 (用于低秩训练阶段) ===
   
    def L2_decay(self, type='frobenius'):

        """计算分解层的正则化损失"""
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
       
        if self.rank_rate >= 1.0:

            return loss

        # 获取 FC1 的 loss

        if isinstance(self.fc1, FactorizedLinear):
            if type == 'frobenius': loss += self.fc1.frobenius_loss()
            elif type == 'kronecker': loss += self.fc1.kronecker_loss()
            elif type == 'l2': loss += self.fc1.L2_loss()

           
        return loss

class Scaler(nn.Module):
    def __init__(self, rate):
        super(Scaler, self).__init__()
        self.rate = rate

    def forward(self, x):
        return x / self.rate if self.training else x
    
class HeteroCNNCifar(nn.Module):
    def __init__(
        self,
        dataset,
        width_mult=1.0,
        w_conv_bias=True,
        w_fc_bias=True,
        use_scaler=True,):
        super(HeteroCNNCifar, self).__init__()
        self.num_classes = _decide_num_classes(dataset)

        c1 = max(1, int(32 * width_mult))
        c2 = max(1, int(64 * width_mult))
        c3 = max(1, int(128 * width_mult))
        c4 = max(1, int(256 * width_mult))
        h1 = max(1, int(512 * width_mult))

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c4 * 4 * 4, h1, bias=w_fc_bias)
        self.classifier = nn.Linear(h1, self.num_classes, bias=w_fc_bias)
        self.scaler = Scaler(width_mult) if use_scaler else nn.Identity()


    def forward(self, x):
        x = self.pool(F.relu(self.scaler(self.conv1(x))))
        x = self.pool(F.relu(self.scaler(self.conv2(x))))
        x = self.pool(F.relu(self.scaler(self.conv3(x))))
        x = F.relu(self.scaler(self.conv4(x)))
        x = x.view(x.size(0), -1)
        feat = F.relu(self.scaler(self.fc1(x)))
        logits = self.classifier(feat)
        return feat, logits

class HeteroCNNtiny(nn.Module):
    def __init__(
        self,
        dataset,
        width_mult=1.0,
        w_conv_bias=True,
        w_fc_bias=True,
        use_scaler=True,):
        super(HeteroCNNtiny, self).__init__()
        self.num_classes = _decide_num_classes(dataset)

        c1 = max(1, int(32 * width_mult))
        c2 = max(1, int(64 * width_mult))
        c3 = max(1, int(128 * width_mult))
        c4 = max(1, int(256 * width_mult))
        h1 = max(1, int(512 * width_mult))

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, padding=1, stride=1, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c4 * 8 * 8, h1, bias=w_fc_bias)
        self.classifier = nn.Linear(h1, self.num_classes, bias=w_fc_bias)
        self.scaler = Scaler(width_mult) if use_scaler else nn.Identity()


    def forward(self, x):
        x = self.pool(F.relu(self.scaler(self.conv1(x))))
        x = self.pool(F.relu(self.scaler(self.conv2(x))))
        x = self.pool(F.relu(self.scaler(self.conv3(x))))
        x = F.relu(self.scaler(self.conv4(x)))
        x = x.view(x.size(0), -1)
        feat = F.relu(self.scaler(self.fc1(x)))
        logits = self.classifier(feat)
        return feat, logits

class MetaCNN3_Classifier(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True, rank_rate=1.0):
        super(MetaCNN3_Classifier, self).__init__()
        
        print("分类头低质模型")
        self.num_classes = _decide_num_classes(dataset)
        self.rank_rate = rank_rate  # 秩率
        self.w_conv_bias = w_conv_bias
        self.w_fc_bias = w_fc_bias

        # 4层卷积主干（和原MetaCNN3完全一致，不改动）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=True)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
        self.flat_dim = 4096  # 32x32 cifar 展平维度
        # 可分解FC1
        if rank_rate >= 1.0:
            self.fc1 = nn.Linear(self.flat_dim, 512)
        else:
            self.fc1 = FactorizedLinear(self.flat_dim, out_features=512, 
                                        rank_rate=rank_rate, bias=w_fc_bias)

        # ========== 新增独立标准分类头 ==========
        self.classifier_head = nn.Linear(512, self.num_classes)

        # CLIP 相关模块（保留原有原型推理分支）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_adapter = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
        )

        # 主干序列
        self.base = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu, self.pool,
            self.conv4, self.relu,
            self.flatten,
            self.fc1,
            self.relu
        )

    # def set_text_anchors(self, anchors):
    #     anchors = F.normalize(anchors, p=2, dim=-1)
    #     self.register_buffer('text_anchors', anchors, persistent=False)

    def forward(self, x):
        """
        Args:
            x: 输入图像
            use_classifier_head: True=使用线性分类头输出logits；False=使用文本锚点相似度输出logits
        Return:
            aligned_feature: 经过clip适配器对齐后的512维特征
            logits: 分类分数（二选一）
        """
        # 卷积+FC1提取基础特征
        feat = self.base(x)
        aligned_feature = self.clip_adapter(feat)
        aligned_feature = F.normalize(aligned_feature, p=2, dim=-1)

        logits = self.classifier_head(feat)

        return aligned_feature, logits

    # ====== 以下分解/恢复/正则loss函数完全复用原MetaCNN3逻辑，无需修改 ======
    def decom_model(self, rank_rate):
        if rank_rate >= 1.0:
            return
        print(f"Executing Decomposition (rank_rate={rank_rate})...")
        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
            print(" -> fc1 decomposed.")
        self._rebuid_base()
        self.rank_rate = rank_rate

    def _rebuid_base(self):
        self.base = nn.Sequential(
            self.conv1, self.relu, self.pool,
            self.conv2, self.relu, self.pool,
            self.conv3, self.relu, self.pool,
            self.conv4, self.relu,
            self.flatten,
            self.fc1,
            self.relu
        )

    def recover_model(self):
        if self.rank_rate >= 1.0:
            return
        print("Executing Recovery to Full Rank...")
        if isinstance(self.fc1, FactorizedLinear):
            self.fc1 = Recover_LINEAR(self.fc1)
            print(" -> fc1 recovered.")
        self._rebuid_base()
        self.rank_rate = 1.0

    def L2_decay(self, type='frobenius'):
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        if self.rank_rate >= 1.0:
            return loss
        if isinstance(self.fc1, FactorizedLinear):
            if type == 'frobenius': loss += self.fc1.frobenius_loss()
            elif type == 'kronecker': loss += self.fc1.kronecker_loss()
            elif type == 'l2': loss += self.fc1.L2_loss()
        return loss


class MetaCNN5Fixed025(nn.Module):
    def __init__(self, dataset, w_conv_bias=False, w_fc_bias=True):
        super(MetaCNN5Fixed025, self).__init__()
        
        self.num_classes = _decide_num_classes(dataset)
        # 固定分解秩 0.25，不可外部传入修改
        self.rank_rate = 0.25
        self.w_conv_bias = w_conv_bias
        self.w_fc_bias = w_fc_bias

        # 卷积层完全和 CNNCifar5Layer 保持一致
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        
        # 32x32 cifar 经过两次5卷积+两次池化后展平维度 1600
        self.flat_dim = 1600
        
        # 直接实例化低秩分解FC1，固定rank_rate=0.25，不再做完整nn.Linear分支
        self.fc1 = FactorizedLinear(self.flat_dim, out_features=512,
                                    rank_rate=self.rank_rate, bias=w_fc_bias)

        # 标准分类头，和基线CNNCifar5Layer统一
        self.classifier = nn.Linear(512, self.num_classes, bias=w_fc_bias)
        self.head = self.classifier

        # 构建主干Sequential
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            logits = self.head(x)
            return x, logits
        feat = self.base(x)
        logits = self.head(feat)
        return feat, logits

    # 分解接口（当前已经是分解状态，rank=0.25，重复调用无操作）
    def decom_model(self, rank_rate):
        print(f"Current model is pre-decomposed with fixed rank_rate=0.25, skip re-decompose.")
        return

    def _rebuid_base(self):
        """分解/恢复后重建主干序列"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu
        )

    def recover_model(self):
        """恢复fc1为完整稠密线性层"""
        if self.rank_rate >= 1.0:
            return
        print("Executing Recovery to Full Rank...")
        if isinstance(self.fc1, FactorizedLinear):
            self.fc1 = Recover_LINEAR(self.fc1)
            print(" -> fc1 recovered to full linear layer.")
        self._rebuid_base()
        self.rank_rate = 1.0

    def L2_decay(self, type='frobenius'):
        """低秩正则损失计算，仅fc1参与"""
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        if self.rank_rate >= 1.0:
            return loss
        if isinstance(self.fc1, FactorizedLinear):
            if type == 'frobenius':
                loss += self.fc1.frobenius_loss()
            elif type == 'kronecker':
                loss += self.fc1.kronecker_loss()
            elif type == 'l2':
                loss += self.fc1.L2_loss()
        return loss

def simple_cnn(conf):
    dataset = conf.data
#     获取秩 rank_rate
    rank_rate = conf.ratio_LR
    # rank_rate = 0.5

    if "cifar" in dataset or dataset == "svhn":

        if conf.method == "fedlagc":
            return HeteroCNNCifar(
                dataset,
                width_mult=1.0,
                w_conv_bias=conf.w_conv_bias,
                w_fc_bias=conf.w_fc_bias,
                use_scaler=False,
            )
        
        if conf.method in ["heterofl", "fedrolex","feddropout", "feddf", "fedmd", "fjord","feddse",]:
            return HeteroCNNCifar(
                dataset,
                width_mult=getattr(conf, "model_rate", 1.0),
                w_conv_bias=conf.w_conv_bias,
                w_fc_bias=conf.w_fc_bias,
                use_scaler=getattr(conf, "heterofl_use_scaler", True),
            )
        if conf.meta: # 如果分解
            return MetaCNN3(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias, rank_rate =rank_rate )
            # return MetaCNN3_Classifier(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias, rank_rate =rank_rate )
        # return CNNCifar(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
        # return CNNCifar5Layer3(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
        return MetaCNN5Fixed025(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)

    

    elif "mnist" == dataset:
        return CNNMnist(dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
    elif "tiny" in dataset:
        if conf.method == "fedlagc":
            return HeteroCNNtiny(
                dataset,
                width_mult=1.0,
                w_conv_bias=conf.w_conv_bias,
                w_fc_bias=conf.w_fc_bias,
                use_scaler=False,
            )
        if conf.method in ["heterofl", "fedrolex","feddropout", "feddf", "fedmd", "fjord","feddse",]:
            return HeteroCNNtiny(
                dataset,
                width_mult=getattr(conf, "model_rate", 1.0),
                w_conv_bias=conf.w_conv_bias,
                w_fc_bias=conf.w_fc_bias,
                use_scaler=getattr(conf, "heterofl_use_scaler", True),
            )
        if conf.meta:
            return MetaCNN3_tiny(
                dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias,rank_rate= rank_rate
                )
        return CNNTiny(
            dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias
        )
    elif "femnist" == dataset:
        if conf.meta:
            return MetaCNNFemnist(
                dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias,rank_rate= rank_rate
                )
        return CNNfemnist(
            dataset, w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias
        )
    else:
        raise NotImplementedError(f"not supported yet.")
    




# 分解后的卷积层 
class FactorizedConv(nn.Module):
    '''
    完整的 4维卷积核 转换为 两个较小的 2维矩阵 conv_u 和 conv_v。
    conv_v: 形状为 [rank, in_channels * kernel_size]。这是分解后的第一层权重（降维/特征变换）。
    conv_u: 形状为 [out_channels * kernel_size, rank]。这是分解后的第二层权重（升维/特征重组）。
    
    '''

    def __init__(self, in_channels, out_channels, rank_rate, padding=None, stride=1, kernel_size=3, bias=False):
        super(FactorizedConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding if padding is not None else 0
        self.groups = 1  # 分组卷积支持，默认为1

        # 计算低秩分解的秩
        # self.rank = max(1, round(rank_rate * min(in_channels, out_channels)))
        self.rank = max(1, round(rank_rate * min(out_channels * kernel_size, in_channels * kernel_size)))
        # 使用二维矩阵存储分解参数
        # 通用处理任意kernel_size
        self.dim1 = out_channels * kernel_size #对应卷积层分解成矩阵后的两个维度
        self.dim2 = in_channels * kernel_size

        # 低秩参数矩阵 (二维存储) 可学习 随机初始化分解形状的卷积层

        self.conv_v = nn.Parameter(torch.Tensor(self.rank, self.dim2)) #对应形状 [rank, in_channels * kernel_size]
        self.conv_u = nn.Parameter(torch.Tensor(self.dim1, self.rank)) #对应形状 [out_channels * kernel_size, rank]

        # 偏置参数 
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels)) # 偏执维度对应输出维度
        else:
            self.register_parameter('bias', None) # 注册为空 可以访问
        # 初始化模型参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_u, a=math.sqrt(0)) #使用 Kaiming 均匀分布初始化分解后的卷积参数
        nn.init.kaiming_uniform_(self.conv_v, a=math.sqrt(0))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 空间分解: 1xK + Kx1
        # 垂直卷积 (1xK)
        weight_v = self.conv_v.T.reshape(self.in_channels, self.kernel_size, 1, self.rank).permute(3, 0, 2, 1)
        #对应形状 [rank, in_channels * kernel_size]->转置 [in_channels * kernel_size, rank]
        # ->重塑 [in_channels, kernel_size,1, rank]->换轴 [rank, in_channels,1,kernel_size]
        out = F.conv2d(
            x, weight_v, None,
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, 1),
            groups=self.groups
        )

        # 水平卷积 (Kx1)  不显示reshpe权重
        weight_u = self.conv_u.reshape(self.out_channels, self.kernel_size, self.rank, 1).permute(0, 2, 1, 3)
         #[out_channels * kernel_size, rank] -> 重塑 [out_channels, kernel_size, rank, 1] ->
         #  换轴 [out_channels, rank, kernel_size, 1]

        out = F.conv2d(
            out, weight_u, self.bias,
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dilation=(1, 1),
            groups=self.groups
        )
        return out
# 正则损失
    def frobenius_loss(self):
        return torch.sum((self.conv_u @ self.conv_v) ** 2)


    def reconstruct_full_weight(self):
        """重建完整的卷积核权重 (用于聚合)"""
        # 直接使用矩阵乘法
        A = self.conv_u @ self.conv_v  # [out*K, in*K]
        W = A.reshape(self.out_channels, self.kernel_size, self.in_channels, self.kernel_size)
        W = W.permute(0, 2, 1, 3)  # [out, in, K, K]
        return W

    def L2_loss(self):
        """分解参数的L2范数平方和"""
        return torch.norm(self.conv_u, p='fro') ** 2 + torch.norm(self.conv_v, p='fro') ** 2

    def kronecker_loss(self):
        """Kronecker乘积损失"""
        return (torch.norm(self.conv_u, p='fro') ** 2) * (torch.norm(self.conv_v, p='fro') ** 2)
    
def Decom_COV(conv_model, ratio_LR=0.5):
    # 自动从卷积层获取参数
    in_planes = conv_model.in_channels
    out_planes = conv_model.out_channels
    kernel_size = conv_model.kernel_size[0]
    stride = conv_model.stride[0]
    padding = conv_model.padding[0]
    bias = conv_model.bias is not None

    # 创建分解层 (使用二维矩阵存储)
    factorized_cov = FactorizedConv(
        in_planes,
        out_planes,
        rank_rate=ratio_LR,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias
    )

    # 获取原始权重并重塑
    W = conv_model.weight.data

    # 重塑: [out, in, K, K] -> [out*K, in*K]
    A = W.permute(0, 2, 1, 3).reshape(out_planes * kernel_size, in_planes * kernel_size)

    # SVD分解
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # 计算截断秩
    rank = factorized_cov.rank
    S_sqrt = torch.sqrt(S[:rank]) # 截断秩 开平方 
    # 分配奇异值
    U_weight = U[:, :rank] @ torch.diag(S_sqrt) 
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]

    # 加载参数
    with torch.no_grad():
        factorized_cov.conv_u.copy_(U_weight)
        factorized_cov.conv_v.copy_(V_weight)

        # 复制偏置
        if bias:
            factorized_cov.bias.copy_(conv_model.bias.data)

    return factorized_cov


# 卷积层恢复函数
def Recover_COV(decom_conv):
    # 获取分解层参数
    in_planes = decom_conv.in_channels
    out_planes = decom_conv.out_channels
    kernel_size = decom_conv.kernel_size
    stride = decom_conv.stride
    padding = decom_conv.padding
    bias = decom_conv.bias is not None

    # 重建完整权重
    W = decom_conv.reconstruct_full_weight()

    # 创建原始卷积层
    recovered_conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias
    )

    # 加载权重
    with torch.no_grad():
        recovered_conv.weight.copy_(W)
        if bias:
            recovered_conv.bias.copy_(decom_conv.bias)

    return recovered_conv

# 分解后的全连接层 (二维矩阵存储)
class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_rate, bias=True):
        super(FactorizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 中间rank值
        self.rank = max(1, round(rank_rate * min(in_features, out_features)))

        # 二维矩阵参数
        # 第一个全连接层的参数（维度为 r*in）
        self.weight_v = nn.Parameter(torch.Tensor(self.rank, in_features))
        # 第二个全连接层的参数 (维度为 out*r)
        self.weight_u = nn.Parameter(torch.Tensor(out_features, self.rank))

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_u, a=math.sqrt(0))
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(0))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight_u.size(1))  # rank
            nn.init.uniform_(self.bias, -bound, bound)

    def reconstruct_full_weight(self):
        return self.weight_u @ self.weight_v

    def forward(self, x):
        """
        前向传播分为两步：
        1. 降维投影：x -> (batch,  rank)
        2. 升维投影：x -> (batch, out_features)
        """
        # 第一步：降维投影 (输入特征空间 -> 低秩空间)  传入 F.linear() 的权重存储形式与正常线性层 (nn.Linear) 完全一致必须是（out*in）
        x = F.linear(x, self.weight_v, None)  # 形状: (batch,  rank)

        # 第二步：升维投影 (低秩空间 -> 输出特征空间)
        x = F.linear(x, self.weight_u, self.bias)  # 形状: (batch, out_features)

        return x

    def frobenius_loss(self):
        W = self.weight_u @ self.weight_v
        return torch.sum(W ** 2)

    def L2_loss(self):
        return torch.norm(self.weight_v) ** 2 + torch.norm(self.weight_u) ** 2

    def kronecker_loss(self):
        return (torch.norm(self.weight_v) ** 2) * (torch.norm(self.weight_u) ** 2)


# 全连接层分解函数(将全连接权重  W（out*in）分解为 out*r（第二个全连接权重） r*in （第一个权全连接重）)
def Decom_LINEAR(linear_model, ratio_LR=0.5):
    in_features = linear_model.in_features
    out_features = linear_model.out_features
    has_bias = linear_model.bias is not None

    # 创建分解层
    factorized_linear = FactorizedLinear(in_features, out_features, ratio_LR, has_bias)

    # SVD分解（属注意与torch.svd函数的区别  主要是第三个矩阵）  Vh是V矩阵的转置（就是第三个矩阵）
    U, S, Vh = torch.linalg.svd(linear_model.weight.data, full_matrices=False)

    # 计算截断秩
    rank = factorized_linear.rank

    # 分配奇异值  第一个矩阵切列   第三个矩阵切行
    S_sqrt = torch.sqrt(S[:rank])
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)  # shape out*r
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]  # shape r*in

    # 加载参数
    with torch.no_grad():
        factorized_linear.weight_v.copy_(V_weight)
        factorized_linear.weight_u.copy_(U_weight)
        if has_bias:
            factorized_linear.bias.copy_(linear_model.bias.data)

    return factorized_linear


# 全连接层恢复函数
def Recover_LINEAR(factorized_linear):
    in_features = factorized_linear.in_features
    out_features = factorized_linear.out_features
    has_bias = factorized_linear.bias is not None

    # # 重建权重
    # weight = factorized_linear.weight_u @ factorized_linear.weight_v
    weight = factorized_linear.reconstruct_full_weight()

    # 创建原始线性层
    recovered_linear = nn.Linear(in_features, out_features, bias=has_bias)

    # 加载参数
    with torch.no_grad():
        recovered_linear.weight.copy_(weight)
        if has_bias:
            recovered_linear.bias.copy_(factorized_linear.bias)

    return recovered_linear
