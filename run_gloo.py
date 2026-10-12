
# -*- coding: utf-8 -*-

import os

# ============================================================
# 必须在 import torch 之前设置 CPU / CUDA 环境变量
# ============================================================
CPU_THREADS = int(os.environ.get("CPU_THREADS", "2"))

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

# 你的当前可见 GPU。
# 若之后切换显卡，只改这里，例如 "0" 或 "2,3"。
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import time
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from parameters import get_args

from pcode.masters import *
from pcode.workers import *

import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser

# 新增方法直接导入，避免依赖 pcode/masters/__init__.py。
from pcode.masters.master_feddse import MasterFedDSE
from pcode.workers.worker_feddse import WorkerFedDSE

from pcode.masters.master_fedlagc import MasterFedLAGC
from pcode.workers.worker_fedlagc import WorkerFedLAGC

from pcode.masters.master_fedalign import MasterFedAlign
from pcode.workers.worker_fedalign import WorkerFedAlign

from pcode.masters.master_fedre import MasterFedRE
from pcode.workers.worker_fedre import WorkerFedRE

from pcode.masters.master_fedgmr import MasterFedGMR
from pcode.workers.worker_fedgmr import WorkerFedGMR


# ============================================================
# 方法注册表
# ============================================================
MethodTable = {
    "fedavg": [Master, Worker],
    # "fedavg": [Master, Worker_a],

    "fedgen": [MasterFedgen, WorkerFedGen],
    "feddistill": [MasterFedDistill, WorkerFedDistill],
    "moon": [MasterMoon, WorkerMoon],
    "fedgkd": [MasterFedGKD, WorkerFedGKD],
    "fedprox": [Master, WorkerFedProx],
    "feddyn": [MasterFedDyn, WorkerFedDyn],
    "fedadam": [MasterFedAdam, Worker],
    "fedadam_gkd": [MasterFedAdam, WorkerFedGKD],
    "fedensemble": [MasterFedEnsemble, Worker],
    "fedhm": [MasterFedHM, WorkerFedHM],

    "heterofl": [MasterHeteroFL, Worker],
    "fedrolex": [MasterFedRolex, Worker],
    "feddropout": [MasterFedDropout, Worker],
    "feddf": [MasterFedDF, Worker],
    "fedour": [MasterFedOur, WorkerFedOur],
    "fedmd": [MasterFedMD, Worker],
    "fjord": [MasterFjORD, WorkerFjORD],
    "fiarse": [MasterFIARSE, WorkerFIARSE],
    "feddse": [MasterFedDSE, WorkerFedDSE],
    "fedlagc": [MasterFedLAGC, WorkerFedLAGC],

    "fedsam": [MasterFedSam, WorkerFedSam],
    "fedswa": [MasterFedSWA, WorkerFedSWA],
    "fedmoswa": [MasterFedMoSWA, WorkerFedMoSWA],
    "fedalign": [MasterFedAlign, WorkerFedAlign],
    "fedre": [MasterFedRE, WorkerFedRE],
    "fedgmr": [MasterFedGMR, WorkerFedGMR],

    # "fedhm": [MasterFedHM, Worker_a],
}


# ============================================================
# 随机秩生成：供 MetaCNN / 低秩方法使用
# ============================================================
def random_rank_creater(count: int):
    candidate_values = np.array(
        [1],
        dtype=np.float32,
    )

    return np.random.choice(
        candidate_values,
        size=count,
        replace=True,
    )


# ============================================================
# 客户端容量比例生成
# ============================================================
def hetero_model_rate_creator(conf, rates_text=None):
    if rates_text is None:
        rates_text = conf.hetero_model_rates

    candidates = np.array(
        [
            float(rate.strip())
            for rate in str(rates_text).split(",")
            if len(rate.strip()) > 0
        ],
        dtype=np.float32,
    )

    if len(candidates) == 0:
        raise ValueError(
            "model/capacity rates must contain at least one float value."
        )

    if np.any(candidates <= 0) or np.any(candidates > 1):
        raise ValueError(
            "model/capacity rates must be in the interval (0, 1]."
        )

    if conf.hetero_rate_assignment == "random":
        rng = np.random.RandomState(
            int(conf.manual_seed)
        )

        return rng.choice(
            candidates,
            size=conf.n_clients,
            replace=True,
        ).tolist()

    # cycle：client 1, 2, ..., n_clients 循环分配容量。
    return [
        float(candidates[index % len(candidates)])
        for index in range(conf.n_clients)
    ]


# ============================================================
# 当前实验是否使用 ResNet
# ============================================================
def is_resnet_experiment(conf):
    arch_text = str(
        getattr(conf, "arch", "")
    ).lower()

    complex_arch_text = str(
        getattr(conf, "complex_arch", "")
    ).lower()

    return (
        "resnet" in arch_text
        or "resnet" in complex_arch_text
    )


# ============================================================
# 配置容量异构比例
# ============================================================
def configure_capacity_rates(conf):
    """
    width_hetero_methods:
        HeteroFL 等方法，容量表示网络宽度比例。

    fiarse / fedlagc / fedgmr:
        容量表示参数保留比例或稀疏预算，
        物理网络结构保持完整。
    """
    width_hetero_methods = [
        "heterofl",
        "fedrolex",
        "feddropout",
        "feddf",
        "fedmd",
        "fjord",
        "feddse",
        "fedre",
    ]

    capacity_methods = [
        "heterofl",
        "fedrolex",
        "feddropout",
        "feddf",
        "fedmd",
        "fjord",
        "fiarse",
        "feddse",
        "fedlagc",
        "fedre",
        "fedgmr",
    ]

    if conf.method not in capacity_methods:
        conf.model_rate_list = None
        return width_hetero_methods

    if (
        conf.method == "fiarse"
        and getattr(conf, "fiarse_capacity_rates", None) is not None
    ):
        rates_text = conf.fiarse_capacity_rates

    elif conf.method == "fedlagc":
        rates_text = conf.fedlagc_capacity_rates

    else:
        rates_text = conf.hetero_model_rates

    conf.model_rate_list = hetero_model_rate_creator(
        conf,
        rates_text=rates_text,
    )

    return width_hetero_methods


# ============================================================
# 每个分布式进程的入口
# ============================================================
def main(rank, size, conf, port):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=size,
        )

    except Exception as error:
        print(
            f"Failed to initialize distributed world: {repr(error)}"
        )
        raise

    try:
        init_config(conf)

        if conf.method not in MethodTable:
            raise NotImplementedError(
                f"Unsupported method: {conf.method}"
            )

        master_class, worker_class = MethodTable[
            conf.method
        ]

        process = (
            master_class(conf)
            if conf.graph.rank == 0
            else worker_class(conf)
        )

        process.run()

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


# ============================================================
# 每个进程的配置、随机种子、日志
# ============================================================
def init_config(conf):
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )

    conf.graph.rank = dist.get_rank()

    # 每个 worker 使用独立种子。
    if not conf.same_seed_process:
        conf.manual_seed = (
            1000 * int(conf.manual_seed)
            + int(conf.graph.rank)
        )

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    random.seed(conf.manual_seed)
    np.random.seed(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)

    conf.random_state = np.random.RandomState(
        conf.manual_seed
    )

    init_cuda(conf)

    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {
            "master": conf.arch,
            "worker": conf.arch,
        }
    )

    conf.arch_info["worker"] = (
        conf.arch_info["worker"].split(":")
    )

    checkpoint.init_checkpoint(
        conf,
        rank=str(conf.graph.rank),
    )

    conf.logger = logging.Logger(
        conf.checkpoint_dir
    )

    if conf.graph.rank == 0:
        logging.display_args(conf)

    dist.barrier()


# ============================================================
# CUDA 配置
# ============================================================
def init_cuda(conf):
    if not conf.on_cuda:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "--on_cuda=True, but CUDA is unavailable."
        )

    device_count = torch.cuda.device_count()

    if device_count <= 0:
        raise RuntimeError(
            "No CUDA device is visible."
        )

    device_id = int(conf.graph.rank) % device_count

    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed_all(conf.manual_seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    conf = get_args()

    conf.n_participated = int(
        conf.n_clients * conf.participation_ratio + 0.5
    )

    if conf.n_participated <= 0:
        raise ValueError(
            "n_participated must be positive."
        )

    conf.timestamp = str(int(time.time()))

    clients_nums = conf.n_clients
    size = conf.n_participated + 1
    processes = []

    # --------------------------------------------------------
    # 1. 创建异构方法的 client capacity 列表
    # --------------------------------------------------------
    width_hetero_methods = configure_capacity_rates(conf)

    # --------------------------------------------------------
    # 2. 统一处理 ResNet 的宽度
    #
    # resnet.py 中会优先读取 conf.model_rate。
    # 所以：只要运行的是 ResNet，就让 model_rate 等于
    # --resnet_scaling，保证所有 ResNet 方法都正确生效。
    #
    # 对 FedLAGC + ResNet：
    # 完整物理模型宽度也是 resnet_scaling，
    # 之后再通过参数 mask 做容量异构。
    # --------------------------------------------------------
    use_resnet = is_resnet_experiment(conf)

    if use_resnet:
        conf.model_rate = float(
            getattr(conf, "resnet_scaling", 1.0)
        )

    # 非 ResNet 的 FedLAGC：
    # 完整 CNN 物理宽度固定 1.0，容量异构由 mask 实现。
    elif conf.method == "fedlagc":
        conf.model_rate = 1.0

    # 其余 CNN 全局模型默认完整网络。
    else:
        conf.model_rate = 1.0

    # --------------------------------------------------------
    # 3. 仅低秩分解类方法生成 rank_list
    # --------------------------------------------------------
    excluded_random_rank_methods = [
        "heterofl",
        "fedrolex",
        "feddropout",
        "feddf",
        "fedmd",
        "fjord",
        "fiarse",
        "feddse",
        "fedlagc",
        "fedre",
        "fedgmr",
    ]

    if (
        conf.is_random_rank
        and conf.method not in excluded_random_rank_methods
    ):
        conf.rank_list = random_rank_creater(
            clients_nums
        )

    # spawn 比 fork 更适合 CUDA + 多进程。
    mp.set_start_method(
        "spawn",
        force=True,
    )

    for rank in range(size):
        process = mp.Process(
            target=main,
            args=(
                rank,
                size,
                conf,
                conf.port,
            ),
        )

        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    failed_processes = [
        process.pid
        for process in processes
        if process.exitcode not in [0, None]
    ]

    if len(failed_processes) > 0:
        raise RuntimeError(
            "Federated execution failed. "
            f"Abnormal process PIDs: {failed_processes}"
        )
