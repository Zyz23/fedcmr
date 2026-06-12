

import os  # 必须先加这一行导入 os 模块
CPU_THREADS = int(os.environ.get("CPU_THREADS", "2"))

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

import torch

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 然后再设置环境变量 2,3,4,5

import torch.multiprocessing as mp
# -*- coding: utf-8 -*-
from parameters import get_args
from pcode.masters import *
from pcode.workers import *
import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser
import random
import numpy as np

MethodTable = {
    "fedavg": [Master, Worker],
    # "fedavg": [Master, Worker_a],
    "fedgen": [MasterFedgen, WorkerFedGen],
    "feddistill": [MasterFedDistill, WorkerFedDistill],
    "moon": [MasterMoon, WorkerMoon],
    "fedgkd": [MasterFedGKD, WorkerFedGKD],
    "fedprox": [Master, WorkerFedProx],
    "feddyn":[MasterFedDyn, WorkerFedDyn],
    "fedadam":[MasterFedAdam, Worker],
    "fedadam_gkd":[MasterFedAdam, WorkerFedGKD],
    "fedensemble":[MasterFedEnsemble, Worker],
    "fedhm":[MasterFedHM, WorkerFedHM],
    "heterofl":[MasterHeteroFL, Worker],
    "fedrolex":[MasterFedRolex, Worker],
    "feddropout":[MasterFedDropout, Worker],
    "feddf":[MasterFedDF, Worker],
    "fedour":[MasterFedOur, WorkerFedOur],
    "fedmd":[MasterFedMD, Worker],
    "fjord":[MasterFjORD, WorkerFjORD],
    "fiarse":[MasterFIARSE, WorkerFIARSE],
    "fedsam":[MasterFedSam, WorkerFedSam],
    
    # "fedhm":[MasterFedHM, Worker_a]

}
# 随机生成秩
# def random_rank_creater(count: int):
#     return np.random.uniform(0.2, 0.8, count)
def random_rank_creater(count: int):
    # 定义可选的固定数值列表
    candidate_values = np.array([0.25])
    # 从候选值中随机选择count个（可重复选择，replace=True为默认值）
    return np.random.choice(candidate_values, size=count)

def hetero_model_rate_creator(conf, rates_text=None):
    rates_text = conf.hetero_model_rates if rates_text is None else rates_text
    candidates = np.array(
        [float(rate.strip()) for rate in rates_text.split(",") if rate.strip()]
    )
    if len(candidates) == 0:
        raise ValueError("model/capacity rates must contain at least one float value.")
    if np.any(candidates <= 0) or np.any(candidates > 1):
        raise ValueError("model/capacity rates must be in the interval (0, 1].")
    if conf.hetero_rate_assignment == "random":
        rng = np.random.RandomState(conf.manual_seed)
        return rng.choice(candidates, size=conf.n_clients, replace=True)
    return np.array([candidates[idx % len(candidates)] for idx in range(conf.n_clients)])
    
def main(rank, size, conf, port): # rank为进程序号
    # init the distributed world.
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group("gloo", rank=rank, world_size=size)
        # dist.init_process_group("gloo", rank=rank, world_size=size) # 所有进程走到此步成功链接 程序继续执行
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config. 初始化配置 进入方法
    init_config(conf) # 配置图结构 随机数 种子 日志等 

 

    assert MethodTable[conf.method] is not None
    master, worker = MethodTable[conf.method] # 得到对应算法的服务器和客户端



    # start federated learning. 根据 rank 选择服务器或客户端
    process = master(conf) if conf.graph.rank == 0 else worker(conf)
    process.run()


def init_config(conf):
    # define the graph for the computation.
    # 进入工具类 utils.topology 获取图结构
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    # 返回的conf.graph.rank是-1 这里根据进程获取真正的rank
    conf.graph.rank = dist.get_rank() #初始化conf.graph.rank = -1 调用进程 获取对应的id 方便区分服务器和客户端

    # init related to randomness on cpu.
    #如果conf.same_seed_process为fasle 即每个进程的种子不重复 是独立的
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    #os.environ['PYTHONHASHSEED'] = str(conf.manual_seed)
    #random.seed(conf.manual_seed)
    #np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    init_cuda(conf)

    # init the model arch info.
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)  # 构造字典 服务器和客户端的模型
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # define checkpoint for logging (for federated learning server). 初始化检查点
    #为每个进程创建独立的日志和模型存储目录
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    # configure logger. 创建日志 位于检查点设置的文件目录
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info. 打印u服务器的参数配置信息
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes. 阻塞进程 在此等待
    dist.barrier()


def init_cuda(conf):
    torch.cuda.set_device(torch.device("cuda:" + str(conf.graph.rank % torch.cuda.device_count())))
    torch.cuda.manual_seed(conf.manual_seed)
    #torch.cuda.manual_seed_all(conf.manual_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


import time

if __name__ == "__main__":
    conf = get_args()  # 进入 parameter.py 获取配置信息
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5) # 获取参与客户端数
    conf.timestamp = str(int(time.time()))
    size = conf.n_participated + 1 # 总进程数 一个服务器 n个客户端
    processes = []

    clients_nums = conf.n_clients
    
    if conf.method in ["heterofl", "fedrolex", "feddropout", "feddf", "fedmd", "fjord", "fiarse"]:
        rates_text = (
            conf.fiarse_capacity_rates
            if conf.method == "fiarse" and conf.fiarse_capacity_rates is not None
            else conf.hetero_model_rates
        )
        conf.model_rate_list = hetero_model_rate_creator(conf, rates_text=rates_text)
        conf.model_rate = 1.0
    if conf.is_random_rank and conf.method not in ["heterofl", "fedrolex", "feddropout", "feddf", "fedmd", "fjord", "fiarse"]:
        conf.rank_list = random_rank_creater(clients_nums)
    # if conf.is_random_rank:
    #     conf.rank_list = random_rank_creater(clients_nums)  #所有客户端的秩 分解使用
       

    

    mp.set_start_method("spawn") #创建子进程

    # 进程的唯一标识符（0 是 Master，1 及以后是 Worker）。
    for rank in range(size):
        p = mp.Process(target=main, args=(rank, size, conf, conf.port)) # 每一个进程都进行main函数
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # 进入 此类的main方法 初始化配置
