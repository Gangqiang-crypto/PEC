"""统一实验配置。

数据集路径留出位置，请在此处填入你的 txt 文件路径：
  - graph_file:  边列表 txt，每行 'u v' 或 'u v w'，'#' 开头为注释
  - label_pool_file: 标签池 txt，每行一个标签词

索引按数据集**所有点和边**构造；训练集从顶点中随机选取 70%，剩余 30% 为测试集。
查询由系统随机发起（从测试集合格顶点中随机抽取源点与标签），无需手动输入。
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetConfig:
    name: str
    graph_file: str
    label_pool_file: str


# =========================================================================
#  数据集位置：请填入你的 txt 路径（可添加任意多个）
# =========================================================================
DATASETS: List[DatasetConfig] = [
    DatasetConfig(
        name="CA-HepTh",
        graph_file=r"data/graphs/CA-HepTh.txt",
        label_pool_file=r"data/labels/Label_diverse.txt",
    ),
    DatasetConfig(
        name="CA-CondMat",
        graph_file=r"data/graphs/CA-CondMat.txt",
        label_pool_file=r"data/labels/Label_diverse.txt",
    ),
    DatasetConfig(
        name="Oregon-1",
        graph_file=r"data/graphs/Oregon-1.txt",
        label_pool_file=r"data/labels/Label_diverse.txt",
    ),
    DatasetConfig(
        name="Oregon-2",
        graph_file=r"data/graphs/Oregon-2.txt",
        label_pool_file=r"data/labels/Label_diverse.txt",
    ),
    DatasetConfig(
        name="loc-Brightkite",
        graph_file=r"data/graphs/loc-Brightkite.txt",
        label_pool_file=r"data/labels/Label_diverse.txt",
    ),
]


@dataclass
class ExperimentConfig:
    # --- 查询/划分 ---
    num_queries: int = 50
    train_ratio: float = 0.7
    warmup: int = 10
    seed: int = 42

    # --- 并行/设备 ---
    threads: int = 8
    omp_threads: int = 8
    torch_threads: int = 8
    device: str = "cpu"          # "cpu" 或 "cuda"

    # --- 密码学参数 ---
    paillier_key_bits: int = 1024
    beaver_modulus_bits: int = 64

    # --- LSH (Simhash 多表) ---
    lsh_num_tables: int = 12      # ℓ: LSH表数
    lsh_num_functions: int = 6    # τ: 每表哈希函数数
    lsh_bits: int = 16            # simhash 位数

    # --- Node2Vec ---
    # p<1, q>1 倾向 BFS/结构等价，使 embedding 更反映拓扑距离而非纯社区同质性
    node2vec_walk_length: int = 120
    node2vec_num_walks: int = 15
    node2vec_p: float = 0.5
    node2vec_q: float = 2.0
    node2vec_dim: int = 64
    node2vec_window: int = 10
    node2vec_epochs: int = 10

    # --- 双线性模型 (M = A B^T, 低秩) ---
    bilinear_rank: int = 32
    bilinear_epochs: int = 200
    bilinear_lr: float = 0.005
    # 排序损失（三元组 margin ranking loss）
    bilinear_margin: float = 0.2
    bilinear_batch_size: int = 256

    # --- 3-hop 索引 ---
    three_hop_alpha: float = 0.5
    three_hop_k: int = 1000
    three_hop_max_distance: int = 100

    # --- 标签分配 ---
    labels_per_vertex_min: int = 2
    labels_per_vertex_max: int = 3

    # --- 候选/查询 ---
    max_candidates: int = 256
    k: int = 10
    lam: int = 1          # lambda 放大系数（λ=1 表示不放大，C2=top_k）

    # --- 参数扫描 ---
    k_values: List[int] = field(default_factory=lambda: [10, 20, 30])
    lambda_values: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    rank_values: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    dim_values: List[int] = field(default_factory=lambda: [32, 64, 128])


    # --- 输出 ---
    output_dir: str = "results"

    @property
    def beaver_modulus(self) -> int:
        return (1 << self.beaver_modulus_bits)

    def output_path(self, *parts: str) -> str:
        p = os.path.join(self.output_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


def default_config() -> ExperimentConfig:
    return ExperimentConfig()