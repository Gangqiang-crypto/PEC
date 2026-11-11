"""指标记录：QueryResult、CommTracker、Recorder。

Recorder 只记录计时和计数，不接触明文 oracle。
所有时间以秒/毫秒记录，存储/通信以字节记录（输出时按 MB=10^6 bytes 换算）。
通信量内部按4阶段(submit/prediction/exact/return) × 3通道(C→S/S→C/S→S)记录，
输出时只保留3通道总量。
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =========================================================================
#  通信量跟踪器
# =========================================================================

STAGES = ("submit", "prediction", "exact", "return")
CHANNELS = ("client_to_server", "server_to_client", "server_to_server")


class CommTracker:
    """按4阶段×3通道记录通信字节数。

    阶段:
      - submit:     查询提交（客户端→服务器发送查询令牌）
      - prediction: 双线性预测阶段（仅PEC，Beaver乘法通信）
      - exact:      精确距离计算（服务器→客户端发送加密距离密文）
      - return:     结果返回（服务器→客户端返回top-k）

    通道:
      - client_to_server: 客户端发给服务器
      - server_to_client: 服务器发给客户端
      - server_to_server: 服务器间通信（单服务器方法恒为0）
    """

    def __init__(self):
        self._bytes: Dict[str, Dict[str, int]] = {
            s: {c: 0 for c in CHANNELS} for s in STAGES
        }

    def add(self, stage: str, channel: str, nbytes: int):
        if stage in self._bytes and channel in self._bytes[stage]:
            self._bytes[stage][channel] += nbytes

    def get(self, stage: str, channel: str) -> int:
        return self._bytes.get(stage, {}).get(channel, 0)

    def total(self, channel: str) -> int:
        return sum(self._bytes[s].get(channel, 0) for s in STAGES)

    def stage_total(self, stage: str) -> int:
        return sum(self._bytes[stage].values())

    def reset(self):
        self._bytes = {s: {c: 0 for c in CHANNELS} for s in STAGES}

    def as_dict(self) -> Dict[str, int]:
        result = {}

        for c in CHANNELS:
            result[f"comm_total_{c}"] = self.total(c)
        return result


# =========================================================================
#  通信量估算常量与辅助函数
# =========================================================================

def paillier_ciphertext_size(key_bits: int = 1024) -> int:
    """Paillier密文大小（字节）。密文在 Z_{n^2} 中，大小 = 2 * key_bits / 8。"""
    return 2 * key_bits // 8


def enc_id_size() -> int:
    """加密顶点ID大小（字节）。BLAKE2b 16字节digest → 32 hex chars。"""
    return 32


def label_hash_size() -> int:
    """标签哈希大小（字节）。SHA256 → 64 hex chars。"""
    return 64


def beaver_value_size(modulus_bits: int = 64) -> int:
    """Beaver乘法中每个值的字节大小。"""
    return modulus_bits // 8


def estimate_submit_comm(
    num_lsh_tables: int,
    has_projection: bool = False,
    dim: int = 0,
) -> int:
    """估算submit阶段 C→S 通信量。

    enc_source(32) + label_hash(64) + L*(4+4) + k(4) + lambda(4) + seed(4) + dataset_version(20)
    PEC额外: query_projection_shares = dim * 8 * 2 (两方各dim个int64)
    """
    base = enc_id_size() + label_hash_size() + num_lsh_tables * 8 + 4 + 4 + 4 + 20
    if has_projection:
        base += dim * beaver_value_size() * 2
    return base


def estimate_prediction_comm_s2s(
    candidate_count: int,
    dim: int,
    modulus_bits: int = 64,
) -> int:
    """估算prediction阶段 S→S 通信量（Beaver乘法）。

    每次乘法: 两方各发送2个值(d_i, e_i)，每个值 modulus_bits/8 字节
    每次乘法双向通信 = 2 * 2 * (modulus_bits/8) = 4 * modulus_bits/8
    总乘法数 = candidate_count * dim
    """
    per_mul = 4 * beaver_value_size(modulus_bits)
    return candidate_count * dim * per_mul


def estimate_prediction_comm_s2c(selected_count: int) -> int:
    """估算prediction阶段 S→C 通信量（返回C2列表）。"""
    return selected_count * enc_id_size()


def estimate_exact_comm_s2c(
    exact_distance_count: int,
    key_bits: int = 1024,
) -> int:
    """估算exact阶段 S→C 通信量（加密距离密文）。"""
    return exact_distance_count * paillier_ciphertext_size(key_bits)


def estimate_return_comm_s2c(k: int) -> int:
    """估算return阶段 S→C 通信量（top-k结果）。"""
    return k * enc_id_size()


# =========================================================================
#  QueryResult
# =========================================================================

@dataclass
class QueryResult:
    method: str
    qid: str
    candidate_count: int = 0
    selected_count: int = 0
    exact_distance_count: int = 0
    comparison_count: int = 0
    beaver_multiplication_count: int = 0
    fallback_count: int = 0
    total_seconds: float = 0.0

    topk: List[str] = field(default_factory=list)
    recall_at_k: float = 0.0
    jaccard_at_k: float = 0.0
    status: str = "ok"
    lam: int = 1

    saturated: bool = False
    pred_count: int = 0
    pred_mae: float = 0.0
    pred_rmse: float = 0.0
    rounds: int = 0

    k: int = 10
    r: int = 32
    d: int = 64
    lsh_tables: int = 12
    lsh_threshold: int = 6


    comm_total_client_to_server: int = 0
    comm_total_server_to_client: int = 0
    comm_total_server_to_server: int = 0

    @property
    def total_ms(self) -> float:
        return self.total_seconds * 1000.0


# =========================================================================
#  InitResult
# =========================================================================

@dataclass
class InitResult:
    method: str
    dataset: str
    variant: str = "default"
    seed: int = 42
    threads: int = 8
    device: str = "cpu"
    gpu_model: str = ""
    cuda_version: str = ""
    dim: int = 64
    rank: int = 32

    lsh_build_seconds: float = 0.0
    lsh_storage_mb: float = 0.0
    hcl_build_seconds: float = 0.0
    hcl_storage_mb: float = 0.0
    enc_index_build_seconds: float = 0.0
    enc_index_storage_mb: float = 0.0

    node2vec_train_seconds: float = 0.0
    embedding_storage_mb: float = 0.0
    bilinear_train_seconds: float = 0.0
    bilinear_model_storage_mb: float = 0.0
    model_train_seconds: float = 0.0
    model_storage_mb: float = 0.0

    method_prep_seconds: float = 0.0
    method_prep_storage_mb: float = 0.0
    paillier_keygen_seconds: float = 0.0

    gpu_peak_memory_mb: float = 0.0

    @property
    def total_init_seconds(self) -> float:
        return (
            self.enc_index_build_seconds
            + self.model_train_seconds
            + self.method_prep_seconds
        )

    @property
    def total_init_storage_mb(self) -> float:
        return (
            self.enc_index_storage_mb
            + self.model_storage_mb
            + self.method_prep_storage_mb
        )


# =========================================================================
#  Recorder
# =========================================================================

class Recorder:
    """单次查询的计时/计数/通信量记录器。"""

    def __init__(self, method: str, qid: str, lam: int = 1,
                 k: int = 10, r: int = 32, d: int = 64,
                 lsh_tables: int = 12, lsh_threshold: int = 6):
        self.method = method
        self.qid = qid
        self.lam = lam
        self.k = k
        self.r = r
        self.d = d
        self.lsh_tables = lsh_tables
        self.lsh_threshold = lsh_threshold
        self._t0: Optional[float] = None
        self.candidate_count = 0
        self.selected_count = 0
        self.exact_distance_count = 0
        self.comparison_count = 0
        self.beaver_multiplication_count = 0
        self.fallback_count = 0

        self.elapsed = 0.0

        self.comm = CommTracker()
        self.pred_count = 0
        self.pred_errors: List[float] = []
        self.rounds = 0

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        if self._t0 is not None:
            self.elapsed = time.perf_counter() - self._t0
            self._t0 = None

    @contextmanager
    def timer(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def add_comparison(self, n: int = 1):
        self.comparison_count += n

    def add_beaver_mul(self, n: int = 1):
        self.beaver_multiplication_count += n

    def add_exact_distance(self, n: int = 1):
        self.exact_distance_count += n

    def add_fallback(self, n: int = 1):
        self.fallback_count += n

    def add_pred_error(self, predicted: float, actual: float):
        self.pred_count += 1
        err = abs(predicted - actual)
        self.pred_errors.append(err)

    def add_rounds(self, n: int = 1):
        self.rounds += n

    @property
    def pred_mae(self) -> float:
        if not self.pred_errors:
            return 0.0
        return sum(self.pred_errors) / len(self.pred_errors)

    @property
    def pred_rmse(self) -> float:
        if not self.pred_errors:
            return 0.0
        import math
        return math.sqrt(sum(e * e for e in self.pred_errors) / len(self.pred_errors))

    @property
    def saturated(self) -> bool:
        return self.selected_count == self.candidate_count

    def build_result(
        self,
        topk: List[str],
        recall_at_k: float = 0.0,
        jaccard_at_k: float = 0.0,
        status: str = "ok",
    ) -> QueryResult:
        comm_dict = self.comm.as_dict()
        return QueryResult(
            method=self.method,
            qid=self.qid,
            candidate_count=self.candidate_count,
            selected_count=self.selected_count,
            exact_distance_count=self.exact_distance_count,
            comparison_count=self.comparison_count,
            beaver_multiplication_count=self.beaver_multiplication_count,
            fallback_count=self.fallback_count,
            total_seconds=self.elapsed,

            topk=topk,
            recall_at_k=recall_at_k,
            jaccard_at_k=jaccard_at_k,
            status=status,
            lam=self.lam,

            saturated=self.saturated,
            pred_count=self.pred_count,
            pred_mae=self.pred_mae,
            pred_rmse=self.pred_rmse,
            rounds=self.rounds,

            k=self.k,
            r=self.r,
            d=self.d,
            lsh_tables=self.lsh_tables,
            lsh_threshold=self.lsh_threshold,


            comm_total_client_to_server=comm_dict["comm_total_client_to_server"],
            comm_total_server_to_client=comm_dict["comm_total_server_to_client"],
            comm_total_server_to_server=comm_dict["comm_total_server_to_server"],
        )


def storage_mb(obj) -> float:
    """估算 Python 对象序列化大小（MB）。"""
    import pickle
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6
    except Exception:
        return 0.0