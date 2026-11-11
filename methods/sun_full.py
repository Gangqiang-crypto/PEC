"""sun_full：全候选精确查询基线。

保留 Sun 2021 的核心思想：语义候选全部进入精确距离阶段。
C1 = LSH intersection; 对全部 C1 执行 exact_distance; 客户端 top-k。
"""

from typing import Any

from common.metrics import (
    Recorder, QueryResult,
    estimate_submit_comm, estimate_exact_comm_s2c, estimate_return_comm_s2c,
)
from common.secure_ops import client_assisted_topk
from common.shared_index import SharedEncryptedIndex
from common.token import QueryToken, deterministic_truncate
from .base import BaseTopKMethod


class SunFullMethod(BaseTopKMethod):
    name = "sun_full"
    display_name = "SunFull"
    architecture = "single_server"

    def __init__(self):
        self.max_candidates: int = 256

    def prepare(self, shared_index: SharedEncryptedIndex, config: Any, **kwargs):
        self.max_candidates = config.max_candidates

    def query(
        self,
        token: QueryToken,
        shared_index: SharedEncryptedIndex,
        recorder: Recorder,
    ) -> QueryResult:
        with recorder.timer():
            C1 = shared_index.lsh_candidates(token, mode="intersection")
            C1 = deterministic_truncate(C1, self.max_candidates, token.seed)
            recorder.candidate_count = len(C1)
            recorder.selected_count = len(C1)

            cipher_distances = shared_index.exact_distance_many(token.enc_source, C1)
            recorder.add_exact_distance(len(C1))

            distances = {
                t: shared_index.decrypt_distance(ct)
                for t, ct in cipher_distances.items()
            }

            topk = client_assisted_topk(distances, token.k, recorder)

        truth = shared_index.get_ground_truth(token.qid)
        recall, jaccard = _recall_jaccard(topk, truth, token.k)
        return recorder.build_result(topk, recall, jaccard)

    def estimate_comm(
        self,
        recorder: Recorder,
        token: QueryToken,
        config: Any,
    ):
        L = config.lsh_num_tables
        key_bits = config.paillier_key_bits

        recorder.comm.add("submit", "client_to_server",
                          estimate_submit_comm(L, has_projection=False))
        recorder.comm.add("exact", "server_to_client",
                          estimate_exact_comm_s2c(recorder.exact_distance_count, key_bits))
        recorder.comm.add("return", "server_to_client",
                          estimate_return_comm_s2c(token.k))


def _recall_jaccard(pred: list, truth: list, k: int):
    if not truth:
        return 0.0, 0.0
    pred_set = set(pred)
    truth_set = set(truth)
    inter = pred_set & truth_set
    union = pred_set | truth_set
    recall = len(inter) / k if k > 0 else 0.0
    jaccard = len(inter) / len(union) if union else 0.0
    return recall, jaccard