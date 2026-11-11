"""pathfq_full：模糊候选 + 全候选精确距离基线。

保留 PathFQ-GES 2025 的模糊查询思想，扩展为 top-k。
C1_fuzzy = LSH table_union（模糊扩张）; 对全部候选执行 exact_distance; 客户端 top-k。
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
from .sun_full import _recall_jaccard


class PathFQFullMethod(BaseTopKMethod):
    name = "pathfq_full"
    display_name = "PathFQ"
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
            C1_fuzzy = shared_index.lsh_candidates(token, mode="table_union")
            C1_fuzzy = deterministic_truncate(C1_fuzzy, self.max_candidates, token.seed)
            recorder.candidate_count = len(C1_fuzzy)
            recorder.selected_count = len(C1_fuzzy)

            cipher_distances = shared_index.exact_distance_many(
                token.enc_source, C1_fuzzy
            )
            recorder.add_exact_distance(len(C1_fuzzy))

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