"""ours_bilinear：双线性预筛 + 精确 top-k（本文方案）。

C1 = LSH intersection; 双线性模型安全打分; C2 = top_{kλ};
仅对 C2 执行 exact_distance; 客户端完成最终 top-k。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from common.metrics import (
    Recorder, QueryResult,
    estimate_submit_comm, estimate_prediction_comm_s2s,
    estimate_prediction_comm_s2c, estimate_exact_comm_s2c,
    estimate_return_comm_s2c,
)
from common.secure_ops import (
    client_assisted_topk,
    secure_bilinear_scores,
    select_top_by_scores,
)
from common.shared_index import SharedEncryptedIndex
from common.token import QueryToken, deterministic_truncate
from .base import BaseTopKMethod
from .sun_full import _recall_jaccard


@dataclass
class BilinearModelBundle:
    """双线性模型两方共享包。

    M = A B^T，A/B 各拆为两方加法共享。
    candidate_projections[enc_vertex] = (proj_share0, proj_share1)
    query 投影由 token.query_projection_shares 提供。
    """

    A0: List[List[float]] = field(default_factory=list)
    A1: List[List[float]] = field(default_factory=list)
    B0: List[List[float]] = field(default_factory=list)
    B1: List[List[float]] = field(default_factory=list)
    candidate_projections: Dict[str, Tuple[List[float], List[float]]] = field(default_factory=dict)
    rank: int = 8


class OursBilinearMethod(BaseTopKMethod):
    name = "ours_bilinear"
    display_name = "PEC"
    architecture = "two_server"

    def __init__(self):
        self.max_candidates: int = 256
        self.model: Optional[BilinearModelBundle] = None
        self.beaver_modulus: int = 1 << 64

    def prepare(
        self,
        shared_index: SharedEncryptedIndex,
        config: Any,
        model: Optional[BilinearModelBundle] = None,
        **kwargs,
    ):
        self.max_candidates = config.max_candidates
        self.model = model
        self.beaver_modulus = config.beaver_modulus

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

            if token.query_projection_shares is None or self.model is None:
                C2 = C1
                recorder.selected_count = len(C2)
            else:
                proj_share0, proj_share1 = token.query_projection_shares

                cand_shares: Dict[str, Tuple[List[float], List[float]]] = {}
                for enc_v in C1:
                    shares = self.model.candidate_projections.get(enc_v)
                    if shares is None:
                        shares = ([0.0] * len(proj_share0), [0.0] * len(proj_share0))
                    cand_shares[enc_v] = shares

                scores = secure_bilinear_scores(
                    proj_share0,
                    proj_share1,
                    cand_shares,
                    self.model.A0, self.model.A1,
                    self.model.B0, self.model.B1,
                    recorder,
                )

                k_lam = min(token.k * token.lam, len(C1))
                C2 = select_top_by_scores(scores, k_lam, self.beaver_modulus)
                recorder.selected_count = len(C2)

            cipher_distances = shared_index.exact_distance_many(token.enc_source, C2)
            recorder.add_exact_distance(len(C2))

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
        rank = self.model.rank if self.model else config.bilinear_rank
        modulus_bits = config.beaver_modulus_bits

        has_proj = token.query_projection_shares is not None and self.model is not None

        recorder.comm.add("submit", "client_to_server",
                          estimate_submit_comm(L, has_projection=has_proj, dim=rank))

        if has_proj:
            recorder.comm.add("prediction", "server_to_server",
                              estimate_prediction_comm_s2s(
                                  recorder.candidate_count, rank, modulus_bits))
            recorder.comm.add("prediction", "server_to_client",
                              estimate_prediction_comm_s2c(recorder.selected_count))

        recorder.comm.add("exact", "server_to_client",
                          estimate_exact_comm_s2c(recorder.exact_distance_count, key_bits))
        recorder.comm.add("return", "server_to_client",
                          estimate_return_comm_s2c(token.k))