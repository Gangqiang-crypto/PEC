"""统一查询令牌 QueryToken 与令牌派生、确定性截断。

根令牌由 GU 在查询开始时构造一次，所有方法接收同一只读副本。
sun_full / aton_approx / pathfq_full 只使用 enc_source / label_hash / lsh_signatures。
ours_bilinear 额外使用 query_projection_shares。
"""

import hashlib
import hmac
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class QueryToken:
    qid: str
    enc_source: str                       # Enc(s)，源点密文 ID（字符串形式）
    label_hash: str                       # H(label)，标签摘要
    lsh_signatures: List[Tuple[int, int]] # [(table_id, bucket_id), ...]
    k: int
    lam: int                              # lambda 放大系数
    seed: int
    dataset_version: str
    query_projection_shares: Optional[Tuple[List[float], List[float]]] = None
    # 仅 ours_bilinear 使用：[u_s] 的两方加法共享 (share0, share1)
    query_label: Optional[str] = None
    # 仅离线真值预计算使用：查询的明文标签，不参与在线令牌哈希

    def token_hash(self) -> str:
        """所有方法接收相同令牌字节的审计哈希。"""
        h = hashlib.sha256()
        h.update(self.qid.encode())
        h.update(self.enc_source.encode())
        h.update(self.label_hash.encode())
        for t, b in self.lsh_signatures:
            h.update(f"{t}:{b}".encode())
        h.update(str(self.k).encode())
        h.update(str(self.lam).encode())
        h.update(str(self.seed).encode())
        h.update(self.dataset_version.encode())
        return h.hexdigest()

    def derived_token_hash(self, extra: str = "") -> str:
        """派生令牌审计哈希。"""
        h = hashlib.sha256()
        h.update(self.token_hash().encode())
        h.update(extra.encode())
        return h.hexdigest()


def hash_label(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def hmac_vertex(seed: int, enc_vertex: str) -> str:
    """确定性 HMAC，用于候选截断排序。"""
    key = seed.to_bytes(8, "big", signed=False)
    return hmac.new(key, enc_vertex.encode("utf-8"), hashlib.sha256).hexdigest()


def deterministic_truncate(
    candidates: List[str],
    max_candidates: int,
    seed: int,
) -> List[str]:
    """按 HMAC(seed, enc_vertex) 确定性排序后截断。

    所有方法使用同一截断顺序，避免随机采样差异。
    """
    if len(candidates) <= max_candidates:
        return list(candidates)
    ordered = sorted(candidates, key=lambda v: hmac_vertex(seed, v))
    return ordered[:max_candidates]


def derive_fuzzy_candidates(
    lsh_signatures: List[Tuple[int, int]],
    index,
) -> List[str]:
    """PathFQ 模糊候选派生：多表并集。"""
    seen = set()
    result = []
    for table_id, bucket_id in lsh_signatures:
        bucket = index.get_bucket(table_id, bucket_id)
        for v in bucket:
            if v not in seen:
                seen.add(v)
                result.append(v)
    return result