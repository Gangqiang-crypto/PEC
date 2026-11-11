"""共享加密索引：LSH(Simhash 多表) + 3-hop 加密距离索引 + 统一查询接口。

所有方法必须调用 SharedEncryptedIndex 的接口，不得访问各自私有索引。
3-hop 构建参考用户提供的 ThreeHopIndexBuilder 思路（中心性排名 + 跳数扩展），
在此集成 networkx 输入并增加 Paillier 加密层。
"""

import hashlib
import heapq
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .secure_ops import PaillierCipher, EncryptedID
from .token import QueryToken


# =========================================================================
#  EncryptedLSHIndex — Simhash 多表 LSH
# =========================================================================

class EncryptedLSHIndex:
    """对顶点标签文本做 Simhash 签名，多表 LSH 候选生成。

    - L 个表，每表一个 bits 维随机超平面投影
    - 候选模式 intersection: 各表桶交集; table_union: 各表桶并集
    """

    FEATURE_DIM = 128

    def __init__(self, num_tables: int, num_functions: int, bits: int, seed: int):
        self.L = num_tables
        self.K = num_functions
        self.bits = bits
        self.rng = random.Random(seed)
        self.planes: List[List[List[float]]] = []
        for _ in range(self.L):
            table_planes = []
            for _ in range(self.bits):
                table_planes.append(
                    [self.rng.gauss(0, 1) for _ in range(self.FEATURE_DIM)]
                )
            self.planes.append(table_planes)
        self.tables: List[Dict[int, List[str]]] = [defaultdict(list) for _ in range(self.L)]
        self._vertex_labels: Dict[str, List[str]] = {}

    def _label_to_vector(self, labels: List[str]) -> List[float]:
        vec = [0.0] * self.FEATURE_DIM
        for label in labels:
            h = hashlib.md5(label.encode("utf-8")).digest()
            for i in range(self.FEATURE_DIM):
                vec[i] += (h[i % len(h)] / 255.0 - 0.5)
        return vec

    def _simhash_int(self, vec: List[float], table_id: int) -> int:
        bits_val = 0
        for b in range(self.bits):
            plane = self.planes[table_id][b]
            s = sum(v * p for v, p in zip(vec, plane))
            if s >= 0:
                bits_val |= (1 << b)
        return bits_val

    def build(self, vertex_labels: Dict[str, List[str]]):
        """对每个顶点的每个标签单独入桶，使单标签查询能检索到候选。"""
        self._vertex_labels = dict(vertex_labels)
        for v, labels in vertex_labels.items():
            for label in labels:
                vec = self._label_to_vector([label])
                for t in range(self.L):
                    sig = self._simhash_int(vec, t)
                    bucket = self.tables[t][sig]
                    if v not in bucket:
                        bucket.append(v)

    def query_signatures(self, labels: List[str]) -> List[Tuple[int, int]]:
        vec = self._label_to_vector(labels)
        return [(t, self._simhash_int(vec, t)) for t in range(self.L)]

    def get_bucket(self, table_id: int, bucket_id: int) -> List[str]:
        return self.tables[table_id].get(bucket_id, [])

    def candidates_intersection(self, signatures: List[Tuple[int, int]]) -> List[str]:
        sets = []
        for t, sig in signatures:
            sets.append(set(self.get_bucket(t, sig)))
        if not sets:
            return []
        result = sets[0]
        for s in sets[1:]:
            result = result & s
        return list(result)

    def candidates_union(self, signatures: List[Tuple[int, int]]) -> List[str]:
        result = set()
        for t, sig in signatures:
            for v in self.get_bucket(t, sig):
                result.add(v)
        return list(result)


# =========================================================================
#  ThreeHopIndex — 3-hop 距离索引（参考用户代码 + networkx 输入）
# =========================================================================

class ThreeHopIndex:
    """3-hop 距离索引。

    参考用户提供的 ThreeHopIndexBuilder：中心性排名 + 跳数扩展构建 I[v][w]。
    在此基础上：
      - 输入为 networkx 图
      - 构建距离表 distance_table[v][w] = d(v, w)
      - 提取 hub 集合（排名前 k 的顶点）
      - 提供 exact_distance(s, t) 与 common_hubs(s, t)
    """

    def __init__(self, alpha: float = 0.5, k: int = 1000, max_distance: int = 100):
        self.alpha = alpha
        self.k = k
        self.max_distance = max_distance
        self.graph: Optional[nx.Graph] = None
        self.ranking: Dict[str, int] = {}
        self.distance_table: Dict[str, Dict[str, float]] = {}
        self.hub_set: set = set()
        self.hub_list: List[str] = []

    def build(self, graph: nx.Graph):
        self.graph = graph
        self._compute_centrality_ranking()
        self._build_distance_table()

    def _compute_centrality_ranking(self):
        vertex_stats = {}
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree > 0:
                total_weight = sum(
                    data.get("weight", 1.0)
                    for _, _, data in self.graph.edges(node, data=True)
                )
                avg_weight = total_weight / degree
            else:
                avg_weight = 0.0
            vertex_stats[node] = (degree, avg_weight)

        sorted_vertices = sorted(
            vertex_stats.items(),
            key=lambda x: (x[1][0], x[1][1]),
            reverse=True,
        )
        self.ranking = {}
        for rank, (vertex, _) in enumerate(sorted_vertices):
            self.ranking[vertex] = len(sorted_vertices) - rank - 1

        self.hub_list = [v for v, _ in sorted_vertices[: self.k]]
        self.hub_set = set(self.hub_list)

    def _build_distance_table(self):
        """对每个顶点用 Dijkstra 计算到 hub 集合及近邻的距离。

        distance_table[v][w] = d(v, w)，其中 w ∈ hub_set ∪ neighbors(v) ∪ {v}
        这样 exact_distance 可通过 hub labeling 估计。
        """
        for v in self.graph.nodes():
            self.distance_table[v] = {v: 0.0}

        for v in self.graph.nodes():
            dist = self._dijkstra(v)
            row = self.distance_table[v]
            for target, d in dist.items():
                if d > self.max_distance:
                    continue
                if target in self.hub_set or target == v:
                    row[target] = d
            for nbr in self.graph.neighbors(v):
                if nbr not in row:
                    w = self.graph[v][nbr].get("weight", 1.0)
                    row[nbr] = w

    def _dijkstra(self, source) -> Dict[str, float]:
        dist = {source: 0.0}
        visited = set()
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if d > self.max_distance:
                continue
            for nbr in self.graph.neighbors(u):
                w = self.graph[u][nbr].get("weight", 1.0)
                nd = d + w
                if nd < dist.get(nbr, math.inf) and nd <= self.max_distance:
                    dist[nbr] = nd
                    heapq.heappush(heap, (nd, nbr))
        return dist

    def exact_distance(self, s: str, t: str) -> float:
        """通过 hub labeling 估计最短距离。

        d(s,t) = min(
            distance_table[s][t] if exists,
            min over h in common_hubs of d(s,h) + d(t,h)
        )
        """
        if s == t:
            return 0.0
        row_s = self.distance_table.get(s, {})
        if t in row_s:
            return row_s[t]
        row_t = self.distance_table.get(t, {})

        best = math.inf
        for h in self.hub_list:
            ds = row_s.get(h)
            if ds is None:
                continue
            dt = row_t.get(h)
            if dt is None:
                continue
            total = ds + dt
            if total < best:
                best = total
        return best

    def common_hubs(self, s: str, t: str) -> List[str]:
        row_s = self.distance_table.get(s, {})
        row_t = self.distance_table.get(t, {})
        return [h for h in self.hub_list if h in row_s and h in row_t]

    def hub_distance(self, s: str, h: str) -> Optional[float]:
        return self.distance_table.get(s, {}).get(h)


# =========================================================================
#  SharedEncryptedIndex — 统一共享加密索引
# =========================================================================

class SharedEncryptedIndex:
    """所有方法共享的加密索引。

    组成：
      - lsh_index: EncryptedLSHIndex
      - graph_index: ThreeHopIndex
      - paillier_key: PaillierCipher
      - encrypted_id: EncryptedID
      - label_map: 顶点 -> 标签列表
      - ground_truth_cache: 离线 top-k 真值
    """

    def __init__(
        self,
        paillier_key_bits: int = 1024,
        lsh_num_tables: int = 8,
        lsh_num_functions: int = 4,
        lsh_bits: int = 16,
        lsh_seed: int = 42,
        three_hop_alpha: float = 0.5,
        three_hop_k: int = 1000,
        three_hop_max_distance: int = 100,
    ):
        self.paillier = PaillierCipher(key_bits=paillier_key_bits)
        self.encrypted_id = EncryptedID()
        self.lsh_index = EncryptedLSHIndex(
            lsh_num_tables, lsh_num_functions, lsh_bits, lsh_seed
        )
        self.graph_index = ThreeHopIndex(
            alpha=three_hop_alpha,
            k=three_hop_k,
            max_distance=three_hop_max_distance,
        )
        self.label_map: Dict[str, List[str]] = {}
        self.graph: Optional[nx.Graph] = None
        self.enc_id_to_vertex: Dict[str, str] = {}
        self.vertex_to_enc_id: Dict[str, str] = {}
        self.ground_truth_cache: Dict[str, List[str]] = {}
        self.dataset_version: str = ""
        self.enc_id_build_seconds: float = 0.0
        self.lsh_build_seconds: float = 0.0
        self.graph_index_build_seconds: float = 0.0

    # --- 构建 ---

    def build(
        self,
        graph: nx.Graph,
        vertex_labels: Dict[str, List[str]],
        dataset_version: str = "",
    ):
        self.graph = graph
        self.label_map = dict(vertex_labels)
        self.dataset_version = dataset_version

        import time as _time
        _t0 = _time.perf_counter()
        for v in graph.nodes():
            enc = self.encrypted_id.encrypt_id(str(v))
            self.vertex_to_enc_id[str(v)] = enc
            self.enc_id_to_vertex[enc] = str(v)
        self.enc_id_build_seconds = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        self.lsh_index.build(vertex_labels)
        self.lsh_build_seconds = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        self.graph_index.build(graph)
        self.graph_index_build_seconds = _time.perf_counter() - _t0

    # --- 辅助 ---

    def enc(self, vertex: str) -> str:
        return self.vertex_to_enc_id.get(str(vertex), self.encrypted_id.encrypt_id(str(vertex)))

    def dec(self, enc_vertex: str) -> str:
        return self.enc_id_to_vertex.get(enc_vertex, enc_vertex)

    def num_vertices(self) -> int:
        return self.graph.number_of_nodes() if self.graph else 0

    # --- 统一查询接口 ---

    def lsh_candidates(self, token: QueryToken, mode: str = "intersection") -> List[str]:
        """返回候选顶点的 enc_id 列表。"""
        if mode == "table_union":
            vertices = self.lsh_index.candidates_union(token.lsh_signatures)
        else:
            vertices = self.lsh_index.candidates_intersection(token.lsh_signatures)
        return [self.enc(v) for v in vertices if v != self.dec(token.enc_source)]

    def get_bucket(self, table_id: int, bucket_id: int) -> List[str]:
        return self.lsh_index.get_bucket(table_id, bucket_id)

    def common_hubs(self, enc_source: str, enc_target: str) -> List[str]:
        s = self.dec(enc_source)
        t = self.dec(enc_target)
        hubs = self.graph_index.common_hubs(s, t)
        return [self.enc(h) for h in hubs]

    def encrypted_hub_distances(
        self, enc_source: str, enc_target: str
    ) -> Dict[str, object]:
        """返回 {enc_hub: E(d(s,h) + d(t,h))} 的 Paillier 密文。"""
        s = self.dec(enc_source)
        t = self.dec(enc_target)
        result: Dict[str, object] = {}
        for h in self.graph_index.common_hubs(s, t):
            ds = self.graph_index.hub_distance(s, h)
            dt = self.graph_index.hub_distance(t, h)
            if ds is None or dt is None:
                continue
            total = int(round(ds + dt))
            result[self.enc(h)] = self.paillier.encrypt(total)
        return result

    def exact_distance_cipher(self, enc_source: str, enc_target: str):
        """返回 E(d(s, t)) 的 Paillier 密文。"""
        s = self.dec(enc_source)
        t = self.dec(enc_target)
        d = self.graph_index.exact_distance(s, t)
        if d == math.inf:
            d = self.graph_index.max_distance * 10
        return self.paillier.encrypt(int(round(d)))

    def exact_distance_many(
        self, enc_source: str, targets: List[str]
    ) -> Dict[str, object]:
        """批量精确距离：{enc_target: E(d(s, t))}。"""
        return {t: self.exact_distance_cipher(enc_source, t) for t in targets}

    def decrypt_distance(self, ciphertext) -> int:
        """客户端解密距离。"""
        return self.paillier.decrypt(ciphertext)

    # --- 真值缓存（离线预计算，不参与在线计时）---

    def precompute_ground_truth(
        self,
        queries: List[Tuple[str, str]],
        k: int,
    ) -> Dict[str, List[str]]:
        """离线预计算精确 top-k 真值。

        queries: [(qid, enc_source, label)] 中提取 (qid, enc_source)
        此处用 networkx 真实最短路径作为真值（不通过 3-hop 近似）。
        """
        truth: Dict[str, List[str]] = {}
        for qid, enc_source, _label in queries:
            s = self.dec(enc_source)
            dist_map = self.graph_index._dijkstra(s)
            ordered = sorted(dist_map.items(), key=lambda kv: (kv[1], kv[0]))
            truth[qid] = [self.enc(v) for v, _ in ordered[:k]]
        self.ground_truth_cache.update(truth)
        return truth

    def get_ground_truth(self, qid: str) -> List[str]:
        return self.ground_truth_cache.get(qid, [])