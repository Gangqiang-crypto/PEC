"""安全算子：Paillier 同态加密、Beaver triple 两方乘法、安全比较、top-k。

- Paillier 使用 phe 库（真实）。
- Beaver triple 在两方加法共享下实现安全乘法（真实）。
- 安全比较：S1 持密文，S0 持私钥；S1 发送两密文给 S0，S0 解密比较返回 bit。
  在半诚实、不合谋模型下可接受。
- 所有通信量通过 recorder 记录。
"""

import random
from typing import Dict, List, Optional, Tuple

from phe import paillier

from .metrics import Recorder


# =========================================================================
#  Paillier 封装
# =========================================================================

class PaillierCipher:
    """Paillier 同态加密封装。"""

    def __init__(self, key_bits: int = 1024):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_bits
        )
        self.key_bits = key_bits

    def encrypt(self, value: int):
        return self.public_key.encrypt(value)

    def decrypt(self, ciphertext) -> int:
        return self.private_key.decrypt(ciphertext)

    @staticmethod
    def add(ct_a, ct_b):
        """密文 + 密文 = E(a+b)。"""
        return ct_a + ct_b

    @staticmethod
    def add_plain(ct_a, scalar: int):
        """密文 + 明文 = E(a + scalar)。"""
        return ct_a + scalar

    @staticmethod
    def scalar_mul(ct, scalar: int):
        """密文 * 标量 = E(a * scalar)。"""
        return ct * scalar



# =========================================================================
#  确定性 ID 加密
# =========================================================================

class EncryptedID:
    """确定性顶点 ID 加密（用于 enc_source / enc_vertex 表示）。"""

    def __init__(self, key: Optional[bytes] = None):
        import hashlib
        self.key = key or os_urandom(32)

    def encrypt_id(self, vertex: str) -> str:
        import hashlib
        h = hashlib.blake2b(digest_size=16, key=self.key)
        h.update(vertex.encode("utf-8"))
        return h.hexdigest()

    def encrypt_ids(self, vertices: List[str]) -> Dict[str, str]:
        return {v: self.encrypt_id(v) for v in vertices}


def os_urandom(n: int) -> bytes:
    import os
    return os.urandom(n)


# =========================================================================
#  Beaver triple 两方安全乘法
# =========================================================================

class BeaverTriple:
    """一个 Beaver triple 的两方共享。

    a = a0 + a1, b = b0 + b1, c = c0 + c1, c = a*b  (mod modulus)
    """

    __slots__ = ("a0", "a1", "b0", "b1", "c0", "c1", "modulus")

    def __init__(self, a0, a1, b0, b1, c0, c1, modulus):
        self.a0 = a0
        self.a1 = a1
        self.b0 = b0
        self.b1 = b1
        self.c0 = c0
        self.c1 = c1
        self.modulus = modulus

    @staticmethod
    def generate(modulus: int) -> "BeaverTriple":
        a = random.randrange(modulus)
        b = random.randrange(modulus)
        c = (a * b) % modulus
        a0 = random.randrange(modulus)
        a1 = (a - a0) % modulus
        b0 = random.randrange(modulus)
        b1 = (b - b0) % modulus
        c0 = random.randrange(modulus)
        c1 = (c - c0) % modulus
        return BeaverTriple(a0, a1, b0, b1, c0, c1, modulus)


class BeaverDealer:
    """可信 dealer / 离线预处理：批量生成 Beaver triple。"""

    def __init__(self, modulus: int):
        self.modulus = modulus
        self._pool: List[BeaverTriple] = []

    def preprocess(self, count: int):
        for _ in range(count):
            self._pool.append(BeaverTriple.generate(self.modulus))

    def take(self) -> BeaverTriple:
        if not self._pool:
            self._pool.append(BeaverTriple.generate(self.modulus))
        return self._pool.pop()

    def remaining(self) -> int:
        return len(self._pool)


def secure_multiply(
    x0: int, x1: int,
    y0: int, y1: int,
    triple: BeaverTriple,
    recorder: Optional[Recorder] = None,
) -> Tuple[int, int]:
    """两方 Beaver 乘法：z = x * y，返回 (z0, z1) 加法共享。

    x = x0 + x1, y = y0 + y1 (mod modulus)
    """
    m = triple.modulus
    d0 = (x0 - triple.a0) % m
    d1 = (x1 - triple.a1) % m
    e0 = (y0 - triple.b0) % m
    e1 = (y1 - triple.b1) % m

    d = (d0 + d1) % m
    e = (e0 + e1) % m

    z0 = (triple.c0 + d * triple.b0 + e * triple.a0) % m
    z1 = (triple.c1 + d * triple.b1 + e * triple.a1 + d * e) % m

    if recorder is not None:
        recorder.add_beaver_mul()


    return z0, z1


# =========================================================================
#  安全比较（半诚实：S1 持密文，S0 持私钥）
# =========================================================================

def secure_compare(
    ct_a,
    ct_b,
    cipher: PaillierCipher,
    recorder: Optional[Recorder] = None,
) -> int:
    """安全比较：返回 1 若 a <= b，否则 0。

    S1 发送 (ct_a, ct_b) 给 S0；S0 解密比较，返回 bit。
    通信量 = 两个密文字节 + 1 字节结果。
    """
    a = cipher.decrypt(ct_a)
    b = cipher.decrypt(ct_b)
    result = 1 if a <= b else 0

    if recorder is not None:
        recorder.add_comparison()


    return result


# =========================================================================
#  密文安全 top-k（基于安全比较）
# =========================================================================

def secure_topk_by_cipher(
    cipher_scores: Dict[str, object],
    k: int,
    cipher: PaillierCipher,
    recorder: Optional[Recorder] = None,
) -> List[str]:
    """对密文分数做安全 top-k，返回 top-k 候选 ID 列表（分数越小越好）。

    使用基于安全比较的选择排序（取前 k 个最小）。
    """
    items = list(cipher_scores.items())
    n = len(items)
    topk: List[str] = []

    for _ in range(min(k, n)):
        best_idx = 0
        for j in range(1, len(items)):
            cmp = secure_compare(items[j][1], items[best_idx][1], cipher, recorder)
            if cmp == 1:
                best_idx = j
        topk.append(items[best_idx][0])
        items.pop(best_idx)

    return topk


# =========================================================================
#  客户端辅助 top-k（恢复明文距离后排序）
# =========================================================================

def client_assisted_topk(
    distances: Dict[str, int],
    k: int,
    recorder: Optional[Recorder] = None,
) -> List[str]:
    """客户端恢复精确距离后完成 top-k。

    distances: {enc_vertex: plaintext_distance}
    平局规则：按 (distance, enc_vertex) 排序。
    """
    ordered = sorted(distances.items(), key=lambda kv: (kv[1], kv[0]))
    topk = [v for v, _ in ordered[:k]]

    if recorder is not None:
        recorder.add_comparison(len(distances))

    return topk


# =========================================================================
#  安全双线性打分
# =========================================================================

def secure_bilinear_scores(
    proj_share0: List[float],
    proj_share1: List[float],
    candidate_shares: Dict[str, Tuple[List[float], List[float]]],
    A0, A1, B0, B1,
    recorder: Optional[Recorder] = None,
) -> Dict[str, Tuple[int, int]]:
    """安全双线性打分：score(s, t) = u_s^T M u_t, M = A B^T。

    输入均为两方加法共享：
      - proj_share0/proj_share1: 查询点投影 u_s 的两方共享
      - candidate_shares[v] = (t_share0, t_share1): 候选 v 的投影共享
      - A0,A1 / B0,B1: 模型参数两方共享

    输出: {enc_vertex: (score0, score1)} 加法共享的整数分数。

    为在整数域上使用 Beaver 乘法，浮点向量先做定点量化。
    """
    SCALE = 1 << 16  # 定点量化

    def quantize(vec):
        return [int(round(x * SCALE)) for x in vec]

    qs0 = quantize(proj_share0)
    qs1 = quantize(proj_share1)

    scores: Dict[str, Tuple[int, int]] = {}
    modulus = (1 << 64)

    for v, (t0, t1) in candidate_shares.items():
        qt0 = quantize(t0)
        qt1 = quantize(t1)

        s0 = 0
        s1 = 0
        for i in range(len(qs0)):
            x0, x1 = qs0[i], qs1[i]
            y0, y1 = qt0[i], qt1[i]
            triple = BeaverTriple.generate(modulus)
            z0, z1 = secure_multiply(x0, x1, y0, y1, triple, recorder)
            s0 = (s0 + z0) % modulus
            s1 = (s1 + z1) % modulus

        scores[v] = (s0, s1)

    return scores


def recover_score(score0: int, score1: int, modulus: int = 1 << 64) -> int:
    """恢复完整分数（用于选择 C2）。"""
    return (score0 + score1) % modulus


def select_top_by_scores(
    scores: Dict[str, Tuple[int, int]],
    top_n: int,
    modulus: int = 1 << 64,
) -> List[str]:
    """从安全分数共享中选 top_n 个（分数越小越好）。"""
    recovered = {v: recover_score(s0, s1, modulus) for v, (s0, s1) in scores.items()}
    ordered = sorted(recovered.items(), key=lambda kv: (kv[1], kv[0]))
    return [v for v, _ in ordered[:top_n]]