"""统一实验入口 runner.py。

支持完整实验矩阵：
  1. 主实验 + k扫描：k={10,20,30}, λ={1,3,5,7,9}, 默认参数, 3方法
  2. 消融r：k={10,20,30}, λ={1,3,5,7,9}, r={8,16,64}, PEC only
  3. 消融d：k={10,20,30}, λ={1,3,5,7,9}, d={32,128}, PEC only

LSH参数固定：ℓ=12表, τ=6阈值（不做消融）。

召回率基准：PathFQ的top-k结果视为真值（recall=1.0），
其他方法的recall = |该方法结果 ∩ PathFQ结果| / k。

执行策略：完成当前数据集的预测模型训练后执行查询。

组件复用：
  - 2HCL索引：每图只构建一次
  - Paillier密钥：每图只生成一次
  - Landmark embedding：按d值缓存
  - 双线性模型：按(r,d)组合缓存
"""

import csv
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx

from config import DatasetConfig, ExperimentConfig, default_config, DATASETS
from common.metrics import Recorder, QueryResult, InitResult, storage_mb
from common.shared_index import SharedEncryptedIndex
from common.token import QueryToken, hash_label
from methods.base import BaseTopKMethod
from methods.sun_full import SunFullMethod
from methods.pathfq_full import PathFQFullMethod
from methods.ours_bilinear import OursBilinearMethod, BilinearModelBundle


# =========================================================================
#  数据加载
# =========================================================================

def load_graph(graph_file: str) -> nx.Graph:
    g = nx.Graph()
    with open(graph_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u = parts[0]
            v = parts[1]
            if len(parts) >= 3:
                try:
                    w = float(parts[2])
                except ValueError:
                    w = 1.0
            else:
                w = 1.0
            g.add_edge(u, v, weight=w)
    return g


def load_label_pool(label_file: str) -> List[str]:
    labels = []
    with open(label_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                labels.append(line)
    return labels


def assign_labels(vertices, label_pool, min_labels, max_labels, rng):
    vertex_labels = {}
    for v in vertices:
        n = rng.randint(min_labels, max_labels)
        n = min(n, len(label_pool))
        labels = rng.sample(label_pool, n)
        vertex_labels[v] = labels
    return vertex_labels


def train_test_split(vertices, ratio, rng):
    shuffled = list(vertices)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * ratio)
    return shuffled[:n_train], shuffled[n_train:]


# =========================================================================
#  Landmark embedding
# =========================================================================

def train_landmark_embeddings(graph, index, dim, rng):
    t0 = time.perf_counter()
    hubs = [str(h) for h in index.graph_index.hub_list[:dim]]
    if len(hubs) < dim:
        hub_set = set(hubs)
        rest = [str(v) for v in graph.nodes() if str(v) not in hub_set]
        rng.shuffle(rest)
        hubs = hubs + rest[:dim - len(hubs)]
    hubs = hubs[:dim]

    hub_dists = {}
    for h in hubs:
        try:
            hub_dists[h] = nx.single_source_dijkstra_path_length(graph, h, weight="weight")
        except nx.NodeNotFound:
            hub_dists[h] = {}

    fallback = float(index.graph_index.max_distance)
    embeddings = {}
    for v in graph.nodes():
        vs = str(v)
        vec = [float(hub_dists[h].get(vs, fallback)) for h in hubs]
        embeddings[vs] = vec

    for v, vec in embeddings.items():
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 1e-9:
            embeddings[v] = [x / norm for x in vec]

    train_seconds = time.perf_counter() - t0
    return embeddings, train_seconds


# =========================================================================
#  双线性模型训练
# =========================================================================

def train_bilinear_model(graph, embeddings, train_vertices, rank, dim, config, rng, index):
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    margin = config.bilinear_margin
    batch_size = config.bilinear_batch_size

    torch.manual_seed(config.seed)
    if config.device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    A = torch.randn(dim, rank, device=dev, requires_grad=True)
    B = torch.randn(dim, rank, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([A, B], lr=config.bilinear_lr, weight_decay=1e-4)

    dist_cache = {}
    cache_max_sources = 500

    def _dist_map(s):
        if s not in dist_cache:
            if len(dist_cache) >= cache_max_sources:
                dist_cache.clear()
            try:
                dist_cache[s] = nx.single_source_dijkstra_path_length(graph, s, weight="weight")
            except nx.NodeNotFound:
                dist_cache[s] = {}
        return dist_cache[s]

    n_triplets = min(max(15000, len(train_vertices) * 8), 60000)
    triplets = []
    attempts = 0
    max_attempts = n_triplets * 6
    while len(triplets) < n_triplets and attempts < max_attempts:
        attempts += 1
        s = rng.choice(train_vertices)
        dm = _dist_map(s)
        if len(dm) < 4:
            continue
        items = list(dm.items())
        items = [(v, d) for v, d in items if v != s]
        if len(items) < 4:
            continue
        items.sort(key=lambda kv: kv[1])
        near_pool = items[:max(4, len(items) // 4)]
        far_pool = items[-(max(4, len(items) // 4)):]
        t_near, d_near = rng.choice(near_pool)
        t_far, d_far = rng.choice(far_pool)
        if d_far - d_near < 1e-6:
            continue
        triplets.append((str(s), str(t_near), str(t_far)))

    dist_cache.clear()

    if not triplets:
        train_seconds = time.perf_counter() - t0
        return BilinearModelBundle(rank=rank), train_seconds

    emb_cache = {v: torch.tensor(embeddings[v], device=dev) for v in embeddings}

    def _score_batch(batch):
        us_list, ut_near_list, ut_far_list = [], [], []
        for s, tn, tf in batch:
            us_list.append(emb_cache.get(s))
            ut_near_list.append(emb_cache.get(tn))
            ut_far_list.append(emb_cache.get(tf))
        us = torch.stack(us_list)
        ut_near = torch.stack(ut_near_list)
        ut_far = torch.stack(ut_far_list)
        proj_s = us @ A
        proj_near = ut_near @ B
        proj_far = ut_far @ B
        score_near = (proj_s * proj_near).sum(dim=1)
        score_far = (proj_s * proj_far).sum(dim=1)
        return score_near, score_far

    n = len(triplets)
    for _ in range(config.bilinear_epochs):
        perm = torch.randperm(n).tolist()
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = [triplets[i] for i in idx]
            try:
                score_near, score_far = _score_batch(batch)
            except Exception:
                continue
            loss = F.relu(margin + score_near - score_far).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if dev.type == "cuda":
        torch.cuda.synchronize()

    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()

    A0 = (A_np / 2.0).tolist()
    A1 = (A_np / 2.0).tolist()
    B0 = (B_np / 2.0).tolist()
    B1 = (B_np / 2.0).tolist()

    candidate_projections = {}
    for v in graph.nodes():
        key = index.enc(str(v))
        uv = embeddings.get(str(v), [0.0] * dim)
        uv_t = torch.tensor(uv, device=dev)
        proj = (B.t() @ uv_t).detach().cpu().numpy()
        share0 = (proj / 2.0).tolist()
        share1 = (proj / 2.0).tolist()
        candidate_projections[key] = (share0, share1)

    train_seconds = time.perf_counter() - t0
    bundle = BilinearModelBundle(
        A0=A0, A1=A1, B0=B0, B1=B1,
        candidate_projections=candidate_projections,
        rank=rank,
    )
    return bundle, train_seconds


# =========================================================================
#  查询生成
# =========================================================================

def generate_query_pairs(test_vertices, vertex_labels, num_queries, rng):
    eligible = [v for v in test_vertices if len(vertex_labels.get(v, [])) > 0]
    if not eligible:
        return []
    pairs = []
    for _ in range(num_queries):
        source = rng.choice(eligible)
        labels = vertex_labels[source]
        label = rng.choice(labels)
        pairs.append((source, label))
    return pairs


def build_query_token(qid, source, label, index, config, embeddings, bilinear_bundle, k, lam):
    lsh_sigs = index.lsh_index.query_signatures([label])

    query_proj_shares = None
    if bilinear_bundle is not None and str(source) in embeddings:
        import numpy as np
        uv = embeddings[str(source)]
        A_np = np.array(bilinear_bundle.A0) + np.array(bilinear_bundle.A1)
        proj = A_np.T @ np.array(uv)
        share0 = (proj / 2.0).tolist()
        share1 = (proj / 2.0).tolist()
        query_proj_shares = (share0, share1)

    return QueryToken(
        qid=qid,
        enc_source=index.enc(str(source)),
        label_hash=hash_label(label),
        lsh_signatures=lsh_sigs,
        k=k,
        lam=lam,
        seed=config.seed,
        dataset_version=index.dataset_version,
        query_projection_shares=query_proj_shares,
        query_label=label,
    )


# =========================================================================
#  运行实验配置
# =========================================================================

def run_experiment_config(
    method, query_pairs, index, config,
    k, lam, r, d, lsh_tables, lsh_threshold,
    embeddings, bilinear_bundle, ground_truth_cache,
    warmup=3,
):
    results = []

    for i in range(min(warmup, len(query_pairs))):
        source, label = query_pairs[i]
        token = build_query_token(
            f"warmup{i}", source, label, index, config,
            embeddings, bilinear_bundle, k, lam,
        )
        index.ground_truth_cache[token.qid] = ground_truth_cache.get((f"q{i:04d}", k), [])
        rec = Recorder(
            method.display_name, token.qid, lam=lam,
            k=k, r=r, d=d, lsh_tables=lsh_tables, lsh_threshold=lsh_threshold,
        )
        if method.name == "ours_bilinear":
            method.prepare(index, config, model=bilinear_bundle)
        else:
            method.prepare(index, config)
        try:
            method.query(token, index, rec)
        except Exception:
            pass

    for i, (source, label) in enumerate(query_pairs):
        qid = f"q{i:04d}"
        token = build_query_token(
            qid, source, label, index, config,
            embeddings, bilinear_bundle, k, lam,
        )
        index.ground_truth_cache[qid] = ground_truth_cache.get((qid, k), [])

        rec = Recorder(
            method.display_name, qid, lam=lam,
            k=k, r=r, d=d, lsh_tables=lsh_tables, lsh_threshold=lsh_threshold,
        )

        if method.name == "ours_bilinear":
            method.prepare(index, config, model=bilinear_bundle)
        else:
            method.prepare(index, config)

        try:
            result = method.query(token, index, rec)
        except Exception as e:
            result = rec.build_result([], status=f"error:{e}")

        method.estimate_comm(rec, token, config)
        comm_dict = rec.comm.as_dict()
        for key, val in comm_dict.items():
            setattr(result, key, val)

        results.append(result)

    return results


def run_config_with_pathfq_baseline(
    query_pairs, index, config,
    k, lambda_values, r, d, lsh_tables, lsh_threshold,
    embeddings, bilinear_bundle, ground_truth_cache,
):
    """运行一个k配置：PathFQ和SunFull各只运行一次（与λ无关），PEC对所有λ运行。

    PathFQ的top-k结果存入ground_truth_cache作为真值，
    PathFQ自身recall设为1.0，其他方法recall = |结果 ∩ PathFQ结果| / k。
    """
    results = []

    pathfq_results = run_experiment_config(
        PathFQFullMethod(), query_pairs, index, config,
        k=k, lam=lambda_values[0], r=r, d=d,
        lsh_tables=lsh_tables, lsh_threshold=lsh_threshold,
        embeddings=embeddings, bilinear_bundle=bilinear_bundle,
        ground_truth_cache={},
    )
    for res in pathfq_results:
        ground_truth_cache[(res.qid, k)] = res.topk
        res.recall_at_k = 1.0
    results.extend(pathfq_results)

    sunfull_results = run_experiment_config(
        SunFullMethod(), query_pairs, index, config,
        k=k, lam=lambda_values[0], r=r, d=d,
        lsh_tables=lsh_tables, lsh_threshold=lsh_threshold,
        embeddings=embeddings, bilinear_bundle=bilinear_bundle,
        ground_truth_cache=ground_truth_cache,
    )
    results.extend(sunfull_results)

    for lam in lambda_values:
        pec_results = run_experiment_config(
            OursBilinearMethod(), query_pairs, index, config,
            k=k, lam=lam, r=r, d=d,
            lsh_tables=lsh_tables, lsh_threshold=lsh_threshold,
            embeddings=embeddings, bilinear_bundle=bilinear_bundle,
            ground_truth_cache=ground_truth_cache,
        )
        results.extend(pec_results)

    return results


# =========================================================================
#  CSV输出
# =========================================================================

CSV_FIELDS = [
    "dataset", "method", "architecture", "k", "lambda", "r", "d",
    "lsh_tables", "lsh_threshold",
    "qid", "query_time_ms",
    "candidate_count", "selected_count", "saturated",
    "exact_distance_count", "comparison_count", "beaver_multiplication_count",
    "recall_at_k",
    "pred_count", "pred_mae", "pred_rmse",
    "rounds",
    "comm_total_client_to_server", "comm_total_server_to_client", "comm_total_server_to_server",
]

ARCHITECTURE_MAP = {
    "SunFull": "single_server",
    "PathFQ": "single_server",
    "PEC": "two_server",
}


def write_query_results(results, dataset, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({
                "dataset": dataset,
                "method": r.method,
                "architecture": ARCHITECTURE_MAP.get(r.method, "unknown"),
                "k": r.k,
                "lambda": r.lam,
                "r": r.r,
                "d": r.d,
                "lsh_tables": r.lsh_tables,
                "lsh_threshold": r.lsh_threshold,
                "qid": r.qid,
                "query_time_ms": f"{r.total_ms:.4f}",
                "candidate_count": r.candidate_count,
                "selected_count": r.selected_count,
                "saturated": int(r.saturated),
                "exact_distance_count": r.exact_distance_count,
                "comparison_count": r.comparison_count,
                "beaver_multiplication_count": r.beaver_multiplication_count,
                "recall_at_k": f"{r.recall_at_k:.6f}",
                "pred_count": r.pred_count,
                "pred_mae": f"{r.pred_mae:.6f}",
                "pred_rmse": f"{r.pred_rmse:.6f}",
                "rounds": r.rounds,
                "comm_total_client_to_server": r.comm_total_client_to_server,
                "comm_total_server_to_client": r.comm_total_server_to_client,
                "comm_total_server_to_server": r.comm_total_server_to_server,
            })


def write_init_results(init_rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "component", "param_key", "param_value",
            "build_seconds", "storage_mb",
        ])
        w.writeheader()
        for row in init_rows:
            w.writerow(row)


# =========================================================================
#  单数据集实验
# =========================================================================

def run_dataset(ds, config):
    print(f"\n{'='*60}")
    print(f"数据集: {ds.name}")
    print(f"{'='*60}")

    rng = random.Random(config.seed)

    graph = load_graph(ds.graph_file)
    print(f"  顶点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")

    label_pool = load_label_pool(ds.label_pool_file)
    vertices = [str(v) for v in graph.nodes()]
    vertex_labels = assign_labels(
        vertices, label_pool,
        config.labels_per_vertex_min, config.labels_per_vertex_max, rng,
    )
    train_vertices, test_vertices = train_test_split(vertices, config.train_ratio, rng)

    index = SharedEncryptedIndex(
        paillier_key_bits=config.paillier_key_bits,
        lsh_num_tables=config.lsh_num_tables,
        lsh_num_functions=config.lsh_num_functions,
        lsh_bits=config.lsh_bits,
        lsh_seed=config.seed,
        three_hop_alpha=config.three_hop_alpha,
        three_hop_k=config.three_hop_k,
        three_hop_max_distance=config.three_hop_max_distance,
    )
    index.build(graph, vertex_labels, dataset_version=ds.name)
    print(f"  索引构建完成: {index.enc_id_build_seconds + index.lsh_build_seconds + index.graph_index_build_seconds:.2f}s")

    embeddings_cache = {}
    bilinear_cache = {}
    init_rows = []

    default_d = config.node2vec_dim
    default_r = config.bilinear_rank
    fixed_ell = config.lsh_num_tables
    fixed_tau = config.lsh_num_functions

    init_rows.append({
        "dataset": ds.name, "component": "2HCL",
        "param_key": "", "param_value": "",
        "build_seconds": f"{index.graph_index_build_seconds:.4f}",
        "storage_mb": f"{storage_mb(index.graph_index):.4f}",
    })
    init_rows.append({
        "dataset": ds.name, "component": "Paillier+EncID",
        "param_key": "", "param_value": "",
        "build_seconds": f"{index.enc_id_build_seconds:.4f}",
        "storage_mb": f"{storage_mb(index.paillier) + storage_mb(index.encrypted_id):.4f}",
    })
    init_rows.append({
        "dataset": ds.name, "component": "LSH",
        "param_key": "ell,tau", "param_value": f"{fixed_ell},{fixed_tau}",
        "build_seconds": f"{index.lsh_build_seconds:.4f}",
        "storage_mb": f"{storage_mb(index.lsh_index):.4f}",
    })

    query_pairs = generate_query_pairs(test_vertices, vertex_labels, config.num_queries, rng)
    print(f"  查询数: {len(query_pairs)}")

    ground_truth_cache = {}
    all_results = []
    lambda_values = config.lambda_values
    k_values = config.k_values

    # ====== 默认 embedding + 双线性模型训练 ======
    print("训练 Landmark embedding (d=64)...")
    emb, emb_t = train_landmark_embeddings(graph, index, default_d, rng)
    embeddings_cache[default_d] = (emb, emb_t)
    init_rows.append({
        "dataset": ds.name, "component": "Landmark",
        "param_key": "d", "param_value": str(default_d),
        "build_seconds": f"{emb_t:.4f}",
        "storage_mb": f"{storage_mb(emb):.4f}",
    })

    print("训练双线性模型 (r=32, d=64)...")
    bundle, bil_t = train_bilinear_model(
        graph, emb, train_vertices, default_r, default_d, config, rng, index,
    )
    bilinear_cache[(default_r, default_d)] = (bundle, bil_t)
    init_rows.append({
        "dataset": ds.name, "component": "Bilinear",
        "param_key": "r,d", "param_value": f"{default_r},{default_d}",
        "build_seconds": f"{bil_t:.4f}",
        "storage_mb": f"{storage_mb(bundle):.4f}",
    })

    # ====== 主实验 + k扫描 ======
    print(f"\n--- 主实验 + k扫描 (k={k_values}, λ={lambda_values}) ---")
    for k in k_values:
        print(f"  k={k}: PathFQ(1次) → SunFull(1次) → PEC(所有λ)")
        results = run_config_with_pathfq_baseline(
            query_pairs, index, config,
            k=k, lambda_values=lambda_values, r=default_r, d=default_d,
            lsh_tables=fixed_ell, lsh_threshold=fixed_tau,
            embeddings=embeddings_cache[default_d][0],
            bilinear_bundle=bilinear_cache[(default_r, default_d)][0],
            ground_truth_cache=ground_truth_cache,
        )
        all_results.extend(results)

    # ====== 消融r：每训练完一个r，立即运行该r的所有k和λ查询 ======
    for r in [8, 16, 64]:
        print(f"\n训练双线性模型 (r={r}, d={default_d})...")
        b, t = train_bilinear_model(
            graph, embeddings_cache[default_d][0], train_vertices,
            r, default_d, config, rng, index,
        )
        bilinear_cache[(r, default_d)] = (b, t)
        init_rows.append({
            "dataset": ds.name, "component": "Bilinear",
            "param_key": "r,d", "param_value": f"{r},{default_d}",
            "build_seconds": f"{t:.4f}",
            "storage_mb": f"{storage_mb(b):.4f}",
        })

        print(f"--- 消融r: PEC r={r}, k={k_values}, λ={lambda_values} ---")
        for k in k_values:
            for lam in lambda_values:
                print(f"  PEC r={r} k={k} λ={lam}")
                results = run_experiment_config(
                    OursBilinearMethod(), query_pairs, index, config,
                    k=k, lam=lam, r=r, d=default_d,
                    lsh_tables=fixed_ell, lsh_threshold=fixed_tau,
                    embeddings=embeddings_cache[default_d][0],
                    bilinear_bundle=bilinear_cache[(r, default_d)][0],
                    ground_truth_cache=ground_truth_cache,
                )
                all_results.extend(results)

    # ====== 消融d：每训练完一个d的embedding+模型，立即运行该d的所有k和λ查询 ======
    for d_val in [32, 128]:
        print(f"\n训练 Landmark embedding (d={d_val})...")
        e, t = train_landmark_embeddings(graph, index, d_val, rng)
        embeddings_cache[d_val] = (e, t)
        init_rows.append({
            "dataset": ds.name, "component": "Landmark",
            "param_key": "d", "param_value": str(d_val),
            "build_seconds": f"{t:.4f}",
            "storage_mb": f"{storage_mb(e):.4f}",
        })

        print(f"训练双线性模型 (r={default_r}, d={d_val})...")
        b, t2 = train_bilinear_model(
            graph, embeddings_cache[d_val][0], train_vertices,
            default_r, d_val, config, rng, index,
        )
        bilinear_cache[(default_r, d_val)] = (b, t2)
        init_rows.append({
            "dataset": ds.name, "component": "Bilinear",
            "param_key": "r,d", "param_value": f"{default_r},{d_val}",
            "build_seconds": f"{t2:.4f}",
            "storage_mb": f"{storage_mb(b):.4f}",
        })

        print(f"--- 消融d: PEC d={d_val}, k={k_values}, λ={lambda_values} ---")
        for k in k_values:
            for lam in lambda_values:
                print(f"  PEC d={d_val} k={k} λ={lam}")
                results = run_experiment_config(
                    OursBilinearMethod(), query_pairs, index, config,
                    k=k, lam=lam, r=default_r, d=d_val,
                    lsh_tables=fixed_ell, lsh_threshold=fixed_tau,
                    embeddings=embeddings_cache[d_val][0],
                    bilinear_bundle=bilinear_cache[(default_r, d_val)][0],
                    ground_truth_cache=ground_truth_cache,
                )
                all_results.extend(results)

    write_query_results(all_results, ds.name, config.output_path(ds.name, "query_results.csv"))
    write_init_results(init_rows, config.output_path(ds.name, "init_results.csv"))

    return all_results, init_rows


# =========================================================================
#  主入口
# =========================================================================

def main():
    config = default_config()
    os.makedirs(config.output_dir, exist_ok=True)

    all_results = []
    all_init_rows = []

    for ds in DATASETS:
        if not os.path.exists(ds.graph_file):
            print(f"跳过 {ds.name}: 图文件不存在")
            continue
        if not os.path.exists(ds.label_pool_file):
            print(f"跳过 {ds.name}: 标签文件不存在")
            continue

        results, init_rows = run_dataset(ds, config)
        all_results.extend(results)
        all_init_rows.extend(init_rows)

    if not all_results:
        print("没有可运行的数据集。")
        return

    write_query_results(all_results, "ALL", config.output_path("query_results.csv"))
    write_init_results(all_init_rows, config.output_path("init_results.csv"))

    summary = {
        "total_queries": len(all_results),
        "datasets": [ds.name for ds in DATASETS if os.path.exists(ds.graph_file)],
        "methods": ["SunFull", "PathFQ", "PEC"],
        "config": {
            "num_queries": config.num_queries,
            "k_values": config.k_values,
            "lambda_values": config.lambda_values,
            "default_r": config.bilinear_rank,
            "default_d": config.node2vec_dim,
            "lsh_num_tables": config.lsh_num_tables,
            "lsh_num_functions": config.lsh_num_functions,
        },
    }
    with open(config.output_path("summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n完成。结果写入 {config.output_dir}/")


if __name__ == "__main__":
    main()