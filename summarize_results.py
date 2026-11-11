"""聚合 query_results.csv / init_results.csv 为论文表格。

适配新CSV格式：包含 comm_*、saturated、pred_* 等字段。
输出 paper_metrics.csv（长表格式）和论文主表打印。
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def load_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(output_dir: str = "results"):
    query_path = os.path.join(output_dir, "query_results.csv")
    init_path = os.path.join(output_dir, "init_results.csv")

    queries = load_csv(query_path)
    init_rows = load_csv(init_path)

    if not queries:
        print(f"未找到 {query_path}，请先运行 runner.py")
        return

    groups = defaultdict(list)
    for row in queries:
        key = (row["dataset"], row["method"], row["k"], row["lambda"])
        groups[key].append(row)

    paper_rows = []

    print("\n" + "=" * 110)
    print("查询阶段主表")
    print("=" * 110)
    print(f"{'Dataset':<16} {'Method':<10} {'Arch':<14} {'k':>3} {'λ':>3} {'N':>4} "
          f"{'Avg(ms)':>10} {'Recall@k':>10} {'Cand':>6} {'Sel':>6} {'Sat%':>5} "
          f"{'C→S':>10} {'S→C':>10} {'S→S':>10}")
    print("-" * 110)

    for (dataset, method, k, lam), rows in sorted(groups.items()):
        n = len(rows)
        avg_ms = sum(float(r["query_time_ms"]) for r in rows) / n if n else 0
        recalls = [float(r["recall_at_k"]) for r in rows]
        mean_recall = sum(recalls) / len(recalls) if recalls else 0
        avg_cand = sum(int(r["candidate_count"]) for r in rows) / n if n else 0
        avg_sel = sum(int(r["selected_count"]) for r in rows) / n if n else 0
        sat_frac = sum(int(r["saturated"]) for r in rows) / n if n else 0
        avg_c2s = sum(int(r["comm_total_client_to_server"]) for r in rows) / n if n else 0
        avg_s2c = sum(int(r["comm_total_server_to_client"]) for r in rows) / n if n else 0
        avg_s2s = sum(int(r["comm_total_server_to_server"]) for r in rows) / n if n else 0
        arch = rows[0].get("architecture", "")

        print(f"{dataset:<16} {method:<10} {arch:<14} {k:>3} {lam:>3} {n:>4} "
              f"{avg_ms:>10.2f} {mean_recall:>10.4f} {avg_cand:>6.1f} {avg_sel:>6.1f} {sat_frac*100:>5.1f} "
              f"{avg_c2s:>10.0f} {avg_s2c:>10.0f} {avg_s2s:>10.0f}")

        for metric, val, unit in [
            ("QueryStageTimeAvg", avg_ms, "ms"),
            ("MeanRecallAtK", mean_recall, ""),
            ("AvgCandidateCount", avg_cand, ""),
            ("AvgSelectedCount", avg_sel, ""),
            ("SaturatedFrac", sat_frac, ""),
            ("AvgCommC2S", avg_c2s, "bytes"),
            ("AvgCommS2C", avg_s2c, "bytes"),
            ("AvgCommS2S", avg_s2s, "bytes"),
        ]:
            paper_rows.append({
                "stage": "query", "dataset": dataset, "method": method,
                "architecture": arch, "k": k, "lambda": lam,
                "metric": metric, "value": f"{val:.6f}", "unit": unit,
            })

    if init_rows:
        print("\n" + "=" * 80)
        print("初始化阶段")
        print("=" * 80)
        print(f"{'Dataset':<16} {'Component':<12} {'Param':<12} {'Time(s)':>10} {'Storage(MB)':>12}")
        print("-" * 80)

        for r in init_rows:
            print(f"{r['dataset']:<16} {r['component']:<12} {r.get('param_value',''):<12} "
                  f"{float(r['build_seconds']):>10.4f} {float(r['storage_mb']):>12.4f}")

            paper_rows.append({
                "stage": "init", "dataset": r["dataset"], "method": "",
                "architecture": "", "k": "", "lambda": "",
                "metric": f"{r['component']}BuildTime",
                "value": r["build_seconds"], "unit": "s",
            })
            paper_rows.append({
                "stage": "init", "dataset": r["dataset"], "method": "",
                "architecture": "", "k": "", "lambda": "",
                "metric": f"{r['component']}StorageMB",
                "value": r["storage_mb"], "unit": "MB",
            })

    paper_path = os.path.join(output_dir, "paper_metrics.csv")
    with open(paper_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stage", "dataset", "method", "architecture", "k", "lambda",
            "metric", "value", "unit",
        ])
        w.writeheader()
        w.writerows(paper_rows)

    print(f"\n论文指标已写入 {paper_path}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    summarize(output_dir)