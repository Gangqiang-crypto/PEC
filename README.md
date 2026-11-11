# Privacy-Preserving Top-k Query over Encrypted Graphs (Two-Server PEC)

A two-server architecture for privacy-preserving top-k keyword queries over encrypted labeled graphs. The proposed method (**PEC**) combines Locality-Sensitive Hashing (LSH) for candidate generation, a securely trained bilinear scoring model for candidate filtering, and Paillier homomorphic encryption for exact distance computation — reducing the number of costly exact distance evaluations while maintaining high recall.

## Architecture

```
                    ┌──────────────┐
                    │   Client (GU) │
                    │  Query Token  │
                    └──────┬───────┘
                           │ submit (C→S)
                    ┌──────▼───────┐
                    │  Server (S0)  │◄──── Beaver triples ────► Server (S1)
                    │  Paillier key │       (S↔S, PEC only)     │  Encrypted index
                    │  Bilinear A/B │                            │  Bilinear A/B shares
                    └──────┬───────┘                            └──────────┘
                           │ exact distances (S→C)
                    ┌──────▼───────┐
                    │   Client (GU) │
                    │  Final top-k  │
                    └──────────────┘
```

**PEC (two-server):** The client submits an encrypted query token. Server S0 holds the Paillier private key and bilinear model shares; Server S1 holds the encrypted index and bilinear model shares. The two servers collaboratively compute bilinear scores via Beaver triple multiplication to filter candidates (C1 → C2), then S0 returns encrypted exact distances for the top-kλ candidates. The client decrypts and selects the final top-k.

**Baselines (single-server):** SunFull and PathFQ operate on a single server — all LSH candidates proceed directly to exact distance computation without the bilinear filtering step.

## Methods

| Method | Architecture | Candidate Generation | Filtering | Exact Distance |
|--------|-------------|----------------------|-----------|----------------|
| **PEC** (ours) | Two-server | LSH intersection | Secure bilinear scoring → top-kλ | Only on C2 |
| **SunFull** | Single-server | LSH intersection | None | All candidates |
| **PathFQ** | Single-server | LSH table union (fuzzy) | None | All candidates |

### PEC Query Pipeline

1. **Candidate Generation (C1):** LSH Simhash multi-table intersection produces candidate vertices matching the query label. Candidates are deterministically truncated to `max_candidates` (256) via HMAC ordering.
2. **Secure Bilinear Scoring (C1 → C2):** A low-rank bilinear model `M = AB^T` scores each candidate's relevance to the query source. Model parameters and embeddings are split into additive shares across S0 and S1. Scoring uses Beaver triple multiplication in a 64-bit integer ring with fixed-point quantization (scale = 2^16). The top-kλ candidates (C2) are selected by recovered scores.
3. **Exact Distance (C2):** Paillier-encrypted shortest-path distances are computed for the filtered candidates only, using a 3-hop hub-labeling index.
4. **Client Top-k:** The client decrypts distances and selects the final top-k.

## Key Components

### `common/shared_index.py` — SharedEncryptedIndex
- **EncryptedLSHIndex:** Simhash-based multi-table LSH with `ℓ=12` tables, `τ=6` hash functions per table, 16-bit signatures. Supports intersection and table-union candidate modes.
- **ThreeHopIndex:** Hub-labeling distance index using centrality-ranked landmarks (top-k hubs) with Dijkstra-based distance tables. Provides `exact_distance(s, t)` via hub labeling.
- **PaillierCipher:** 1024-bit Paillier homomorphic encryption for distance encryption.
- **EncryptedID:** Deterministic BLAKE2b-based vertex ID encryption.

### `common/secure_ops.py` — Cryptographic Operations
- **BeaverTriple / BeaverDealer:** Two-party additive secret sharing with Beaver triple multiplication for secure bilinear scoring.
- **secure_bilinear_scores:** Computes `u_s^T M u_t` under two-party additive sharing with fixed-point quantization.
- **client_assisted_topk:** Client-side plaintext top-k selection after distance decryption.

### `common/token.py` — QueryToken
- Encapsulates encrypted source ID, label hash, LSH signatures, k, λ, and optional bilinear projection shares.
- Deterministic truncation via HMAC ensures consistent candidate ordering across methods.

### `common/metrics.py` — Recording & Communication Estimation
- **Recorder:** Per-query timer/counter for candidates, exact distances, Beaver multiplications, comparisons.
- **CommTracker:** Communication estimation across 4 stages (submit, prediction, exact, return) × 3 channels (C→S, S→C, S↔S).

### `methods/ours_bilinear.py` — PEC Method
- **BilinearModelBundle:** Holds additive shares of model parameters A, B and pre-computed candidate projections.
- **OursBilinearMethod:** Implements the full PEC pipeline (LSH → bilinear filter → exact distance → client top-k).

### `methods/sun_full.py` / `methods/pathfq_full.py` — Baselines
- Single-server methods that skip the bilinear filtering stage, sending all candidates directly to exact distance computation.

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `paillier_key_bits` | 1024 | Paillier key size |
| `beaver_modulus_bits` | 64 | Beaver triple ring size |
| `lsh_num_tables` (ℓ) | 12 | Number of LSH tables |
| `lsh_num_functions` (τ) | 6 | Hash functions per table |
| `lsh_bits` | 16 | Simhash signature width |
| `bilinear_rank` (r) | 32 | Low-rank factorization dimension |
| `bilinear_epochs` | 200 | Training epochs |
| `bilinear_margin` | 0.2 | Triplet margin ranking loss margin |
| `node2vec_dim` (d) | 64 | Landmark embedding dimension |
| `max_candidates` | 256 | Maximum LSH candidates |
| `k_values` | [10, 20, 30] | Top-k values to scan |
| `lambda_values` | [1, 3, 5, 7, 9] | Candidate amplification factors |
| `rank_values` | [8, 16, 32, 64] | Bilinear rank ablation |
| `dim_values` | [32, 64, 128] | Embedding dimension ablation |

## Datasets

Place graph edge-list files in `data/graphs/` and label pool files in `data/labels/`. Configure paths in `config.py`.

**Graph format:** Edge list, one edge per line (`u v` or `u v w`), `#` for comments.

**Label format:** One label word per line.

Pre-configured datasets: CA-HepTh, CA-CondMat, Oregon-1, Oregon-2.

## Usage

### Run experiments

```bash
python runner.py
```

This executes the full experiment matrix:
1. **Main experiment + k-scan:** k ∈ {10, 20, 30}, λ ∈ {1, 3, 5, 7, 9}, all 3 methods
2. **Rank ablation (r):** k ∈ {10, 20, 30}, λ ∈ {1, 3, 5, 7, 9}, r ∈ {8, 16, 64}, PEC only
3. **Dimension ablation (d):** k ∈ {10, 20, 30}, λ ∈ {1, 3, 5, 7, 9}, d ∈ {32, 128}, PEC only

LSH parameters (ℓ=12, τ=6) are fixed — no LSH ablation.

### Summarize results

```bash
python summarize_results.py [output_dir]
```

Aggregates `query_results.csv` and `init_results.csv` into `paper_metrics.csv` with per-group averages for query time, recall@k, candidate/selected counts, and communication estimates.

### Recall baseline

PathFQ's top-k results serve as ground truth (recall = 1.0). Other methods' recall = |result ∩ PathFQ result| / k.

## Output

Results are written to `results/`:
- `query_results.csv` — Per-query metrics (time, recall, candidate counts, communication)
- `init_results.csv` — Index/model build time and storage
- `paper_metrics.csv` — Aggregated summary table
- `summary.json` — Experiment configuration summary

## Dependencies

- `networkx` — Graph data structures and shortest-path algorithms
- `phe` — Paillier homomorphic encryption
- `torch` — Bilinear model training (triplet margin ranking loss)
- `numpy` — Numerical operations

## Project Structure

```
two_server_pec/
├── config.py              # Experiment configuration
├── runner.py              # Main experiment runner
├── summarize_results.py   # Result aggregation
├── common/
│   ├── metrics.py         # Recorder, QueryResult, CommTracker
│   ├── secure_ops.py      # Paillier, Beaver triples, secure bilinear scoring
│   ├── shared_index.py    # LSH, 3-hop index, SharedEncryptedIndex
│   └── token.py           # QueryToken, deterministic truncation
├── methods/
│   ├── base.py            # BaseTopKMethod interface
│   ├── ours_bilinear.py   # PEC (proposed method)
│   ├── sun_full.py        # SunFull baseline
│   └── pathfq_full.py     # PathFQ baseline
├── data/
│   ├── graphs/            # Graph edge-list files
│   ├── labels/            # Label pool files
│   └── README.md          # Data format documentation
└── results/               # Output directory (auto-created)
```