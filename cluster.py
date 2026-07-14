"""
Memory Fragment Clustering — standalone module.

Takes a list of memory fragments (text + embedding) and discovers
narrative clusters using Leiden community detection + LLM rerank.

Dependencies: numpy, python-igraph, leidenalg
Optional: an OpenAI-compatible LLM API for rerank (DeepSeek, OpenAI, etc.)
"""

import json
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    id: str
    text: str
    embedding: np.ndarray
    importance: float = 5.0
    created_at: str = ""
    tags: str = ""


@dataclass
class Cluster:
    fragments: list[Fragment]
    total_importance: float = 0.0
    rerank_scores: dict = field(default_factory=dict)   # id -> 0/1/2


# ---------------------------------------------------------------------------
# Leiden clustering
# ---------------------------------------------------------------------------

def discover_clusters(
    fragments: list[Fragment],
    *,
    threshold: float = 0.50,
    max_cluster_size: int = 10,
    resolution: float = 6.0,
    top_n: int = 1,
) -> list[Cluster]:
    """
    Cluster fragments by embedding similarity using Leiden community detection.

    Args:
        fragments: list of Fragment objects (must have embeddings)
        threshold: minimum cosine similarity to keep an edge in the k-NN graph.
                   Lower = more connections = bigger clusters
        max_cluster_size: cap per cluster (keeps the top-importance fragments)
        resolution: Leiden resolution parameter.
                    ~1.0 → few large theme-level blobs
                    ~6.0 → many small event-level clusters (recommended)
        top_n: how many top clusters to return (ranked by total importance)

    Returns:
        list of Cluster objects, sorted by total importance descending
    """
    import igraph as ig
    import leidenalg

    if len(fragments) < 2:
        return []

    # Normalize embeddings and compute cosine similarity matrix
    embs = np.array([f.embedding for f in fragments])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embs / norms
    sim_matrix = normed @ normed.T

    # Build k-NN graph (k=10, edges below threshold discarded)
    n = len(fragments)
    k = min(10, n - 1)
    edges = {}
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0
        if n - 1 > k:
            nbrs = np.argpartition(-sims, k)[:k]
        else:
            nbrs = [j for j in range(n) if j != i]
        for j in nbrs:
            j = int(j)
            w = float(sim_matrix[i][j])
            if w < threshold:
                continue
            key = (i, j) if i < j else (j, i)
            if w > edges.get(key, 0.0):
                edges[key] = w

    if not edges:
        return []

    g = ig.Graph(n=n, edges=list(edges.keys()))
    g.es["weight"] = [edges[e] for e in edges.keys()]
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=42,
    )
    components = [list(c) for c in partition if len(c) >= 2]

    if not components:
        return []

    # Rank by total importance, take top_n
    def score(comp):
        return sum(fragments[i].importance for i in comp)

    components.sort(key=score, reverse=True)
    results = []
    for comp in components[:top_n]:
        if len(comp) > max_cluster_size:
            comp.sort(key=lambda i: fragments[i].importance, reverse=True)
            comp = comp[:max_cluster_size]
        cluster_frags = [fragments[i] for i in comp]
        results.append(Cluster(
            fragments=cluster_frags,
            total_importance=sum(f.importance for f in cluster_frags),
        ))
    return results


# ---------------------------------------------------------------------------
# LLM rerank (optional — works without it, just noisier)
# ---------------------------------------------------------------------------

RERANK_PROMPT = """你是记忆聚类质量判断器。以下碎片由embedding相似度聚在一起，请判断它们是否真的在说同一件事或同一条叙事线。

碎片列表：
{candidates}

打分规则：
2 = 确实是同一件事/同一条叙事线的不同时间点
1 = 有关联但不是同一件事（比如都是关于工作但不是同一个项目）
0 = 不相关，embedding误聚

严格判断。仅仅关键词相似不算2，必须是同一件事的不同时间切片。

返回JSON数组，只返回JSON：
[{{"id":"fragment_id","score":2,"reason":"..."}}]"""


def rerank_cluster(
    cluster: Cluster,
    *,
    api_url: str,
    api_key: str,
    model: str = "deepseek-chat",
    min_score: int = 1,
) -> Cluster:
    """
    Use an LLM to verify cluster coherence. Removes fragments scoring below
    min_score and records scores on the Cluster object.

    Args:
        cluster: a Cluster from discover_clusters()
        api_url: OpenAI-compatible chat completions endpoint
        api_key: API key
        model: model name
        min_score: fragments below this are dropped (0=noise, 1=related, 2=same thing)

    Returns:
        filtered Cluster with rerank_scores populated
    """
    lines = []
    for i, f in enumerate(cluster.fragments):
        lines.append(f'[{i+1}] (id={f.id}, {f.created_at[:10]}, tags={f.tags}) {f.text[:200]}')
    prompt = RERANK_PROMPT.format(candidates="\n".join(lines))

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 800,
    }).encode()
    req = urllib.request.Request(api_url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    text = data["choices"][0]["message"]["content"]
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    scores = json.loads(text)

    score_map = {str(s["id"]): int(s.get("score", 0)) for s in scores}
    good_ids = {sid for sid, sc in score_map.items() if sc >= min_score}

    cluster.fragments = [f for f in cluster.fragments if f.id in good_ids]
    cluster.rerank_scores = score_map
    cluster.total_importance = sum(f.importance for f in cluster.fragments)
    return cluster


# ---------------------------------------------------------------------------
# Centroid matching (find related existing events)
# ---------------------------------------------------------------------------

def find_related(
    cluster: Cluster,
    candidates: list[tuple[str, np.ndarray]],
    *,
    min_cosine: float = 0.50,
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """
    Find existing events/documents most similar to a cluster's centroid.

    Args:
        cluster: a Cluster object
        candidates: list of (id, embedding) tuples to match against
        min_cosine: minimum cosine similarity to include
        top_n: max results

    Returns:
        list of (id, cosine_similarity), sorted descending
    """
    if not cluster.fragments or not candidates:
        return []

    centroid = np.mean([f.embedding for f in cluster.fragments], axis=0)
    c_norm = np.linalg.norm(centroid)
    if c_norm == 0:
        return []
    centroid = centroid / c_norm

    results = []
    for cid, emb in candidates:
        en = np.linalg.norm(emb)
        if en > 0:
            cos = float(np.dot(centroid, emb / en))
            if cos >= min_cosine:
                results.append((cid, round(cos, 3)))

    results.sort(key=lambda x: -x[1])
    return results[:top_n]


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal example with random embeddings (replace with real ones)
    rng = np.random.default_rng(42)

    # Simulate two clusters: fragments 0-4 about "topic A", 5-9 about "topic B"
    base_a = rng.normal(size=384)
    base_b = rng.normal(size=384)

    frags = []
    for i in range(10):
        base = base_a if i < 5 else base_b
        emb = base + rng.normal(size=384) * 0.3  # add noise
        frags.append(Fragment(
            id=f"frag_{i}",
            text=f"Fragment {i} about {'topic A' if i < 5 else 'topic B'}",
            embedding=emb.astype(np.float32),
            importance=float(rng.integers(4, 10)),
            created_at=f"2026-07-{10+i}",
            tags="example",
        ))

    clusters = discover_clusters(frags, top_n=2)
    for i, c in enumerate(clusters):
        print(f"\nCluster {i+1} (importance={c.total_importance:.0f}):")
        for f in c.fragments:
            print(f"  - {f.id}: {f.text}")
