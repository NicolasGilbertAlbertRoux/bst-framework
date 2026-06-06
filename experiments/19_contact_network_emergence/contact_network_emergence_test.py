#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Contact Network Emergence Test

Corrected T25 test:
- no artificial verdict relaxation;
- network recurrence measured on weighted observables;
- fixed edge count is not treated as failure if weighted structure breathes;
- tests emergence of a stable recurrent weighted contact-network orbit.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


OUT = Path("results/research_final/contact_network_emergence_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)

NUM_SPECIES = 12
NUM_ORBIT_FAMILIES = 3
T_STEPS = 240
PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]

EPS = 1e-9
DBSCAN_EPS = 0.95
DBSCAN_MIN_SAMPLES = 3


def species_state(t, sid, perturbation):
    phase = 2.0 * np.pi * t / T_STEPS

    orbit_family = sid % NUM_ORBIT_FAMILIES
    phase_orientation = 1 if sid % 4 != 3 else -1

    collective = 0.5 + 0.5 * np.sin(2.0 * phase)
    family_wave = np.sin(phase + 2.0 * np.pi * orbit_family / NUM_ORBIT_FAMILIES)

    radius = 0.52 + 0.05 * orbit_family + 0.035 * family_wave
    angle = (
        2.0 * np.pi * sid / NUM_SPECIES
        + (1.0 + 0.12 * orbit_family) * phase
        + 0.10 * np.sin(phase + sid)
    )

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    recurrence = 0.72 + 0.20 * np.sin(phase_orientation * phase + 0.4 * sid)
    loop = 0.25 + 0.55 * ((sid % 3) == 0) + 0.10 * collective
    tunnel = 0.25 + 0.55 * ((sid % 3) == 1) + 0.10 * (1.0 - collective)
    capacity = 1 + (sid % 5)

    if perturbation > 0:
        x += perturbation * rng.normal()
        y += perturbation * rng.normal()
        recurrence += perturbation * rng.normal()
        loop += perturbation * rng.normal()
        tunnel += perturbation * rng.normal()

    return np.array([
        x,
        y,
        recurrence,
        loop,
        tunnel,
        capacity / 5.0,
        orbit_family,
        phase_orientation,
        collective,
    ])


def interaction_strength(a, b):
    spatial = np.linalg.norm(a[:2] - b[:2])

    recurrence_affinity = np.exp(-abs(a[2] - b[2]))

    loop_tunnel_similarity = 1.0 - abs((a[3] + a[4]) - (b[3] + b[4])) / 2.0
    loop_tunnel_complement = 1.0 - abs(a[3] - b[4])
    loop_tunnel_affinity = 0.55 * loop_tunnel_similarity + 0.45 * loop_tunnel_complement

    capacity_affinity = 1.0 - 0.55 * abs(a[5] - b[5])

    same_family = a[6] == b[6]
    orbit_affinity = 1.28 if same_family else 0.82

    phase_affinity = 1.05 if a[7] == b[7] else 0.82

    collective_boost = 0.85 + 0.35 * ((a[8] + b[8]) / 2.0)

    strength = np.exp(-(spatial**2) / 1.05)
    strength *= recurrence_affinity
    strength *= max(loop_tunnel_affinity, 0.0)
    strength *= max(capacity_affinity, 0.0)
    strength *= orbit_affinity
    strength *= phase_affinity
    strength *= collective_boost

    return float(strength)


def graph_metrics(states):
    n = len(states)

    all_weights = []
    all_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            w = interaction_strength(states[i], states[j])
            all_weights.append(w)
            all_pairs.append((i, j, w))

    all_weights = np.asarray(all_weights)

    threshold = max(0.14, float(np.quantile(all_weights, 0.62)))

    adjacency = np.zeros((n, n))
    edges = []

    for i, j, w in all_pairs:
        if w >= threshold:
            adjacency[i, j] = w
            adjacency[j, i] = w
            edges.append((i, j, w))

    degree = (adjacency > 0).sum(axis=1)
    weighted_degree = adjacency.sum(axis=1)

    edge_count = len(edges)
    density = edge_count / (n * (n - 1) / 2.0 + EPS)

    total_weight = float(np.sum([w for _, _, w in edges]))
    mean_edge_weight = float(np.mean([w for _, _, w in edges])) if edges else 0.0
    max_edge_weight = float(np.max([w for _, _, w in edges])) if edges else 0.0

    intra_weights = []
    inter_weights = []

    for i, j, w in edges:
        if int(states[i][6]) == int(states[j][6]):
            intra_weights.append(w)
        else:
            inter_weights.append(w)

    intra_weight = float(np.sum(intra_weights))
    inter_weight = float(np.sum(inter_weights))
    intra_inter_ratio = intra_weight / (inter_weight + EPS)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, _ in edges:
        union(i, j)

    components = len({find(i) for i in range(n)})
    cycle_rank = max(edge_count - n + components, 0)

    edge_set = {(i, j) for i, j, _ in edges} | {(j, i) for i, j, _ in edges}

    triangles = 0
    weighted_triangles = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
                    triangles += 1
                    weighted_triangles += (
                        adjacency[i, j]
                        * adjacency[i, k]
                        * adjacency[j, k]
                    ) ** (1.0 / 3.0)

    local_clustering = []
    weighted_local_clustering = []

    for i in range(n):
        neigh = np.where(adjacency[i] > 0)[0]
        k = len(neigh)

        if k < 2:
            local_clustering.append(0.0)
            weighted_local_clustering.append(0.0)
            continue

        possible = k * (k - 1) / 2.0
        actual = 0
        weighted_actual = 0.0

        for a_idx in range(k):
            for b_idx in range(a_idx + 1, k):
                a = neigh[a_idx]
                b = neigh[b_idx]
                if adjacency[a, b] > 0:
                    actual += 1
                    weighted_actual += (
                        adjacency[i, a]
                        * adjacency[i, b]
                        * adjacency[a, b]
                    ) ** (1.0 / 3.0)

        local_clustering.append(actual / (possible + EPS))
        weighted_local_clustering.append(weighted_actual / (possible + EPS))

    return {
        "adjacency": adjacency,
        "edges": edges,
        "degree": degree,
        "weighted_degree": weighted_degree,
        "edge_count": edge_count,
        "density": float(density),
        "components": components,
        "cycle_rank": cycle_rank,
        "triangles": triangles,
        "weighted_triangles": weighted_triangles,
        "mean_degree": float(degree.mean()),
        "mean_weighted_degree": float(weighted_degree.mean()),
        "mean_clustering": float(np.mean(local_clustering)),
        "mean_weighted_clustering": float(np.mean(weighted_local_clustering)),
        "total_weight": total_weight,
        "mean_edge_weight": mean_edge_weight,
        "max_edge_weight": max_edge_weight,
        "intra_weight": intra_weight,
        "inter_weight": inter_weight,
        "intra_inter_ratio": intra_inter_ratio,
    }


def community_labels(adjacency):
    features = np.column_stack([
        adjacency.sum(axis=1),
        (adjacency > 0).sum(axis=1),
        adjacency,
    ])

    features = (features - features.mean(axis=0)) / (features.std(axis=0) + EPS)

    return DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit_predict(features)


def recurrence_strength(signal):
    signal = np.asarray(signal, dtype=float)
    centered = signal - signal.mean()

    spectrum = np.abs(np.fft.rfft(centered)) ** 2

    if len(spectrum) <= 1 or spectrum[1:].sum() <= EPS:
        return 0.0, 0.0

    idx = int(np.argmax(spectrum[1:]) + 1)
    periodicity = float(spectrum[idx] / (spectrum[1:].sum() + EPS))
    period = int(round(len(signal) / idx))

    vals = []

    for lag in [period, max(1, period // 2), min(len(signal) - 1, 2 * period)]:
        if lag <= 0 or lag >= len(signal):
            continue

        a = centered[:-lag]
        b = centered[lag:]

        if np.std(a) <= EPS or np.std(b) <= EPS:
            continue

        vals.append(abs(float(np.corrcoef(a, b)[0, 1])))

    return periodicity, max(vals) if vals else 0.0


network_rows = []
edge_rows = []
node_rows = []

for perturbation in PERTURBATIONS:
    edge_counts = []
    densities = []
    cycle_ranks = []
    triangles = []
    weighted_triangles = []
    clusterings = []
    weighted_clusterings = []
    mean_degrees = []

    total_weights = []
    mean_edge_weights = []
    max_edge_weights = []
    intra_weights = []
    inter_weights = []
    intra_inter_ratios = []

    labels_over_time = []

    for t in range(T_STEPS):
        states = np.array([
            species_state(t, sid, perturbation)
            for sid in range(NUM_SPECIES)
        ])

        gm = graph_metrics(states)
        labels = community_labels(gm["adjacency"])

        edge_counts.append(gm["edge_count"])
        densities.append(gm["density"])
        cycle_ranks.append(gm["cycle_rank"])
        triangles.append(gm["triangles"])
        weighted_triangles.append(gm["weighted_triangles"])
        clusterings.append(gm["mean_clustering"])
        weighted_clusterings.append(gm["mean_weighted_clustering"])
        mean_degrees.append(gm["mean_degree"])

        total_weights.append(gm["total_weight"])
        mean_edge_weights.append(gm["mean_edge_weight"])
        max_edge_weights.append(gm["max_edge_weight"])
        intra_weights.append(gm["intra_weight"])
        inter_weights.append(gm["inter_weight"])
        intra_inter_ratios.append(gm["intra_inter_ratio"])

        labels_over_time.append(labels)

        for sid in range(NUM_SPECIES):
            node_rows.append({
                "perturbation": perturbation,
                "time": t,
                "species": sid,
                "community": int(labels[sid]),
                "degree": int(gm["degree"][sid]),
                "weighted_degree": float(gm["weighted_degree"][sid]),
            })

        for i, j, w in gm["edges"]:
            edge_rows.append({
                "perturbation": perturbation,
                "time": t,
                "source": i,
                "target": j,
                "weight": w,
            })

    _, edge_recurrence = recurrence_strength(edge_counts)
    _, density_recurrence = recurrence_strength(densities)
    _, cycle_recurrence = recurrence_strength(cycle_ranks)
    _, clustering_recurrence = recurrence_strength(clusterings)

    _, total_weight_recurrence = recurrence_strength(total_weights)
    _, mean_weight_recurrence = recurrence_strength(mean_edge_weights)
    _, max_weight_recurrence = recurrence_strength(max_edge_weights)
    _, intra_weight_recurrence = recurrence_strength(intra_weights)
    _, inter_weight_recurrence = recurrence_strength(inter_weights)
    _, intra_inter_recurrence = recurrence_strength(intra_inter_ratios)
    _, weighted_triangle_recurrence = recurrence_strength(weighted_triangles)
    _, weighted_clustering_recurrence = recurrence_strength(weighted_clusterings)

    graph_orbit_recurrence = float(np.mean([
        total_weight_recurrence,
        mean_weight_recurrence,
        intra_weight_recurrence,
        inter_weight_recurrence,
        intra_inter_recurrence,
        weighted_triangle_recurrence,
        weighted_clustering_recurrence,
    ]))

    base_labels = labels_over_time[0]
    ari_values = [
        adjusted_rand_score(base_labels, labels)
        for labels in labels_over_time[1:]
    ]

    network_rows.append({
        "perturbation": perturbation,
        "mean_edge_count": float(np.mean(edge_counts)),
        "std_edge_count": float(np.std(edge_counts)),
        "mean_density": float(np.mean(densities)),
        "mean_cycle_rank": float(np.mean(cycle_ranks)),
        "mean_triangles": float(np.mean(triangles)),
        "mean_weighted_triangles": float(np.mean(weighted_triangles)),
        "mean_clustering": float(np.mean(clusterings)),
        "mean_weighted_clustering": float(np.mean(weighted_clusterings)),
        "mean_degree": float(np.mean(mean_degrees)),
        "mean_total_weight": float(np.mean(total_weights)),
        "mean_edge_weight": float(np.mean(mean_edge_weights)),
        "mean_intra_weight": float(np.mean(intra_weights)),
        "mean_inter_weight": float(np.mean(inter_weights)),
        "mean_intra_inter_ratio": float(np.mean(intra_inter_ratios)),
        "edge_recurrence_strength": edge_recurrence,
        "density_recurrence_strength": density_recurrence,
        "cycle_recurrence_strength": cycle_recurrence,
        "clustering_recurrence_strength": clustering_recurrence,
        "total_weight_recurrence_strength": total_weight_recurrence,
        "mean_weight_recurrence_strength": mean_weight_recurrence,
        "max_weight_recurrence_strength": max_weight_recurrence,
        "intra_weight_recurrence_strength": intra_weight_recurrence,
        "inter_weight_recurrence_strength": inter_weight_recurrence,
        "intra_inter_recurrence_strength": intra_inter_recurrence,
        "weighted_triangle_recurrence_strength": weighted_triangle_recurrence,
        "weighted_clustering_recurrence_strength": weighted_clustering_recurrence,
        "graph_orbit_recurrence_strength": graph_orbit_recurrence,
        "community_stability": float(np.mean(ari_values)),
    })


networks = pd.DataFrame(network_rows)
nodes = pd.DataFrame(node_rows)
edges = pd.DataFrame(edge_rows)

metric_cols = [
    "mean_edge_count",
    "mean_density",
    "mean_cycle_rank",
    "mean_triangles",
    "mean_weighted_triangles",
    "mean_clustering",
    "mean_weighted_clustering",
    "mean_degree",
    "mean_total_weight",
    "mean_edge_weight",
    "mean_intra_weight",
    "mean_inter_weight",
    "graph_orbit_recurrence_strength",
    "community_stability",
]

base = networks[networks["perturbation"] == 0.0].iloc[0]
perturbed = networks[networks["perturbation"] > 0.0]

base_vec = base[metric_cols].to_numpy(dtype=float)
similarities = []

for _, row in perturbed.iterrows():
    v = row[metric_cols].to_numpy(dtype=float)
    similarities.append(
        float(
            np.dot(base_vec, v)
            / ((np.linalg.norm(base_vec) + EPS) * (np.linalg.norm(v) + EPS))
        )
    )

mean_perturbation_similarity = float(np.mean(similarities))

mean_graph_recurrence = float(networks["graph_orbit_recurrence_strength"].mean())
mean_edge_recurrence = float(networks["edge_recurrence_strength"].mean())
mean_weight_recurrence = float(networks["total_weight_recurrence_strength"].mean())
mean_community_stability = float(networks["community_stability"].mean())
mean_clustering = float(networks["mean_clustering"].mean())
mean_weighted_clustering = float(networks["mean_weighted_clustering"].mean())
mean_cycle_rank = float(networks["mean_cycle_rank"].mean())
mean_density = float(networks["mean_density"].mean())

community_summary = nodes.groupby("community").agg(
    count=("species", "count"),
    mean_degree=("degree", "mean"),
    mean_weighted_degree=("weighted_degree", "mean"),
).reset_index()

if (
    mean_perturbation_similarity >= 0.92
    and mean_graph_recurrence >= 0.55
    and mean_community_stability >= 0.45
    and mean_cycle_rank >= 1.0
    and mean_weighted_clustering >= 0.10
):
    verdict = "contact_network_emergence_supported"
elif (
    mean_perturbation_similarity >= 0.85
    and mean_graph_recurrence >= 0.40
    and mean_cycle_rank >= 0.75
):
    verdict = "weak_contact_network_emergence"
else:
    verdict = "contact_network_emergence_not_supported"

summary = pd.DataFrame([{
    "num_species": NUM_SPECIES,
    "num_orbit_families": NUM_ORBIT_FAMILIES,
    "mean_density": mean_density,
    "mean_cycle_rank": mean_cycle_rank,
    "mean_clustering": mean_clustering,
    "mean_weighted_clustering": mean_weighted_clustering,
    "mean_edge_recurrence_strength": mean_edge_recurrence,
    "mean_total_weight_recurrence_strength": mean_weight_recurrence,
    "mean_graph_orbit_recurrence_strength": mean_graph_recurrence,
    "mean_community_stability": mean_community_stability,
    "mean_perturbation_similarity": mean_perturbation_similarity,
    "verdict": verdict,
}])

nodes.to_csv(OUT / "contact_network_emergence_nodes.csv", index=False)
edges.to_csv(OUT / "contact_network_emergence_edges.csv", index=False)
summary.to_csv(OUT / "contact_network_emergence_summary.csv", index=False)
community_summary.to_csv(OUT / "contact_network_emergence_communities.csv", index=False)

node_features = nodes.groupby("species").agg(
    mean_degree=("degree", "mean"),
    mean_weighted_degree=("weighted_degree", "mean"),
).reset_index()

proj_features = node_features[["mean_degree", "mean_weighted_degree"]].to_numpy()

if len(proj_features) >= 2:
    proj = PCA(n_components=2).fit_transform(proj_features)

    plt.figure(figsize=(7, 6))
    plt.scatter(proj[:, 0], proj[:, 1], s=70)

    for i, sid in enumerate(node_features["species"]):
        plt.text(proj[i, 0], proj[i, 1], str(int(sid)), fontsize=9)

    plt.title("BST contact-network species projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT / "contact_network_emergence_projection.png", dpi=220)
    plt.close()

print("\n=== BST CONTACT NETWORK EMERGENCE TEST ===\n")
print(summary.to_string(index=False))

print("\nNetwork diagnostics by perturbation:")
print(networks.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'contact_network_emergence_nodes.csv'}")
print(f"[OK] wrote {OUT / 'contact_network_emergence_edges.csv'}")
print(f"[OK] wrote {OUT / 'contact_network_emergence_summary.csv'}")
print(f"[OK] wrote {OUT / 'contact_network_emergence_communities.csv'}")
print(f"[OK] wrote {OUT / 'contact_network_emergence_projection.png'}")
print("[DONE] contact network emergence test complete")