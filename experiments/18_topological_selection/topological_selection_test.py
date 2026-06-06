#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Stable Topological Orbit Selection Test

Purpose
-------
Test whether BST contact-matter dynamics selects stable loop/tunnel
topological orbits rather than merely low-entropy static topologies.

Corrected hypothesis for T24
----------------------------
Stable matter-like species may visit several graph/topological states
during a breathing cycle. High topological entropy is therefore not
necessarily instability. The relevant criterion is recurrence and
robustness of the orbit under perturbation.

Outputs
-------
results/research_final/topological_selection_test/
    topological_selection_candidates.csv
    topological_selection_families.csv
    topological_selection_summary.csv
    topological_selection_projection.png
"""

from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


OUT = Path("results/research_final/topological_selection_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)

NUM_CONFIGS = 160
T_STEPS = 240
PERTURBATIONS = [0.0, 0.015, 0.035]

EPS = 1e-9

DBSCAN_EPS = 0.92
DBSCAN_MIN_SAMPLES = 4


def centers_4d(t, cfg, perturbation):
    latent_type = cfg % 6
    n = 3 + (cfg % 6)

    phase = 2.0 * np.pi * t / T_STEPS

    base_radius = 0.38 + 0.045 * latent_type
    breath = base_radius + 0.08 * np.sin((1 + latent_type % 3) * phase)

    twist = 0.10 * latent_type * np.sin(phase)
    z_amp = 0.06 + 0.015 * latent_type
    phase_amp = 0.16 + 0.02 * latent_type

    pts = []

    for k in range(n):
        a = 2.0 * np.pi * k / n
        a += twist
        a += 0.05 * np.sin(phase + 0.7 * k)

        anisotropy = 1.0 + 0.10 * np.sin(cfg + k)

        x = breath * anisotropy * np.cos(a)
        y = breath / anisotropy * np.sin(a)
        z = z_amp * np.sin(phase + 2.0 * np.pi * k / n)
        q = phase_amp * np.cos(phase + 0.5 * k + 0.3 * latent_type)

        if perturbation > 0:
            x += perturbation * rng.normal()
            y += perturbation * rng.normal()
            z += perturbation * rng.normal()
            q += perturbation * rng.normal()

        pts.append([x, y, z, q])

    return np.asarray(pts)


def contact_strengths(points):
    n = len(points)
    strengths = []

    sigma = 0.42

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            s = np.exp(-(d**2) / (2.0 * sigma**2))
            strengths.append((i, j, float(s)))

    return strengths


def graph_signature(strengths, n):
    vals = np.asarray([s for _, _, s in strengths])

    if len(vals) == 0:
        return {
            "edges": 0,
            "components": n,
            "cycle_rank": 0,
            "triangles": 0,
            "density": 0.0,
            "mean_strength": 0.0,
            "max_strength": 0.0,
        }

    threshold = max(0.42, float(np.quantile(vals, 0.62)))
    edges = [(i, j) for i, j, s in strengths if s >= threshold]

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

    for i, j in edges:
        union(i, j)

    components = len({find(i) for i in range(n)})

    edge_set = {tuple(sorted(e)) for e in edges}

    triangles = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (
                    (i, j) in edge_set
                    and (i, k) in edge_set
                    and (j, k) in edge_set
                ):
                    triangles += 1

    edge_count = len(edges)
    cycle_rank = max(edge_count - n + components, 0)
    density = edge_count / (n * (n - 1) / 2.0 + EPS)

    return {
        "edges": edge_count,
        "components": components,
        "cycle_rank": cycle_rank,
        "triangles": triangles,
        "density": float(density),
        "mean_strength": float(vals.mean()),
        "max_strength": float(vals.max()),
    }


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


def entropy_of_signatures(signatures):
    counts = Counter(signatures)
    probs = np.asarray(list(counts.values()), dtype=float)
    probs /= probs.sum() + EPS

    entropy = -np.sum(probs * np.log(probs + EPS))
    max_entropy = np.log(len(counts) + EPS)

    if max_entropy <= EPS:
        return 0.0, len(counts)

    return float(entropy / max_entropy), len(counts)


def run_config(cfg, perturbation):
    rows = []
    signatures = []

    for t in range(T_STEPS):
        pts = centers_4d(t, cfg, perturbation)
        strengths = contact_strengths(pts)
        sig = graph_signature(strengths, len(pts))

        signatures.append(
            (
                sig["edges"],
                sig["components"],
                sig["cycle_rank"],
                sig["triangles"],
            )
        )

        rows.append(sig)

    df = pd.DataFrame(rows)

    topo_entropy, unique_signatures = entropy_of_signatures(signatures)
    periodicity, recurrence = recurrence_strength(df["edges"].to_numpy())

    return {
        "cfg": cfg,
        "perturbation": perturbation,
        "num_clusters": 3 + (cfg % 6),
        "latent_type": cfg % 6,
        "mean_edges": float(df["edges"].mean()),
        "mean_components": float(df["components"].mean()),
        "mean_cycle_rank": float(df["cycle_rank"].mean()),
        "mean_triangles": float(df["triangles"].mean()),
        "mean_density": float(df["density"].mean()),
        "mean_contact_strength": float(df["mean_strength"].mean()),
        "max_contact_strength": float(df["max_strength"].max()),
        "topological_entropy": topo_entropy,
        "unique_signatures": unique_signatures,
        "periodicity_score": periodicity,
        "recurrence_strength": recurrence,
    }


all_rows = []

for cfg in range(NUM_CONFIGS):
    for p in PERTURBATIONS:
        all_rows.append(run_config(cfg, p))

raw = pd.DataFrame(all_rows)

feature_cols = [
    "mean_edges",
    "mean_components",
    "mean_cycle_rank",
    "mean_triangles",
    "mean_density",
    "mean_contact_strength",
    "max_contact_strength",
    "topological_entropy",
    "unique_signatures",
    "periodicity_score",
    "recurrence_strength",
]


candidate_rows = []

for cfg in range(NUM_CONFIGS):
    sub = raw[raw["cfg"] == cfg].sort_values("perturbation")

    base = sub[sub["perturbation"] == 0.0].iloc[0]
    perturbed = sub[sub["perturbation"] > 0.0]

    base_vec = base[feature_cols].to_numpy(dtype=float)

    sims = []

    for _, row in perturbed.iterrows():
        v = row[feature_cols].to_numpy(dtype=float)

        sim = float(
            np.dot(base_vec, v)
            / ((np.linalg.norm(base_vec) + EPS) * (np.linalg.norm(v) + EPS))
        )

        sims.append(sim)

    mean_similarity = float(np.mean(sims)) if sims else 0.0

    # Corrected selection:
    # high entropy is no longer rejected. It can represent a rich but stable
    # topological orbit. Selection depends on perturbative robustness,
    # recurrence, and non-trivial loop/tunnel cyclicity.
    selected = (
        mean_similarity >= 0.88
        and base["recurrence_strength"] >= 0.60
        and base["mean_cycle_rank"] >= 0.05
    )

    record = base.to_dict()
    record["mean_perturbation_similarity"] = mean_similarity
    record["selected_topology"] = bool(selected)

    candidate_rows.append(record)

candidates = pd.DataFrame(candidate_rows)
stable = candidates[candidates["selected_topology"]].copy()

if len(stable) >= DBSCAN_MIN_SAMPLES:
    Xf = stable[feature_cols].to_numpy(dtype=float)
    Xf = (Xf - Xf.mean(axis=0)) / (Xf.std(axis=0) + EPS)

    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit_predict(Xf)

    stable["topological_family"] = labels
else:
    labels = np.array([])
    stable["topological_family"] = []

num_families = (
    len(set(labels)) - (1 if -1 in labels else 0)
    if len(labels)
    else 0
)

noise_fraction = (
    float(np.mean(labels == -1))
    if len(labels)
    else np.nan
)

if num_families >= 2 and np.any(labels >= 0):
    sil = float(silhouette_score(Xf[labels >= 0], labels[labels >= 0]))
else:
    sil = np.nan

family_rows = []

if len(stable):
    for label in sorted(set(stable["topological_family"])):
        if label < 0:
            continue

        sub = stable[stable["topological_family"] == label]

        family_rows.append(
            {
                "topological_family": int(label),
                "count": int(len(sub)),
                "mean_cycle_rank": float(sub["mean_cycle_rank"].mean()),
                "mean_triangles": float(sub["mean_triangles"].mean()),
                "mean_topological_entropy": float(sub["topological_entropy"].mean()),
                "mean_recurrence_strength": float(sub["recurrence_strength"].mean()),
                "mean_perturbation_similarity": float(
                    sub["mean_perturbation_similarity"].mean()
                ),
                "mean_contact_strength": float(sub["mean_contact_strength"].mean()),
            }
        )

families = pd.DataFrame(family_rows)

selected_fraction = float(candidates["selected_topology"].mean())
selection_pressure = 1.0 - selected_fraction

mean_similarity = float(candidates["mean_perturbation_similarity"].mean())
mean_recurrence = float(candidates["recurrence_strength"].mean())
mean_entropy = float(candidates["topological_entropy"].mean())

if (
    selected_fraction < 0.75
    and len(stable) >= 20
    and num_families >= 3
    and mean_recurrence >= 0.60
):
    verdict = "topological_orbit_selection_supported"
elif (
    selected_fraction < 0.90
    and len(stable) >= 10
    and num_families >= 2
):
    verdict = "weak_topological_orbit_selection"
else:
    verdict = "topological_orbit_selection_not_supported"

summary = pd.DataFrame(
    [
        {
            "num_candidates": int(NUM_CONFIGS),
            "num_selected_topologies": int(len(stable)),
            "selected_fraction": selected_fraction,
            "selection_pressure": selection_pressure,
            "num_topological_families": int(num_families),
            "noise_fraction": noise_fraction,
            "silhouette_score": sil,
            "mean_perturbation_similarity": mean_similarity,
            "mean_recurrence_strength": mean_recurrence,
            "mean_topological_entropy": mean_entropy,
            "verdict": verdict,
        }
    ]
)

candidates.to_csv(OUT / "topological_selection_candidates.csv", index=False)
families.to_csv(OUT / "topological_selection_families.csv", index=False)
summary.to_csv(OUT / "topological_selection_summary.csv", index=False)

if len(stable) >= 2:
    Xplot = stable[feature_cols].to_numpy(dtype=float)
    Xplot = (Xplot - Xplot.mean(axis=0)) / (Xplot.std(axis=0) + EPS)
    proj = PCA(n_components=2).fit_transform(Xplot)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        proj[:, 0],
        proj[:, 1],
        c=stable["topological_family"],
        s=28,
    )
    plt.title("BST selected loop/tunnel topological orbit families")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT / "topological_selection_projection.png", dpi=220)
    plt.close()

print("\n=== BST TOPOLOGICAL ORBIT SELECTION TEST ===\n")
print(summary.to_string(index=False))

print("\nTopological orbit families:")
if len(families):
    print(families.to_string(index=False))
else:
    print("(no stable topological orbit families detected)")

print(f"\n[OK] wrote {OUT / 'topological_selection_candidates.csv'}")
print(f"[OK] wrote {OUT / 'topological_selection_families.csv'}")
print(f"[OK] wrote {OUT / 'topological_selection_summary.csv'}")
print(f"[OK] wrote {OUT / 'topological_selection_projection.png'}")
print("[DONE] topological orbit selection test complete")