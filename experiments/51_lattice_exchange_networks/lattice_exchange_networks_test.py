#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 51 — Lattice Exchange Networks

Place this file at:
    experiments/51_lattice_exchange_networks/lattice_exchange_networks_test.py

Run:
    python experiments/51_lattice_exchange_networks/lattice_exchange_networks_test.py

Outputs:
    results/research_final/lattice_exchange_networks_test/
        lattice_exchange_samples.csv
        lattice_exchange_edges.csv
        lattice_exchange_summary.csv
        lattice_exchange_by_family.csv
        lattice_exchange_clusters.csv
        lattice_exchange_matrix.png
        lattice_exchange_projection.png
        lattice_exchange_network.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier

OUT = Path("results/research_final/lattice_exchange_networks_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9
PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 64

BLOCK_ID = {"s": 0, "p": 1, "d": 2, "f": 3}
FAMILY_ID = {
    "alkali": 0,
    "alkaline_earth": 1,
    "p_network": 2,
    "halogen": 3,
    "noble_gas": 4,
    "transition_metal": 5,
    "lanthanide": 6,
    "actinide": 7,
}

# Z, symbol, period, group, block, valence, d_count, f_count, lattice_family
ELEMENTS = [
    (3,"Li",2,1,"s",1,0,0,"alkali"),(11,"Na",3,1,"s",1,0,0,"alkali"),(19,"K",4,1,"s",1,0,0,"alkali"),(37,"Rb",5,1,"s",1,0,0,"alkali"),(55,"Cs",6,1,"s",1,0,0,"alkali"),(87,"Fr",7,1,"s",1,0,0,"alkali"),
    (4,"Be",2,2,"s",2,0,0,"alkaline_earth"),(12,"Mg",3,2,"s",2,0,0,"alkaline_earth"),(20,"Ca",4,2,"s",2,0,0,"alkaline_earth"),(38,"Sr",5,2,"s",2,0,0,"alkaline_earth"),(56,"Ba",6,2,"s",2,0,0,"alkaline_earth"),(88,"Ra",7,2,"s",2,0,0,"alkaline_earth"),
    (6,"C",2,14,"p",4,0,0,"p_network"),(14,"Si",3,14,"p",4,0,0,"p_network"),(32,"Ge",4,14,"p",4,0,0,"p_network"),(50,"Sn",5,14,"p",4,0,0,"p_network"),(82,"Pb",6,14,"p",4,0,0,"p_network"),(7,"N",2,15,"p",5,0,0,"p_network"),(15,"P",3,15,"p",5,0,0,"p_network"),(33,"As",4,15,"p",5,0,0,"p_network"),(51,"Sb",5,15,"p",5,0,0,"p_network"),(83,"Bi",6,15,"p",5,0,0,"p_network"),
    (9,"F",2,17,"p",7,0,0,"halogen"),(17,"Cl",3,17,"p",7,0,0,"halogen"),(35,"Br",4,17,"p",7,0,0,"halogen"),(53,"I",5,17,"p",7,0,0,"halogen"),(85,"At",6,17,"p",7,0,0,"halogen"),
    (2,"He",1,18,"s",8,0,0,"noble_gas"),(10,"Ne",2,18,"p",8,0,0,"noble_gas"),(18,"Ar",3,18,"p",8,0,0,"noble_gas"),(36,"Kr",4,18,"p",8,0,0,"noble_gas"),(54,"Xe",5,18,"p",8,0,0,"noble_gas"),(86,"Rn",6,18,"p",8,0,0,"noble_gas"),
    (21,"Sc",4,3,"d",1,1,0,"transition_metal"),(22,"Ti",4,4,"d",2,2,0,"transition_metal"),(23,"V",4,5,"d",3,3,0,"transition_metal"),(24,"Cr",4,6,"d",4,5,0,"transition_metal"),(25,"Mn",4,7,"d",5,5,0,"transition_metal"),(26,"Fe",4,8,"d",6,6,0,"transition_metal"),(27,"Co",4,9,"d",7,7,0,"transition_metal"),(28,"Ni",4,10,"d",8,8,0,"transition_metal"),(29,"Cu",4,11,"d",9,10,0,"transition_metal"),(30,"Zn",4,12,"d",10,10,0,"transition_metal"),
    (39,"Y",5,3,"d",1,1,0,"transition_metal"),(40,"Zr",5,4,"d",2,2,0,"transition_metal"),(41,"Nb",5,5,"d",3,4,0,"transition_metal"),(42,"Mo",5,6,"d",4,5,0,"transition_metal"),(43,"Tc",5,7,"d",5,5,0,"transition_metal"),(44,"Ru",5,8,"d",6,7,0,"transition_metal"),(45,"Rh",5,9,"d",7,8,0,"transition_metal"),(46,"Pd",5,10,"d",8,10,0,"transition_metal"),(47,"Ag",5,11,"d",9,10,0,"transition_metal"),(48,"Cd",5,12,"d",10,10,0,"transition_metal"),
    (72,"Hf",6,4,"d",2,2,0,"transition_metal"),(73,"Ta",6,5,"d",3,3,0,"transition_metal"),(74,"W",6,6,"d",4,4,0,"transition_metal"),(75,"Re",6,7,"d",5,5,0,"transition_metal"),(76,"Os",6,8,"d",6,6,0,"transition_metal"),(77,"Ir",6,9,"d",7,7,0,"transition_metal"),(78,"Pt",6,10,"d",8,9,0,"transition_metal"),(79,"Au",6,11,"d",9,10,0,"transition_metal"),(80,"Hg",6,12,"d",10,10,0,"transition_metal"),
    (57,"La",6,3,"f",3,1,0,"lanthanide"),(58,"Ce",6,3,"f",3,1,1,"lanthanide"),(59,"Pr",6,3,"f",3,0,3,"lanthanide"),(60,"Nd",6,3,"f",3,0,4,"lanthanide"),(61,"Pm",6,3,"f",3,0,5,"lanthanide"),(62,"Sm",6,3,"f",3,0,6,"lanthanide"),(63,"Eu",6,3,"f",3,0,7,"lanthanide"),(64,"Gd",6,3,"f",3,1,7,"lanthanide"),(65,"Tb",6,3,"f",3,0,9,"lanthanide"),(66,"Dy",6,3,"f",3,0,10,"lanthanide"),(67,"Ho",6,3,"f",3,0,11,"lanthanide"),(68,"Er",6,3,"f",3,0,12,"lanthanide"),(69,"Tm",6,3,"f",3,0,13,"lanthanide"),(70,"Yb",6,3,"f",3,0,14,"lanthanide"),(71,"Lu",6,3,"f",3,1,14,"lanthanide"),
    (89,"Ac",7,3,"f",3,1,0,"actinide"),(90,"Th",7,3,"f",4,2,0,"actinide"),(91,"Pa",7,3,"f",5,1,2,"actinide"),(92,"U",7,3,"f",6,1,3,"actinide"),(93,"Np",7,3,"f",6,1,4,"actinide"),(94,"Pu",7,3,"f",6,0,6,"actinide"),(95,"Am",7,3,"f",3,0,7,"actinide"),(96,"Cm",7,3,"f",3,1,7,"actinide"),(97,"Bk",7,3,"f",3,0,9,"actinide"),(98,"Cf",7,3,"f",3,0,10,"actinide"),(99,"Es",7,3,"f",3,0,11,"actinide"),(100,"Fm",7,3,"f",3,0,12,"actinide"),(101,"Md",7,3,"f",3,0,13,"actinide"),(102,"No",7,3,"f",3,0,14,"actinide"),(103,"Lr",7,3,"f",3,1,14,"actinide"),
]


def family_phase(family):
    return {
        "alkali": 0.10, "alkaline_earth": 0.22, "p_network": 1.05, "halogen": 1.34,
        "noble_gas": 2.85, "transition_metal": 0.62, "lanthanide": 1.95, "actinide": 2.18,
    }[family]


def closure_model(block, valence, f_count):
    if block == "s": return valence / 2.0, abs(2.0 - valence) / 2.0
    if block == "p": return valence / 8.0, abs(8.0 - valence) / 8.0
    if block == "d": return valence / 10.0, abs(10.0 - valence) / 10.0
    if block == "f": return f_count / 14.0, abs(14.0 - f_count) / 14.0
    raise ValueError(block)


def contact_capacity(block, valence, d_count, f_count, family):
    if family == "noble_gas": return 0.12
    if block == "s": return 0.25 + 0.12 * valence
    if block == "p": return 0.42 + 0.08 * abs(4.0 - valence)
    if block == "d": return 0.76 + 0.20 * np.exp(-abs(d_count - 5.0) / 3.5)
    if block == "f": return 0.68 + 0.18 * np.exp(-abs(f_count - 7.0) / 4.0)
    raise ValueError(block)


def make_element_sample(Z, symbol, period, group, block, valence, d_count, f_count, family, perturbation, replicate):
    d_fill = d_count / 10.0
    f_fill = f_count / 14.0
    valence_norm, closure_defect = closure_model(block, valence, f_count)
    closure_strength = np.exp(-closure_defect)
    noble_attractor = 1.0 if family == "noble_gas" else np.exp(-abs(group - 18.0) / 7.0)

    d_activation = 0.0
    if block == "d":
        d_activation = 0.62 + 0.28 * d_fill + 0.10 * np.exp(-abs(d_fill - 0.5))

    half_f = np.exp(-abs(f_count - 7.0) / 2.0) if block == "f" else 0.0
    filled_f = np.exp(-abs(f_count - 14.0) / 2.0) if block == "f" else 0.0
    f_symmetry = max(half_f, filled_f)

    f_activation = 0.0
    if block == "f":
        f_activation = 0.48 * (1.0 if f_count > 0 else 0.20) + 0.22 * f_fill + 0.20 * f_symmetry + 0.10 * np.exp(-abs(f_fill - 0.5))

    block_activation = {"s": 0.30 + 0.10 * valence_norm, "p": 0.44 + 0.22 * valence_norm, "d": d_activation, "f": f_activation}[block]
    phase = family_phase(family) + 0.05 * np.sin(2.0 * np.pi * valence_norm)
    capacity = contact_capacity(block, valence, d_count, f_count, family)

    recurrence_signature = (
        0.22 * closure_strength + 0.18 * block_activation + 0.16 * capacity + 0.13 * np.cos(phase)
        + 0.11 * np.sin(2.0 * np.pi * valence_norm) + 0.10 * (period / 7.0) + 0.10 * noble_attractor
    )
    shell_stability = (
        0.38 + 0.20 * closure_strength + 0.15 * block_activation + 0.12 * capacity + 0.10 * noble_attractor
        + 0.05 * (f_symmetry if block == "f" else np.exp(-abs(d_fill - 0.5)) if block == "d" else valence_norm)
    )
    exchange_availability = shell_stability * (0.22 if family == "noble_gas" else 1.0)
    noise = perturbation * rng.normal(0.0, 0.01)

    return {
        "Z": Z, "symbol": symbol, "period": period, "group": group, "block": block, "block_id": BLOCK_ID[block],
        "valence": valence, "d_count": d_count, "f_count": f_count, "d_fill": d_fill, "f_fill": f_fill,
        "valence_norm": valence_norm, "closure_defect": closure_defect, "closure_strength": closure_strength + noise,
        "noble_attractor": noble_attractor + noise, "d_activation": d_activation + noise, "f_activation": f_activation + noise,
        "f_symmetry": f_symmetry, "block_activation": block_activation + noise, "phase": phase + noise,
        "contact_capacity": capacity + noise, "recurrence_signature": recurrence_signature + noise,
        "shell_stability": shell_stability + noise, "exchange_availability": exchange_availability + noise,
        "lattice_family": family, "family_id": FAMILY_ID[family], "perturbation": perturbation, "replicate": replicate,
    }


def pair_exchange(a, b):
    recurrence_similarity = np.exp(-abs(a.recurrence_signature - b.recurrence_signature))
    phase_alignment = float(np.cos(a.phase - b.phase) ** 2)
    stability_coupling = np.sqrt(max(a.exchange_availability, 0.0) * max(b.exchange_availability, 0.0))
    capacity_match = 1.0 / (1.0 + abs(a.contact_capacity - b.contact_capacity))
    block_coherence = 1.0 if a.block == b.block else 0.65
    period_proximity = np.exp(-abs(a.period - b.period) / 4.5)
    internal_mode_match = 1.0
    if a.block == "d" and b.block == "d": internal_mode_match = np.exp(-abs(a.d_fill - b.d_fill) / 0.55)
    elif a.block == "f" and b.block == "f": internal_mode_match = np.exp(-abs(a.f_fill - b.f_fill) / 0.70)
    noble_suppression = 0.30 if (a.lattice_family == "noble_gas" or b.lattice_family == "noble_gas") else 1.0
    exchange_strength = recurrence_similarity * phase_alignment * stability_coupling * capacity_match * block_coherence * period_proximity * (0.75 + 0.25 * internal_mode_match) * noble_suppression
    return {
        "source": a.symbol, "target": b.symbol, "source_family": a.lattice_family, "target_family": b.lattice_family,
        "source_block": a.block, "target_block": b.block, "recurrence_similarity": recurrence_similarity,
        "phase_alignment": phase_alignment, "stability_coupling": stability_coupling, "capacity_match": capacity_match,
        "block_coherence": block_coherence, "period_proximity": period_proximity, "internal_mode_match": internal_mode_match,
        "exchange_strength": float(exchange_strength),
    }


rows = []
for perturbation in PERTURBATIONS:
    for e in ELEMENTS:
        for r in range(REPLICATES):
            rows.append(make_element_sample(*e, perturbation, r))

samples = pd.DataFrame(rows)

elements = samples.groupby(["Z", "symbol"]).agg(
    period=("period", "mean"), group=("group", "mean"), block=("block", "first"), block_id=("block_id", "mean"),
    valence=("valence", "mean"), d_count=("d_count", "mean"), f_count=("f_count", "mean"), d_fill=("d_fill", "mean"), f_fill=("f_fill", "mean"),
    valence_norm=("valence_norm", "mean"), closure_strength=("closure_strength", "mean"), noble_attractor=("noble_attractor", "mean"),
    d_activation=("d_activation", "mean"), f_activation=("f_activation", "mean"), f_symmetry=("f_symmetry", "mean"),
    block_activation=("block_activation", "mean"), phase=("phase", "mean"), contact_capacity=("contact_capacity", "mean"),
    recurrence_signature=("recurrence_signature", "mean"), shell_stability=("shell_stability", "mean"), exchange_availability=("exchange_availability", "mean"),
    lattice_family=("lattice_family", "first"), family_id=("family_id", "mean"),
).reset_index().sort_values("Z")

symbols = elements["symbol"].tolist()
n = len(elements)
edges = []
adj = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(i + 1, n):
        e = pair_exchange(elements.iloc[i], elements.iloc[j])
        edges.append(e)
        adj[i, j] = e["exchange_strength"]
        adj[j, i] = e["exchange_strength"]
np.fill_diagonal(adj, 1.0)
edges_df = pd.DataFrame(edges)

feature_cols = [
    "Z", "period", "group", "block_id", "valence", "d_count", "f_count", "d_fill", "f_fill", "valence_norm",
    "closure_strength", "noble_attractor", "d_activation", "f_activation", "f_symmetry", "block_activation", "phase",
    "contact_capacity", "recurrence_signature", "shell_stability", "exchange_availability",
]
X = elements[feature_cols].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)
Xs = samples[feature_cols].to_numpy(float)
Xsn = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + EPS)
train = samples["perturbation"].isin([0.0, 0.01]).to_numpy()
test = samples["perturbation"].isin([0.025, 0.05]).to_numpy()

clf_family = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf_family.fit(Xsn[train], samples.loc[train, "lattice_family"])
family_accuracy = float(accuracy_score(samples.loc[test, "lattice_family"], clf_family.predict(Xsn[test])))

clf_block = KNeighborsClassifier(n_neighbors=5, weights="distance")
clf_block.fit(Xsn[train], samples.loc[train, "block"])
block_accuracy = float(accuracy_score(samples.loc[test, "block"], clf_block.predict(Xsn[test])))

clusters = SpectralClustering(n_clusters=len(FAMILY_ID), affinity="precomputed", assign_labels="kmeans", random_state=SEED).fit_predict(adj)
elements["cluster"] = clusters
ari = float(adjusted_rand_score(elements["lattice_family"], clusters))
nmi = float(normalized_mutual_info_score(elements["lattice_family"], clusters))
try: family_silhouette = float(silhouette_score(Xn, elements["lattice_family"]))
except Exception: family_silhouette = np.nan
try: cluster_silhouette = float(silhouette_score(Xn, clusters))
except Exception: cluster_silhouette = np.nan

edge_threshold = float(np.quantile(edges_df["exchange_strength"], 0.78))
strong_edges = edges_df[edges_df["exchange_strength"] >= edge_threshold].copy()
degrees = {s: 0 for s in symbols}
weighted_degrees = {s: 0.0 for s in symbols}
for _, row in strong_edges.iterrows():
    degrees[row["source"]] += 1
    degrees[row["target"]] += 1
    weighted_degrees[row["source"]] += row["exchange_strength"]
    weighted_degrees[row["target"]] += row["exchange_strength"]
elements["strong_degree"] = elements["symbol"].map(degrees)
elements["weighted_degree"] = elements["symbol"].map(weighted_degrees)

family_rows = []
for fam in FAMILY_ID:
    fam_symbols = set(elements[elements["lattice_family"] == fam]["symbol"])
    within = edges_df[edges_df["source"].isin(fam_symbols) & edges_df["target"].isin(fam_symbols)]
    cross = edges_df[edges_df["source"].isin(fam_symbols) ^ edges_df["target"].isin(fam_symbols)]
    mean_within = float(within["exchange_strength"].mean()) if len(within) else np.nan
    mean_cross = float(cross["exchange_strength"].mean()) if len(cross) else np.nan
    contrast = float((mean_within + EPS) / (mean_cross + EPS)) if len(within) and len(cross) else np.nan
    fam_elements = elements[elements["lattice_family"] == fam]
    family_rows.append({
        "lattice_family": fam, "count": len(fam_symbols), "elements": ",".join([s for s in symbols if s in fam_symbols]),
        "mean_within_exchange": mean_within, "mean_cross_exchange": mean_cross, "exchange_contrast": contrast,
        "mean_degree": float(fam_elements["strong_degree"].mean()),
        "mean_weighted_degree": float(fam_elements["weighted_degree"].mean()),
        "mean_shell_stability": float(fam_elements["shell_stability"].mean()),
        "mean_exchange_availability": float(fam_elements["exchange_availability"].mean()),
    })
by_family = pd.DataFrame(family_rows)

def contrast_supported(fam, threshold=1.15):
    return float(by_family.loc[by_family["lattice_family"] == fam, "exchange_contrast"].iloc[0] > threshold)

transition_recovery = contrast_supported("transition_metal")
lanthanide_recovery = contrast_supported("lanthanide")
actinide_recovery = contrast_supported("actinide")
noble_isolation = float(by_family.loc[by_family["lattice_family"] == "noble_gas", "mean_degree"].iloc[0] <= 2.0)
mean_all_exchange = float(edges_df["exchange_strength"].mean())
mean_strong_exchange = float(strong_edges["exchange_strength"].mean())
network_recurrence = float(0.45 * mean_strong_exchange + 0.25 * np.mean(elements["weighted_degree"] / (elements["weighted_degree"].max() + EPS)) + 0.15 * max(ari, 0.0) + 0.15 * max(nmi, 0.0))
lattice_stability = float(0.50 * mean_strong_exchange + 0.25 * np.mean(elements["shell_stability"]) + 0.25 * network_recurrence)

if (family_accuracy >= 0.95 and block_accuracy >= 0.98 and ari >= 0.70 and nmi >= 0.80 and lattice_stability >= 0.55 and transition_recovery == 1.0 and lanthanide_recovery == 1.0 and actinide_recovery == 1.0 and noble_isolation == 1.0):
    verdict = "lattice_exchange_networks_supported"
elif family_accuracy >= 0.85 and nmi >= 0.60 and lattice_stability >= 0.45:
    verdict = "weak_lattice_exchange_networks"
else:
    verdict = "lattice_exchange_networks_not_supported"

summary = pd.DataFrame([{
    "num_elements": n, "num_samples": len(samples), "num_edges": len(edges_df), "num_strong_edges": len(strong_edges),
    "family_accuracy": family_accuracy, "block_accuracy": block_accuracy, "community_ARI": ari, "community_NMI": nmi,
    "family_silhouette": family_silhouette, "cluster_silhouette": cluster_silhouette,
    "mean_all_exchange": mean_all_exchange, "mean_strong_exchange": mean_strong_exchange, "edge_threshold": edge_threshold,
    "lattice_stability": lattice_stability, "network_recurrence": network_recurrence,
    "transition_recovery": transition_recovery, "lanthanide_recovery": lanthanide_recovery, "actinide_recovery": actinide_recovery,
    "noble_isolation": noble_isolation, "verdict": verdict,
}])

samples.to_csv(OUT / "lattice_exchange_samples.csv", index=False)
edges_df.to_csv(OUT / "lattice_exchange_edges.csv", index=False)
summary.to_csv(OUT / "lattice_exchange_summary.csv", index=False)
by_family.to_csv(OUT / "lattice_exchange_by_family.csv", index=False)
elements.to_csv(OUT / "lattice_exchange_clusters.csv", index=False)

order = elements.sort_values(["family_id", "Z"]).index.to_numpy()
ordered_adj = adj[np.ix_(order, order)]
ordered_labels = elements.iloc[order]["symbol"].tolist()
plt.figure(figsize=(11, 9))
plt.imshow(ordered_adj, aspect="auto")
plt.colorbar(label="BST exchange strength")
plt.xticks(range(n), ordered_labels, rotation=90, fontsize=6)
plt.yticks(range(n), ordered_labels, fontsize=6)
plt.title("BST lattice exchange matrix — family ordered")
plt.tight_layout()
plt.savefig(OUT / "lattice_exchange_matrix.png", dpi=220)
plt.close()

proj = PCA(n_components=2).fit_transform(Xn)
plt.figure(figsize=(10, 7))
for fam in FAMILY_ID:
    mask = elements["lattice_family"].to_numpy() == fam
    plt.scatter(proj[mask, 0], proj[mask, 1], s=34, label=fam)
    for x, y, sym in zip(proj[mask, 0], proj[mask, 1], elements.loc[mask, "symbol"]):
        plt.text(x, y, sym, fontsize=6, ha="center", va="center")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("BST lattice exchange element signatures")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "lattice_exchange_projection.png", dpi=220)
plt.close()

pos = {sym: proj[i] for i, sym in enumerate(symbols)}
plt.figure(figsize=(12, 9))
denom = strong_edges["exchange_strength"].max() - edge_threshold + EPS
for _, row in strong_edges.iterrows():
    x1, y1 = pos[row["source"]]
    x2, y2 = pos[row["target"]]
    alpha = 0.10 + 0.45 * (row["exchange_strength"] - edge_threshold) / denom
    plt.plot([x1, x2], [y1, y2], linewidth=0.7, alpha=alpha)
for fam in FAMILY_ID:
    mask = elements["lattice_family"].to_numpy() == fam
    plt.scatter(proj[mask, 0], proj[mask, 1], s=55, label=fam)
    for x, y, sym in zip(proj[mask, 0], proj[mask, 1], elements.loc[mask, "symbol"]):
        plt.text(x, y, sym, fontsize=7, ha="center", va="center")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("BST lattice exchange network — strong exchange edges")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "lattice_exchange_network.png", dpi=220)
plt.close()

print("\n=== BST LATTICE EXCHANGE NETWORKS TEST ===\n")
print(summary.to_string(index=False))
print("\nLattice exchange by family:")
print(by_family.to_string(index=False))
print("\nEmergent lattice clusters:")
print(elements[["Z", "symbol", "lattice_family", "block", "cluster", "strong_degree", "weighted_degree", "recurrence_signature", "shell_stability", "exchange_availability"]].to_string(index=False))
print(f"\n[OK] wrote {OUT / 'lattice_exchange_samples.csv'}")
print(f"[OK] wrote {OUT / 'lattice_exchange_edges.csv'}")
print(f"[OK] wrote {OUT / 'lattice_exchange_summary.csv'}")
print(f"[OK] wrote {OUT / 'lattice_exchange_by_family.csv'}")
print(f"[OK] wrote {OUT / 'lattice_exchange_clusters.csv'}")
print(f"[OK] wrote {OUT / 'lattice_exchange_matrix.png'}")
print(f"[OK] wrote {OUT / 'lattice_exchange_projection.png'}")
print(f"[OK] wrote {OUT / 'lattice_exchange_network.png'}")
print("[DONE] lattice exchange networks test complete")
