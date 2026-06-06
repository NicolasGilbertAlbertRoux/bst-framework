#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 69 — Structure Formation from Wave Clusters

Purpose:
    Test whether the validated BST vacuum wave background produces
    hierarchical structure formation.

Input:
    results/research_final/vacuum_wave_background_test/vacuum_wave_background_samples.csv

Canonical hierarchy:
    vacuum wave background
      -> W/C fluctuations
      -> structured background
      -> matter seeds
      -> clustered structures
      -> large-scale structures

Strict rules:
    - no Big Bang claim;
    - no CMB claim;
    - no galaxy claim;
    - no dark matter claim;
    - no Lambda-CDM claim;
    - no particle identity;
    - no exact W->T dictionary.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


SEED = 1729

INP = Path("results/research_final/vacuum_wave_background_test/vacuum_wave_background_samples.csv")
OUT = Path("results/research_final/structure_formation_from_wave_clusters_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "background_regime",
    "matter_architecture",
    "transition_regime",
    "binding_regime",
    "em_like_regime",
    "stress_regime",
    "metric_regime",
    "field_regime",
    "cluster_class",
    "is_valid_field",
    "field_density",
    "field_coherence",
    "field_range",
    "field_flux",
    "field_presence_score",
    "gauge_invariant_norm",
    "metric_potential",
    "effective_curvature_score",
    "stress_energy_coupling",
    "em_interaction_score",
    "binding_stability_proxy",
    "weak_like_transition_score",
    "matter_architecture_score",
    "hierarchy_consistency_score",
    "architecture_stability_score",
    "architecture_complexity_score",
    "vacuum_wave_density",
    "vacuum_coherence",
    "background_fluctuation",
    "seed_potential",
    "background_order_parameter",
    "vacuum_noise_floor",
    "matter_seed_score",
    "background_stability",
]


def load_input():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing required input: {INP}\n"
            "Run experiments/68_vacuum_wave_background/vacuum_wave_background_test.py first."
        )

    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")

    return df


def classify_structure(row):
    if row["background_regime"] == "background_vacuum":
        return "diffuse_structure"

    if row["background_regime"] == "structured_background":
        return "clustered_structure"

    if row["background_regime"] == "matter_seed_background":
        return "large_scale_structure"

    return "unclassified_structure"


def compute_structure_features(df):
    df = df.copy()

    f_presence = df["field_presence_score"].astype(float)
    f_density = df["field_density"].astype(float)
    f_coherence = df["field_coherence"].astype(float)
    f_range = df["field_range"].astype(float)
    f_flux = df["field_flux"].astype(float)

    gauge = df["gauge_invariant_norm"].astype(float)
    metric = df["metric_potential"].astype(float)
    curvature = df["effective_curvature_score"].astype(float)
    stress = df["stress_energy_coupling"].astype(float)
    em = df["em_interaction_score"].astype(float)
    binding = df["binding_stability_proxy"].astype(float)
    transition = df["weak_like_transition_score"].astype(float)

    matter = df["matter_architecture_score"].astype(float)
    hierarchy = df["hierarchy_consistency_score"].astype(float)
    arch_stability = df["architecture_stability_score"].astype(float)
    arch_complexity = df["architecture_complexity_score"].astype(float)

    vacuum_density = df["vacuum_wave_density"].astype(float)
    vacuum_coherence = df["vacuum_coherence"].astype(float)
    background_fluctuation = df["background_fluctuation"].astype(float)
    seed_potential = df["seed_potential"].astype(float)
    background_order = df["background_order_parameter"].astype(float)
    noise_floor = df["vacuum_noise_floor"].astype(float)
    matter_seed = df["matter_seed_score"].astype(float)
    background_stability = df["background_stability"].astype(float)

    df["structure_regime"] = df.apply(classify_structure, axis=1)

    df["structure_seed_density"] = np.clip(
        0.24 * matter_seed
        + 0.20 * seed_potential
        + 0.16 * background_fluctuation
        + 0.14 * vacuum_density
        + 0.12 * f_presence
        + 0.08 * metric
        + 0.06 * stress,
        0,
        5,
    )

    df["structure_coherence"] = np.clip(
        0.26 * vacuum_coherence
        + 0.22 * background_order
        + 0.18 * hierarchy
        + 0.14 * f_coherence
        + 0.10 * gauge
        + 0.10 * background_stability,
        0,
        5,
    )

    df["clustering_potential"] = np.clip(
        0.24 * df["structure_seed_density"]
        + 0.22 * df["structure_coherence"]
        + 0.18 * matter
        + 0.14 * binding
        + 0.12 * curvature
        + 0.10 * transition,
        0,
        5,
    )

    df["network_connectivity"] = np.clip(
        0.22 * f_range
        + 0.20 * f_flux
        + 0.18 * em
        + 0.16 * stress
        + 0.14 * metric
        + 0.10 * arch_complexity,
        0,
        5,
    )

    df["large_scale_coherence"] = np.clip(
        0.24 * df["network_connectivity"]
        + 0.22 * df["clustering_potential"]
        + 0.18 * arch_stability
        + 0.14 * background_stability
        + 0.12 * hierarchy
        + 0.10 * matter_seed,
        0,
        5,
    )

    df["diffusion_suppression"] = np.clip(
        1.0
        - (
            0.22 * df["structure_seed_density"]
            + 0.20 * df["clustering_potential"]
            + 0.18 * df["large_scale_coherence"]
            + 0.14 * matter
            + 0.12 * binding
            + 0.08 * curvature
            + 0.06 * stress
        ),
        0,
        1,
    )

    df["structure_growth_score"] = np.clip(
        0.24 * df["structure_seed_density"]
        + 0.22 * df["clustering_potential"]
        + 0.20 * df["large_scale_coherence"]
        + 0.14 * df["network_connectivity"]
        + 0.10 * arch_stability
        + 0.10 * (1.0 - df["diffusion_suppression"]),
        0,
        5,
    )

    df["structure_stability"] = np.clip(
        0.26 * df["large_scale_coherence"]
        + 0.20 * background_stability
        + 0.18 * arch_stability
        + 0.14 * hierarchy
        + 0.12 * binding
        + 0.10 * df["clustering_potential"],
        0,
        5,
    )

    df["structure_supported"] = (
        ((df["structure_regime"] == "diffuse_structure") & (df["structure_growth_score"] < 0.35))
        | ((df["structure_regime"] == "clustered_structure") & (df["structure_growth_score"] >= 0.35))
        | ((df["structure_regime"] == "large_scale_structure") & (df["structure_growth_score"] >= 0.75))
    ).astype(int)

    df["clustered_structure_supported"] = (
        (df["structure_regime"] == "clustered_structure")
        & (df["structure_growth_score"] >= 0.35)
    ).astype(int)

    df["large_scale_structure_supported"] = (
        (df["structure_regime"] == "large_scale_structure")
        & (df["structure_growth_score"] >= 0.75)
    ).astype(int)

    return df


def supervised_accuracy(df, target):
    features = [
        "field_presence_score",
        "gauge_invariant_norm",
        "metric_potential",
        "effective_curvature_score",
        "stress_energy_coupling",
        "em_interaction_score",
        "binding_stability_proxy",
        "weak_like_transition_score",
        "matter_architecture_score",
        "hierarchy_consistency_score",
        "architecture_stability_score",
        "architecture_complexity_score",
        "vacuum_wave_density",
        "vacuum_coherence",
        "background_fluctuation",
        "seed_potential",
        "background_order_parameter",
        "matter_seed_score",
        "background_stability",
        "structure_seed_density",
        "structure_coherence",
        "clustering_potential",
        "network_connectivity",
        "large_scale_coherence",
        "diffusion_suppression",
        "structure_growth_score",
        "structure_stability",
    ]

    if len(df[target].unique()) < 2:
        return np.nan

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=340,
        max_depth=9,
        random_state=SEED,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    return float(accuracy_score(y_test, pred))


def safe_corr(a, b):
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return np.nan


def safe_silhouette(df, label_col, cols):
    try:
        if len(df[label_col].unique()) < 2:
            return np.nan
        return float(silhouette_score(df[cols], df[label_col]))
    except Exception:
        return np.nan


def summarize(df):
    diffuse = df[df["structure_regime"] == "diffuse_structure"]
    clustered = df[df["structure_regime"] == "clustered_structure"]
    large = df[df["structure_regime"] == "large_scale_structure"]

    structure_accuracy = supervised_accuracy(df, "structure_supported")

    clustered_subset = df[df["structure_regime"].isin(["clustered_structure", "diffuse_structure"])]
    large_subset = df[df["structure_regime"].isin(["large_scale_structure", "diffuse_structure"])]

    clustered_accuracy = supervised_accuracy(clustered_subset, "clustered_structure_supported")
    large_accuracy = supervised_accuracy(large_subset, "large_scale_structure_supported")

    diffuse_support = float((diffuse["structure_growth_score"] < 0.35).mean()) if len(diffuse) else 0.0
    clustered_support = float((clustered["structure_growth_score"] >= 0.35).mean()) if len(clustered) else 0.0
    large_support = float((large["structure_growth_score"] >= 0.75).mean()) if len(large) else 0.0

    mean_diffuse = float(diffuse["structure_growth_score"].mean()) if len(diffuse) else 0.0
    mean_clustered = float(clustered["structure_growth_score"].mean()) if len(clustered) else 0.0
    mean_large = float(large["structure_growth_score"].mean()) if len(large) else 0.0

    clustered_contrast = mean_clustered / max(mean_diffuse, 1e-9)
    large_contrast = mean_large / max(mean_diffuse, 1e-9)

    structure_silhouette = safe_silhouette(
        df,
        "structure_regime",
        [
            "structure_seed_density",
            "structure_coherence",
            "clustering_potential",
            "network_connectivity",
            "large_scale_coherence",
            "structure_growth_score",
            "structure_stability",
        ],
    )

    background_to_structure_corr = safe_corr(df["matter_seed_score"], df["structure_growth_score"])
    seed_to_structure_corr = safe_corr(df["seed_potential"], df["structure_growth_score"])
    matter_to_structure_corr = safe_corr(df["matter_architecture_score"], df["structure_growth_score"])
    stability_to_structure_corr = safe_corr(df["architecture_stability_score"], df["structure_stability"])
    coherence_to_structure_corr = safe_corr(df["vacuum_coherence"], df["structure_coherence"])
    metric_to_structure_corr = safe_corr(df["metric_potential"], df["structure_growth_score"])
    stress_to_structure_corr = safe_corr(df["stress_energy_coupling"], df["structure_growth_score"])

    verdict = (
        "structure_formation_from_wave_clusters_supported"
        if (
            structure_accuracy >= 0.98
            and diffuse_support >= 0.95
            and clustered_support >= 0.95
            and large_support >= 0.90
            and mean_diffuse < mean_clustered < mean_large
            and clustered_contrast > 1.5
            and large_contrast > 2.5
            and background_to_structure_corr > 0.80
            and seed_to_structure_corr > 0.80
            and matter_to_structure_corr > 0.80
            and stability_to_structure_corr > 0.80
            and coherence_to_structure_corr > 0.80
        )
        else "structure_formation_from_wave_clusters_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_diffuse_structure": len(diffuse),
        "num_clustered_structure": len(clustered),
        "num_large_scale_structure": len(large),
        "structure_accuracy": structure_accuracy,
        "clustered_structure_accuracy": clustered_accuracy,
        "large_scale_structure_accuracy": large_accuracy,
        "structure_silhouette": structure_silhouette,
        "diffuse_support": diffuse_support,
        "clustered_support": clustered_support,
        "large_scale_support": large_support,
        "mean_diffuse_structure_score": mean_diffuse,
        "mean_clustered_structure_score": mean_clustered,
        "mean_large_scale_structure_score": mean_large,
        "clustered_contrast": clustered_contrast,
        "large_scale_contrast": large_contrast,
        "background_to_structure_corr": background_to_structure_corr,
        "seed_to_structure_corr": seed_to_structure_corr,
        "matter_to_structure_corr": matter_to_structure_corr,
        "stability_to_structure_corr": stability_to_structure_corr,
        "coherence_to_structure_corr": coherence_to_structure_corr,
        "metric_to_structure_corr": metric_to_structure_corr,
        "stress_to_structure_corr": stress_to_structure_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "galaxy_claim": False,
        "dark_matter_claim": False,
        "lambda_cdm_claim": False,
        "standard_cosmology_claim": False,
        "safe_hierarchy": "vacuum wave background -> W/C fluctuations -> matter seeds -> clustered structures -> large-scale structures",
        "verdict": verdict,
    }])


def aggregate(df):
    by_regime = df.groupby("structure_regime").agg(
        count=("structure_regime", "count"),
        mean_structure_seed_density=("structure_seed_density", "mean"),
        mean_structure_coherence=("structure_coherence", "mean"),
        mean_clustering_potential=("clustering_potential", "mean"),
        mean_network_connectivity=("network_connectivity", "mean"),
        mean_large_scale_coherence=("large_scale_coherence", "mean"),
        mean_diffusion_suppression=("diffusion_suppression", "mean"),
        mean_structure_growth_score=("structure_growth_score", "mean"),
        mean_structure_stability=("structure_stability", "mean"),
        support_ratio=("structure_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        structure_regime=("structure_regime", lambda x: x.mode().iloc[0]),
        mean_structure_seed_density=("structure_seed_density", "mean"),
        mean_clustering_potential=("clustering_potential", "mean"),
        mean_network_connectivity=("network_connectivity", "mean"),
        mean_structure_growth_score=("structure_growth_score", "mean"),
        mean_structure_stability=("structure_stability", "mean"),
    ).reset_index()

    by_background = df.groupby("background_regime").agg(
        count=("background_regime", "count"),
        structure_regime=("structure_regime", lambda x: x.mode().iloc[0]),
        mean_structure_seed_density=("structure_seed_density", "mean"),
        mean_clustering_potential=("clustering_potential", "mean"),
        mean_structure_growth_score=("structure_growth_score", "mean"),
        mean_structure_stability=("structure_stability", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_background


def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("structure_regime"):
        plt.scatter(
            sub["clustering_potential"],
            sub["structure_growth_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("clustering_potential")
    plt.ylabel("structure_growth_score")
    plt.title("BST structure formation from wave clusters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "structure_formation_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("structure_regime"):
        plt.scatter(
            sub["network_connectivity"],
            sub["large_scale_coherence"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("network_connectivity")
    plt.ylabel("large_scale_coherence")
    plt.title("BST network connectivity / large-scale coherence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "structure_network_coherence.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(valid["cluster_class"].astype(str), valid["mean_structure_growth_score"])
    plt.xlabel("cluster_class")
    plt.ylabel("mean_structure_growth_score")
    plt.title("BST structure growth by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "structure_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="structure_regime",
        values="structure_growth_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="structure_growth_score")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("structure_regime")
    plt.ylabel("cluster_class")
    plt.title("BST structure formation heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "structure_formation_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 69 STRUCTURE FORMATION FROM WAVE CLUSTERS TEST ===\n")

    base = load_input()
    samples = compute_structure_features(base)

    summary = summarize(samples)
    by_regime, by_cluster, by_background = aggregate(samples)

    samples.to_csv(OUT / "structure_formation_samples.csv", index=False)
    summary.to_csv(OUT / "structure_formation_summary.csv", index=False)
    by_regime.to_csv(OUT / "structure_formation_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "structure_formation_by_cluster.csv", index=False)
    by_background.to_csv(OUT / "structure_formation_by_background_regime.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nStructure formation by regime:")
    print(by_regime.to_string(index=False))

    print("\nStructure formation by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nStructure formation by background regime:")
    print(by_background.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'structure_formation_samples.csv'}")
    print(f"[OK] wrote {OUT / 'structure_formation_summary.csv'}")
    print(f"[OK] wrote {OUT / 'structure_formation_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'structure_formation_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'structure_formation_by_background_regime.csv'}")
    print(f"[OK] wrote {OUT / 'structure_formation_projection.png'}")
    print(f"[OK] wrote {OUT / 'structure_network_coherence.png'}")
    print(f"[OK] wrote {OUT / 'structure_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'structure_formation_heatmap.png'}")
    print("[DONE] structure formation from wave clusters test complete")


if __name__ == "__main__":
    main()