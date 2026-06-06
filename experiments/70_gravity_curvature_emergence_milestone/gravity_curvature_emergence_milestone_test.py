#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 1729

INP = Path("results/research_final/structure_formation_from_wave_clusters_test/structure_formation_samples.csv")
OUT = Path("results/research_final/gravity_curvature_emergence_milestone_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "structure_regime", "background_regime", "matter_architecture", "cluster_class",
    "field_presence_score", "gauge_invariant_norm", "metric_potential",
    "effective_curvature_score", "curvature_density", "curvature_coherence",
    "stress_energy_coupling", "energy_density_proxy", "matter_seed_score",
    "structure_growth_score", "structure_stability", "large_scale_coherence",
    "network_connectivity", "clustering_potential", "structure_seed_density",
    "hierarchy_consistency_score", "architecture_stability_score",
]

def load_input():
    if not INP.exists():
        raise FileNotFoundError(f"Missing required input: {INP}\nRun test 69 first.")
    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")
    return df

def classify_gravity_regime(row):
    if row["structure_regime"] == "diffuse_structure":
        return "flat_background_geometry"
    if row["structure_regime"] == "clustered_structure":
        return "localized_gravity_well"
    if row["structure_regime"] == "large_scale_structure":
        return "extended_curvature_network"
    return "unclassified_gravity"

def compute_gravity_features(df):
    df = df.copy()

    mp = df["metric_potential"].astype(float)
    curv = df["effective_curvature_score"].astype(float)
    cd = df["curvature_density"].astype(float)
    cc = df["curvature_coherence"].astype(float)
    se = df["stress_energy_coupling"].astype(float)
    ed = df["energy_density_proxy"].astype(float)
    seed = df["matter_seed_score"].astype(float)
    growth = df["structure_growth_score"].astype(float)
    stability = df["structure_stability"].astype(float)
    lsc = df["large_scale_coherence"].astype(float)
    net = df["network_connectivity"].astype(float)
    clust = df["clustering_potential"].astype(float)
    sden = df["structure_seed_density"].astype(float)
    hier = df["hierarchy_consistency_score"].astype(float)
    arch_stab = df["architecture_stability_score"].astype(float)
    field = df["field_presence_score"].astype(float)
    gauge = df["gauge_invariant_norm"].astype(float)

    df["gravity_regime"] = df.apply(classify_gravity_regime, axis=1)

    df["mass_energy_proxy"] = np.clip(
        0.24 * ed + 0.22 * se + 0.18 * seed + 0.14 * growth
        + 0.12 * sden + 0.10 * field,
        0, 5,
    )

    df["geometry_potential"] = np.clip(
        0.26 * mp + 0.22 * curv + 0.18 * cd + 0.14 * cc
        + 0.10 * gauge + 0.10 * hier,
        0, 5,
    )

    df["curvature_gradient_proxy"] = np.clip(
        0.24 * cd + 0.22 * curv + 0.18 * clust + 0.14 * net
        + 0.12 * lsc + 0.10 * growth,
        0, 5,
    )

    df["gravitational_binding_proxy"] = np.clip(
        0.28 * df["mass_energy_proxy"]
        + 0.24 * df["geometry_potential"]
        + 0.18 * df["curvature_gradient_proxy"]
        + 0.14 * stability
        + 0.10 * arch_stab
        + 0.06 * hier,
        0, 5,
    )

    df["geodesic_coherence_proxy"] = np.clip(
        0.26 * lsc + 0.22 * cc + 0.18 * hier + 0.14 * stability
        + 0.12 * net + 0.08 * gauge,
        0, 5,
    )

    df["gravity_emergence_score"] = np.clip(
        0.28 * df["gravitational_binding_proxy"]
        + 0.24 * df["geometry_potential"]
        + 0.18 * df["geodesic_coherence_proxy"]
        + 0.14 * df["curvature_gradient_proxy"]
        + 0.10 * growth
        + 0.06 * seed,
        0, 5,
    )

    df["gravity_suppression_proxy"] = np.clip(
        1.0 - (
            0.24 * df["gravity_emergence_score"]
            + 0.20 * df["geometry_potential"]
            + 0.18 * df["mass_energy_proxy"]
            + 0.14 * growth
            + 0.12 * curv
            + 0.12 * se
        ),
        0, 1,
    )

    df["gravity_supported"] = (
        ((df["gravity_regime"] == "flat_background_geometry") & (df["gravity_emergence_score"] < 0.35))
        | ((df["gravity_regime"] == "localized_gravity_well") & (df["gravity_emergence_score"] >= 0.35))
        | ((df["gravity_regime"] == "extended_curvature_network") & (df["gravity_emergence_score"] >= 0.75))
    ).astype(int)

    df["localized_gravity_supported"] = (
        (df["gravity_regime"] == "localized_gravity_well")
        & (df["gravity_emergence_score"] >= 0.35)
    ).astype(int)

    df["extended_gravity_supported"] = (
        (df["gravity_regime"] == "extended_curvature_network")
        & (df["gravity_emergence_score"] >= 0.75)
    ).astype(int)

    return df

def supervised_accuracy(df, target):
    features = [
        "field_presence_score", "gauge_invariant_norm", "metric_potential",
        "effective_curvature_score", "curvature_density", "curvature_coherence",
        "stress_energy_coupling", "energy_density_proxy", "matter_seed_score",
        "structure_growth_score", "structure_stability", "large_scale_coherence",
        "network_connectivity", "clustering_potential", "structure_seed_density",
        "mass_energy_proxy", "geometry_potential", "curvature_gradient_proxy",
        "gravitational_binding_proxy", "geodesic_coherence_proxy",
        "gravity_emergence_score", "gravity_suppression_proxy",
    ]

    if len(df[target].unique()) < 2:
        return np.nan

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=360, max_depth=9, random_state=SEED, class_weight="balanced"
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
    flat = df[df["gravity_regime"] == "flat_background_geometry"]
    local = df[df["gravity_regime"] == "localized_gravity_well"]
    extended = df[df["gravity_regime"] == "extended_curvature_network"]

    gravity_accuracy = supervised_accuracy(df, "gravity_supported")

    local_subset = df[df["gravity_regime"].isin(["localized_gravity_well", "flat_background_geometry"])]
    extended_subset = df[df["gravity_regime"].isin(["extended_curvature_network", "flat_background_geometry"])]

    local_accuracy = supervised_accuracy(local_subset, "localized_gravity_supported")
    extended_accuracy = supervised_accuracy(extended_subset, "extended_gravity_supported")

    flat_support = float((flat["gravity_emergence_score"] < 0.35).mean()) if len(flat) else 0.0
    local_support = float((local["gravity_emergence_score"] >= 0.35).mean()) if len(local) else 0.0
    extended_support = float((extended["gravity_emergence_score"] >= 0.75).mean()) if len(extended) else 0.0

    mean_flat = float(flat["gravity_emergence_score"].mean()) if len(flat) else 0.0
    mean_local = float(local["gravity_emergence_score"].mean()) if len(local) else 0.0
    mean_extended = float(extended["gravity_emergence_score"].mean()) if len(extended) else 0.0

    local_contrast = mean_local / max(mean_flat, 1e-9)
    extended_contrast = mean_extended / max(mean_flat, 1e-9)

    gravity_silhouette = safe_silhouette(
        df,
        "gravity_regime",
        [
            "mass_energy_proxy", "geometry_potential", "curvature_gradient_proxy",
            "gravitational_binding_proxy", "geodesic_coherence_proxy",
            "gravity_emergence_score",
        ],
    )

    mass_curvature_corr = safe_corr(df["mass_energy_proxy"], df["geometry_potential"])
    energy_gravity_corr = safe_corr(df["energy_density_proxy"], df["gravity_emergence_score"])
    stress_gravity_corr = safe_corr(df["stress_energy_coupling"], df["gravity_emergence_score"])
    metric_gravity_corr = safe_corr(df["metric_potential"], df["gravity_emergence_score"])
    curvature_gravity_corr = safe_corr(df["effective_curvature_score"], df["gravity_emergence_score"])
    structure_gravity_corr = safe_corr(df["structure_growth_score"], df["gravity_emergence_score"])
    coherence_geodesic_corr = safe_corr(df["large_scale_coherence"], df["geodesic_coherence_proxy"])

    verdict = (
        "gravity_curvature_emergence_supported"
        if (
            gravity_accuracy >= 0.98
            and flat_support >= 0.95
            and local_support >= 0.95
            and extended_support >= 0.90
            and mean_flat < mean_local < mean_extended
            and local_contrast > 1.5
            and extended_contrast > 2.5
            and mass_curvature_corr > 0.80
            and stress_gravity_corr > 0.80
            and metric_gravity_corr > 0.80
            and curvature_gravity_corr > 0.80
            and structure_gravity_corr > 0.80
        )
        else "gravity_curvature_emergence_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_flat_background_geometry": len(flat),
        "num_localized_gravity_well": len(local),
        "num_extended_curvature_network": len(extended),
        "gravity_accuracy": gravity_accuracy,
        "localized_gravity_accuracy": local_accuracy,
        "extended_gravity_accuracy": extended_accuracy,
        "gravity_silhouette": gravity_silhouette,
        "flat_background_support": flat_support,
        "localized_gravity_support": local_support,
        "extended_curvature_support": extended_support,
        "mean_flat_gravity_score": mean_flat,
        "mean_local_gravity_score": mean_local,
        "mean_extended_gravity_score": mean_extended,
        "local_gravity_contrast": local_contrast,
        "extended_gravity_contrast": extended_contrast,
        "mass_curvature_corr": mass_curvature_corr,
        "energy_gravity_corr": energy_gravity_corr,
        "stress_gravity_corr": stress_gravity_corr,
        "metric_gravity_corr": metric_gravity_corr,
        "curvature_gravity_corr": curvature_gravity_corr,
        "structure_gravity_corr": structure_gravity_corr,
        "coherence_geodesic_corr": coherence_geodesic_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "general_relativity_claim": False,
        "dark_matter_claim": False,
        "galaxy_claim": False,
        "standard_cosmology_claim": False,
        "safe_hierarchy": "vacuum wave background -> structure formation -> mass-energy proxy -> metric/curvature proxy -> gravity-like emergence",
        "verdict": verdict,
    }])

def aggregate(df):
    by_regime = df.groupby("gravity_regime").agg(
        count=("gravity_regime", "count"),
        mean_mass_energy=("mass_energy_proxy", "mean"),
        mean_geometry_potential=("geometry_potential", "mean"),
        mean_curvature_gradient=("curvature_gradient_proxy", "mean"),
        mean_gravitational_binding=("gravitational_binding_proxy", "mean"),
        mean_geodesic_coherence=("geodesic_coherence_proxy", "mean"),
        mean_gravity_emergence=("gravity_emergence_score", "mean"),
        mean_gravity_suppression=("gravity_suppression_proxy", "mean"),
        support_ratio=("gravity_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        gravity_regime=("gravity_regime", lambda x: x.mode().iloc[0]),
        mean_mass_energy=("mass_energy_proxy", "mean"),
        mean_geometry_potential=("geometry_potential", "mean"),
        mean_gravity_emergence=("gravity_emergence_score", "mean"),
    ).reset_index()

    by_structure = df.groupby("structure_regime").agg(
        count=("structure_regime", "count"),
        gravity_regime=("gravity_regime", lambda x: x.mode().iloc[0]),
        mean_mass_energy=("mass_energy_proxy", "mean"),
        mean_geometry_potential=("geometry_potential", "mean"),
        mean_gravity_emergence=("gravity_emergence_score", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_structure

def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("gravity_regime"):
        plt.scatter(
            sub["mass_energy_proxy"],
            sub["geometry_potential"],
            s=8, alpha=0.45, label=name,
        )
    plt.xlabel("mass_energy_proxy")
    plt.ylabel("geometry_potential")
    plt.title("BST mass-energy to geometry potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "gravity_mass_geometry_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("gravity_regime"):
        plt.scatter(
            sub["curvature_gradient_proxy"],
            sub["gravity_emergence_score"],
            s=8, alpha=0.45, label=name,
        )
    plt.xlabel("curvature_gradient_proxy")
    plt.ylabel("gravity_emergence_score")
    plt.title("BST curvature to gravity-like emergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "gravity_curvature_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(valid["cluster_class"].astype(str), valid["mean_gravity_emergence"])
    plt.xlabel("cluster_class")
    plt.ylabel("mean_gravity_emergence")
    plt.title("BST gravity-like emergence by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "gravity_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="gravity_regime",
        values="gravity_emergence_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="gravity_emergence_score")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("gravity_regime")
    plt.ylabel("cluster_class")
    plt.title("BST gravity / curvature emergence heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "gravity_curvature_heatmap.png", dpi=220)
    plt.close()

def main():
    print("\n=== BST 70 GRAVITY / CURVATURE EMERGENCE MILESTONE TEST ===\n")

    base = load_input()
    samples = compute_gravity_features(base)
    summary = summarize(samples)
    by_regime, by_cluster, by_structure = aggregate(samples)

    samples.to_csv(OUT / "gravity_curvature_samples.csv", index=False)
    summary.to_csv(OUT / "gravity_curvature_summary.csv", index=False)
    by_regime.to_csv(OUT / "gravity_curvature_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "gravity_curvature_by_cluster.csv", index=False)
    by_structure.to_csv(OUT / "gravity_curvature_by_structure_regime.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))
    print("\nGravity / curvature by regime:")
    print(by_regime.to_string(index=False))
    print("\nGravity / curvature by cluster:")
    print(by_cluster.to_string(index=False))
    print("\nGravity / curvature by structure regime:")
    print(by_structure.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'gravity_curvature_samples.csv'}")
    print(f"[OK] wrote {OUT / 'gravity_curvature_summary.csv'}")
    print(f"[OK] wrote {OUT / 'gravity_curvature_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'gravity_curvature_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'gravity_curvature_by_structure_regime.csv'}")
    print(f"[OK] wrote {OUT / 'gravity_mass_geometry_projection.png'}")
    print(f"[OK] wrote {OUT / 'gravity_curvature_projection.png'}")
    print(f"[OK] wrote {OUT / 'gravity_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'gravity_curvature_heatmap.png'}")
    print("[DONE] gravity / curvature emergence milestone test complete")

if __name__ == "__main__":
    main()