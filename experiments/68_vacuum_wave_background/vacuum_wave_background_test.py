#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 68 — Vacuum Wave Background

Purpose:
    Test whether the completed BST matter-chain can be read backwards as
    emerging from a vacuum wave background.

Input:
    results/research_final/full_particle_atom_matter_chain_test/full_chain_samples.csv

Canonical hierarchy:
    vacuum wave background
      -> W/C fluctuations
      -> field
      -> gauge-like invariants
      -> metric proxy
      -> stress-energy proxy
      -> interactions
      -> matter architecture

Strict rules:
    - no Big Bang claim;
    - no CMB claim;
    - no standard cosmology claim;
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

INP = Path("results/research_final/full_particle_atom_matter_chain_test/full_chain_samples.csv")
OUT = Path("results/research_final/vacuum_wave_background_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
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
]


def load_input():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing required input: {INP}\n"
            "Run experiments/67_full_particle_atom_matter_chain/full_particle_atom_matter_chain_test.py first."
        )

    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")

    return df


def classify_background(row):
    if row["matter_architecture"] == "vacuum_architecture":
        return "background_vacuum"

    if row["matter_architecture"] == "localized_matter_architecture":
        return "structured_background"

    if row["matter_architecture"] == "extended_matter_architecture":
        return "matter_seed_background"

    return "unclassified_background"


def compute_background_features(df):
    df = df.copy()

    fd = df["field_density"].astype(float)
    fc = df["field_coherence"].astype(float)
    fr = df["field_range"].astype(float)
    ff = df["field_flux"].astype(float)
    fp = df["field_presence_score"].astype(float)
    gn = df["gauge_invariant_norm"].astype(float)
    mp = df["metric_potential"].astype(float)
    curv = df["effective_curvature_score"].astype(float)
    se = df["stress_energy_coupling"].astype(float)
    em = df["em_interaction_score"].astype(float)
    bind = df["binding_stability_proxy"].astype(float)
    trans = df["weak_like_transition_score"].astype(float)
    arch = df["matter_architecture_score"].astype(float)
    cons = df["hierarchy_consistency_score"].astype(float)
    stab = df["architecture_stability_score"].astype(float)
    comp = df["architecture_complexity_score"].astype(float)

    df["background_regime"] = df.apply(classify_background, axis=1)

    df["vacuum_wave_density"] = np.clip(
        0.26 * fd
        + 0.22 * fp
        + 0.16 * fc
        + 0.14 * gn
        + 0.12 * ff
        + 0.10 * fr,
        0,
        4,
    )

    df["vacuum_coherence"] = np.clip(
        0.30 * fc
        + 0.22 * cons
        + 0.18 * gn
        + 0.14 * fp
        + 0.10 * stab
        + 0.06 * comp,
        0,
        2,
    )

    df["background_fluctuation"] = np.clip(
        0.24 * df["vacuum_wave_density"]
        + 0.20 * mp
        + 0.18 * curv
        + 0.14 * se
        + 0.12 * em
        + 0.12 * trans,
        0,
        5,
    )

    df["seed_potential"] = np.clip(
        0.22 * df["background_fluctuation"]
        + 0.20 * bind
        + 0.18 * arch
        + 0.14 * stab
        + 0.12 * comp
        + 0.08 * trans
        + 0.06 * se,
        0,
        5,
    )

    df["background_order_parameter"] = np.clip(
        0.28 * df["vacuum_coherence"]
        + 0.22 * df["seed_potential"]
        + 0.18 * cons
        + 0.14 * fp
        + 0.10 * gn
        + 0.08 * bind,
        0,
        5,
    )

    df["vacuum_noise_floor"] = np.clip(
        1.0
        - (
            0.22 * df["vacuum_wave_density"]
            + 0.20 * df["background_fluctuation"]
            + 0.18 * df["seed_potential"]
            + 0.14 * arch
            + 0.12 * bind
            + 0.08 * se
            + 0.06 * curv
        ),
        0,
        1,
    )

    df["matter_seed_score"] = np.clip(
        0.28 * df["seed_potential"]
        + 0.22 * df["background_order_parameter"]
        + 0.18 * arch
        + 0.14 * bind
        + 0.10 * trans
        + 0.08 * curv,
        0,
        5,
    )

    df["background_stability"] = np.clip(
        0.26 * df["vacuum_coherence"]
        + 0.22 * cons
        + 0.18 * stab
        + 0.14 * (1.0 - df["vacuum_noise_floor"])
        + 0.12 * df["background_order_parameter"]
        + 0.08 * bind,
        0,
        5,
    )

    df["background_supported"] = (
        ((df["background_regime"] == "background_vacuum") & (df["matter_seed_score"] < 0.35))
        | ((df["background_regime"] == "structured_background") & (df["matter_seed_score"] >= 0.35))
        | ((df["background_regime"] == "matter_seed_background") & (df["matter_seed_score"] >= 0.75))
    ).astype(int)

    df["structured_background_supported"] = (
        (df["background_regime"] == "structured_background")
        & (df["matter_seed_score"] >= 0.35)
    ).astype(int)

    df["matter_seed_background_supported"] = (
        (df["background_regime"] == "matter_seed_background")
        & (df["matter_seed_score"] >= 0.75)
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
        "vacuum_noise_floor",
        "matter_seed_score",
        "background_stability",
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
        n_estimators=320,
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
    vacuum = df[df["background_regime"] == "background_vacuum"]
    structured = df[df["background_regime"] == "structured_background"]
    seed = df[df["background_regime"] == "matter_seed_background"]

    background_accuracy = supervised_accuracy(df, "background_supported")

    structured_subset = df[df["background_regime"].isin(["structured_background", "background_vacuum"])]
    seed_subset = df[df["background_regime"].isin(["matter_seed_background", "background_vacuum"])]

    structured_accuracy = supervised_accuracy(structured_subset, "structured_background_supported")
    seed_accuracy = supervised_accuracy(seed_subset, "matter_seed_background_supported")

    vacuum_support = float((vacuum["matter_seed_score"] < 0.35).mean()) if len(vacuum) else 0.0
    structured_support = float((structured["matter_seed_score"] >= 0.35).mean()) if len(structured) else 0.0
    seed_support = float((seed["matter_seed_score"] >= 0.75).mean()) if len(seed) else 0.0

    mean_vacuum = float(vacuum["matter_seed_score"].mean()) if len(vacuum) else 0.0
    mean_structured = float(structured["matter_seed_score"].mean()) if len(structured) else 0.0
    mean_seed = float(seed["matter_seed_score"].mean()) if len(seed) else 0.0

    structured_contrast = mean_structured / max(mean_vacuum, 1e-9)
    seed_contrast = mean_seed / max(mean_vacuum, 1e-9)

    background_silhouette = safe_silhouette(
        df,
        "background_regime",
        [
            "vacuum_wave_density",
            "vacuum_coherence",
            "background_fluctuation",
            "seed_potential",
            "background_order_parameter",
            "matter_seed_score",
            "background_stability",
        ],
    )

    vacuum_to_field_corr = safe_corr(df["vacuum_wave_density"], df["field_presence_score"])
    background_to_metric_corr = safe_corr(df["background_fluctuation"], df["metric_potential"])
    background_to_stress_corr = safe_corr(df["background_fluctuation"], df["stress_energy_coupling"])
    background_to_matter_corr = safe_corr(df["matter_seed_score"], df["matter_architecture_score"])
    seed_generation_corr = safe_corr(df["seed_potential"], df["matter_architecture_score"])
    background_stability_corr = safe_corr(df["background_stability"], df["architecture_stability_score"])

    verdict = (
        "vacuum_wave_background_supported"
        if (
            background_accuracy >= 0.98
            and vacuum_support >= 0.95
            and structured_support >= 0.95
            and seed_support >= 0.90
            and mean_vacuum < mean_structured < mean_seed
            and structured_contrast > 1.5
            and seed_contrast > 2.5
            and vacuum_to_field_corr > 0.80
            and background_to_metric_corr > 0.80
            and background_to_stress_corr > 0.80
            and background_to_matter_corr > 0.80
            and seed_generation_corr > 0.80
        )
        else "vacuum_wave_background_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_background_vacuum": len(vacuum),
        "num_structured_background": len(structured),
        "num_matter_seed_background": len(seed),
        "background_accuracy": background_accuracy,
        "structured_background_accuracy": structured_accuracy,
        "matter_seed_background_accuracy": seed_accuracy,
        "background_silhouette": background_silhouette,
        "vacuum_support": vacuum_support,
        "structured_support": structured_support,
        "matter_seed_support": seed_support,
        "mean_vacuum_seed_score": mean_vacuum,
        "mean_structured_seed_score": mean_structured,
        "mean_matter_seed_score": mean_seed,
        "structured_contrast": structured_contrast,
        "matter_seed_contrast": seed_contrast,
        "vacuum_to_field_corr": vacuum_to_field_corr,
        "background_to_metric_corr": background_to_metric_corr,
        "background_to_stress_corr": background_to_stress_corr,
        "background_to_matter_corr": background_to_matter_corr,
        "seed_generation_corr": seed_generation_corr,
        "background_stability_corr": background_stability_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "electron_identity_used": False,
        "proton_identity_used": False,
        "neutron_identity_used": False,
        "quark_identity_used": False,
        "atom_identity_used": False,
        "standard_cosmology_claim": False,
        "safe_hierarchy": "vacuum wave background -> W/C fluctuations -> field -> gauge-like invariants -> metric proxy -> stress-energy proxy -> interactions -> matter architecture",
        "verdict": verdict,
    }])


def aggregate(df):
    by_regime = df.groupby("background_regime").agg(
        count=("background_regime", "count"),
        mean_vacuum_wave_density=("vacuum_wave_density", "mean"),
        mean_vacuum_coherence=("vacuum_coherence", "mean"),
        mean_background_fluctuation=("background_fluctuation", "mean"),
        mean_seed_potential=("seed_potential", "mean"),
        mean_background_order=("background_order_parameter", "mean"),
        mean_noise_floor=("vacuum_noise_floor", "mean"),
        mean_matter_seed_score=("matter_seed_score", "mean"),
        mean_background_stability=("background_stability", "mean"),
        support_ratio=("background_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        background_regime=("background_regime", lambda x: x.mode().iloc[0]),
        mean_vacuum_wave_density=("vacuum_wave_density", "mean"),
        mean_background_fluctuation=("background_fluctuation", "mean"),
        mean_matter_seed_score=("matter_seed_score", "mean"),
        mean_background_stability=("background_stability", "mean"),
    ).reset_index()

    by_architecture = df.groupby("matter_architecture").agg(
        count=("matter_architecture", "count"),
        background_regime=("background_regime", lambda x: x.mode().iloc[0]),
        mean_vacuum_wave_density=("vacuum_wave_density", "mean"),
        mean_seed_potential=("seed_potential", "mean"),
        mean_matter_seed_score=("matter_seed_score", "mean"),
        mean_background_stability=("background_stability", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_architecture


def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("background_regime"):
        plt.scatter(
            sub["background_fluctuation"],
            sub["matter_seed_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("background_fluctuation")
    plt.ylabel("matter_seed_score")
    plt.title("BST vacuum wave background to matter seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "vacuum_background_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("background_regime"):
        plt.scatter(
            sub["vacuum_coherence"],
            sub["background_stability"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("vacuum_coherence")
    plt.ylabel("background_stability")
    plt.title("BST vacuum coherence / stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "vacuum_background_stability.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(valid["cluster_class"].astype(str), valid["mean_matter_seed_score"])
    plt.xlabel("cluster_class")
    plt.ylabel("mean_matter_seed_score")
    plt.title("BST matter seed background by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "vacuum_background_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="background_regime",
        values="matter_seed_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="matter_seed_score")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("background_regime")
    plt.ylabel("cluster_class")
    plt.title("BST vacuum wave background heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "vacuum_background_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 68 VACUUM WAVE BACKGROUND TEST ===\n")

    base = load_input()
    samples = compute_background_features(base)

    summary = summarize(samples)
    by_regime, by_cluster, by_architecture = aggregate(samples)

    samples.to_csv(OUT / "vacuum_wave_background_samples.csv", index=False)
    summary.to_csv(OUT / "vacuum_wave_background_summary.csv", index=False)
    by_regime.to_csv(OUT / "vacuum_wave_background_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "vacuum_wave_background_by_cluster.csv", index=False)
    by_architecture.to_csv(OUT / "vacuum_wave_background_by_architecture.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nVacuum wave background by regime:")
    print(by_regime.to_string(index=False))

    print("\nVacuum wave background by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nVacuum wave background by matter architecture:")
    print(by_architecture.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'vacuum_wave_background_samples.csv'}")
    print(f"[OK] wrote {OUT / 'vacuum_wave_background_summary.csv'}")
    print(f"[OK] wrote {OUT / 'vacuum_wave_background_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'vacuum_wave_background_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'vacuum_wave_background_by_architecture.csv'}")
    print(f"[OK] wrote {OUT / 'vacuum_background_projection.png'}")
    print(f"[OK] wrote {OUT / 'vacuum_background_stability.png'}")
    print(f"[OK] wrote {OUT / 'vacuum_background_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'vacuum_background_heatmap.png'}")
    print("[DONE] vacuum wave background test complete")


if __name__ == "__main__":
    main()