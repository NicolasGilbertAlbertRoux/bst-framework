#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 63 — Wave Stress–Energy Tensor Proxy

Input:
    results/research_final/curvature_metric_coupling_proxy_test/curvature_metric_samples.csv

Purpose:
    Build an effective BST stress-energy proxy from the already validated chain:

    W -> C -> field -> gauge-like invariants -> metric proxy
       -> stress-energy proxy -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no particle identity;
    - no full General Relativity claim;
    - stress-energy means BST effective proxy only.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


SEED = 1729

INP = Path("results/research_final/curvature_metric_coupling_proxy_test/curvature_metric_samples.csv")
OUT = Path("results/research_final/wave_stress_energy_tensor_proxy_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "metric_regime",
    "field_regime",
    "cluster_class",
    "is_valid_field",
    "field_density",
    "field_coherence",
    "field_range",
    "field_gradient",
    "field_flux",
    "field_order",
    "field_presence_score",
    "gauge_amplitude",
    "gauge_charge_proxy",
    "field_action_proxy",
    "gauge_invariant_norm",
    "metric_potential",
    "metric_scale_factor",
    "metric_anisotropy",
    "curvature_density",
    "curvature_coherence",
    "metric_coupling_strength",
    "effective_curvature_score",
]


def load_input():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing required input: {INP}\n"
            "Run experiments/62_curvature_metric_coupling_proxy/curvature_metric_coupling_proxy_test.py first."
        )

    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")

    return df


def classify_stress_regime(metric_regime):
    if metric_regime == "flat_invalid":
        return "vacuum_like"
    if metric_regime == "local_metric_well":
        return "localized_stress_energy"
    if metric_regime == "extended_curvature_basin":
        return "curvature_stress_energy_basin"
    return "unclassified_stress_energy"


def compute_tensor_proxy(df):
    df = df.copy()

    fd = df["field_density"].astype(float)
    fc = df["field_coherence"].astype(float)
    fr = df["field_range"].astype(float)
    fg = df["field_gradient"].astype(float)
    ff = df["field_flux"].astype(float)
    fo = df["field_order"].astype(float)
    fp = df["field_presence_score"].astype(float)

    ga = df["gauge_amplitude"].astype(float)
    gc = df["gauge_charge_proxy"].astype(float)
    fa = df["field_action_proxy"].astype(float)
    gn = df["gauge_invariant_norm"].astype(float)

    mp = df["metric_potential"].astype(float)
    ms = df["metric_scale_factor"].astype(float)
    ma = df["metric_anisotropy"].astype(float)
    cd = df["curvature_density"].astype(float)
    cc = df["curvature_coherence"].astype(float)
    mc = df["metric_coupling_strength"].astype(float)
    ecs = df["effective_curvature_score"].astype(float)

    df["stress_regime"] = df["metric_regime"].apply(classify_stress_regime)

    df["energy_density_proxy"] = np.clip(
        0.22 * fd
        + 0.20 * mp
        + 0.18 * cd
        + 0.14 * fc
        + 0.12 * gn
        + 0.08 * fp
        + 0.06 * fa,
        0,
        4,
    )

    df["pressure_proxy"] = np.clip(
        0.28 * fg
        + 0.22 * cd
        + 0.18 * mc
        + 0.14 * ma
        + 0.10 * fo
        + 0.08 * cc,
        0,
        4,
    )

    df["flux_proxy"] = np.clip(
        0.34 * ff
        + 0.22 * fr
        + 0.18 * fc
        + 0.14 * mp
        + 0.12 * ga,
        0,
        4,
    )

    df["anisotropy_proxy"] = np.clip(
        0.45 * ma
        + 0.30 * np.abs(df["pressure_proxy"] - df["flux_proxy"])
        + 0.25 * np.abs(fd - fr),
        0,
        4,
    )

    df["tensor_trace_proxy"] = np.clip(
        df["energy_density_proxy"] - 3.0 * df["pressure_proxy"],
        -4,
        4,
    )

    df["tensor_norm_proxy"] = np.clip(
        np.sqrt(
            df["energy_density_proxy"] ** 2
            + 3.0 * df["pressure_proxy"] ** 2
            + df["flux_proxy"] ** 2
            + df["anisotropy_proxy"] ** 2
        ),
        0,
        6,
    )

    df["stress_energy_coupling"] = np.clip(
        0.35 * df["tensor_norm_proxy"]
        + 0.25 * df["energy_density_proxy"]
        + 0.20 * ecs
        + 0.12 * mc
        + 0.08 * gn,
        0,
        6,
    )

    df["conservation_proxy"] = np.clip(
        1.0
        - (
            0.30 * np.abs(df["energy_density_proxy"] - mp)
            + 0.25 * np.abs(df["flux_proxy"] - ff)
            + 0.20 * np.abs(df["pressure_proxy"] - cd)
            + 0.15 * np.abs(df["anisotropy_proxy"] - ma)
            + 0.10 * np.abs(df["tensor_norm_proxy"] - df["stress_energy_coupling"])
        ),
        0,
        1,
    )

    df["local_tensor_supported"] = (
        (df["stress_regime"] == "localized_stress_energy")
        & (df["stress_energy_coupling"] >= 0.45)
    ).astype(int)

    df["extended_tensor_supported"] = (
        (df["stress_regime"] == "curvature_stress_energy_basin")
        & (df["stress_energy_coupling"] >= 1.25)
    ).astype(int)

    df["tensor_supported"] = (
        (df["is_valid_field"] == 1)
        & (
            ((df["stress_regime"] == "localized_stress_energy") & (df["stress_energy_coupling"] >= 0.45))
            | ((df["stress_regime"] == "curvature_stress_energy_basin") & (df["stress_energy_coupling"] >= 1.25))
        )
    ).astype(int)

    return df


def supervised_accuracy(df, target):
    features = [
        "field_density",
        "field_coherence",
        "field_range",
        "field_gradient",
        "field_flux",
        "field_order",
        "field_presence_score",
        "gauge_amplitude",
        "gauge_charge_proxy",
        "field_action_proxy",
        "gauge_invariant_norm",
        "metric_potential",
        "metric_anisotropy",
        "curvature_density",
        "curvature_coherence",
        "metric_coupling_strength",
        "effective_curvature_score",
        "energy_density_proxy",
        "pressure_proxy",
        "flux_proxy",
        "anisotropy_proxy",
        "tensor_norm_proxy",
        "stress_energy_coupling",
        "conservation_proxy",
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
        n_estimators=260,
        max_depth=8,
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
    invalid = df[df["stress_regime"] == "vacuum_like"]
    local = df[df["stress_regime"] == "localized_stress_energy"]
    extended = df[df["stress_regime"] == "curvature_stress_energy_basin"]

    tensor_accuracy = supervised_accuracy(df, "tensor_supported")

    local_subset = df[df["stress_regime"].isin(["localized_stress_energy", "vacuum_like"])]
    extended_subset = df[df["stress_regime"].isin(["curvature_stress_energy_basin", "vacuum_like"])]

    local_tensor_accuracy = supervised_accuracy(local_subset, "local_tensor_supported")
    extended_tensor_accuracy = supervised_accuracy(extended_subset, "extended_tensor_supported")

    local_support = float((local["stress_energy_coupling"] >= 0.45).mean()) if len(local) else 0.0
    extended_support = float((extended["stress_energy_coupling"] >= 1.25).mean()) if len(extended) else 0.0
    invalid_suppression = float((invalid["stress_energy_coupling"] < 0.45).mean()) if len(invalid) else 0.0

    mean_local = float(local["stress_energy_coupling"].mean()) if len(local) else 0.0
    mean_extended = float(extended["stress_energy_coupling"].mean()) if len(extended) else 0.0
    mean_invalid = float(invalid["stress_energy_coupling"].mean()) if len(invalid) else 0.0

    local_contrast = mean_local / max(mean_invalid, 1e-9)
    extended_contrast = mean_extended / max(mean_invalid, 1e-9)

    tensor_silhouette = safe_silhouette(
        df,
        "stress_regime",
        [
            "energy_density_proxy",
            "pressure_proxy",
            "flux_proxy",
            "anisotropy_proxy",
            "tensor_norm_proxy",
            "stress_energy_coupling",
        ],
    )

    energy_curvature_corr = safe_corr(df["energy_density_proxy"], df["effective_curvature_score"])
    metric_energy_corr = safe_corr(df["metric_potential"], df["energy_density_proxy"])
    gauge_energy_corr = safe_corr(df["gauge_invariant_norm"], df["energy_density_proxy"])
    flux_curvature_corr = safe_corr(df["flux_proxy"], df["curvature_density"])
    conservation_energy_corr = safe_corr(df["conservation_proxy"], df["stress_energy_coupling"])

    verdict = (
        "wave_stress_energy_tensor_supported"
        if (
            tensor_accuracy >= 0.98
            and local_support >= 0.95
            and extended_support >= 0.90
            and invalid_suppression >= 0.95
            and local_contrast > 2.0
            and extended_contrast > 5.0
            and energy_curvature_corr > 0.80
            and metric_energy_corr > 0.80
            and gauge_energy_corr > 0.80
        )
        else "wave_stress_energy_tensor_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_vacuum_like": len(invalid),
        "num_localized_stress_energy": len(local),
        "num_curvature_stress_energy_basin": len(extended),
        "tensor_accuracy": tensor_accuracy,
        "local_tensor_accuracy": local_tensor_accuracy,
        "extended_tensor_accuracy": extended_tensor_accuracy,
        "tensor_silhouette": tensor_silhouette,
        "local_tensor_support": local_support,
        "extended_tensor_support": extended_support,
        "invalid_suppression": invalid_suppression,
        "mean_local_stress_energy": mean_local,
        "mean_extended_stress_energy": mean_extended,
        "mean_invalid_stress_energy": mean_invalid,
        "local_stress_energy_contrast": local_contrast,
        "extended_stress_energy_contrast": extended_contrast,
        "energy_curvature_corr": energy_curvature_corr,
        "metric_energy_corr": metric_energy_corr,
        "gauge_energy_corr": gauge_energy_corr,
        "flux_curvature_corr": flux_curvature_corr,
        "conservation_energy_corr": conservation_energy_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "general_relativity_claim": False,
        "safe_hierarchy": "W -> C -> field -> gauge-like invariants -> metric proxy -> stress-energy proxy -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(df):
    by_regime = df.groupby("stress_regime").agg(
        count=("stress_regime", "count"),
        mean_energy_density=("energy_density_proxy", "mean"),
        mean_pressure=("pressure_proxy", "mean"),
        mean_flux=("flux_proxy", "mean"),
        mean_anisotropy=("anisotropy_proxy", "mean"),
        mean_tensor_norm=("tensor_norm_proxy", "mean"),
        mean_stress_energy_coupling=("stress_energy_coupling", "mean"),
        mean_conservation_proxy=("conservation_proxy", "mean"),
        support_ratio=("tensor_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        stress_regime=("stress_regime", lambda x: x.mode().iloc[0]),
        mean_energy_density=("energy_density_proxy", "mean"),
        mean_pressure=("pressure_proxy", "mean"),
        mean_flux=("flux_proxy", "mean"),
        mean_anisotropy=("anisotropy_proxy", "mean"),
        mean_tensor_norm=("tensor_norm_proxy", "mean"),
        mean_stress_energy_coupling=("stress_energy_coupling", "mean"),
    ).reset_index()

    by_metric = df.groupby("metric_regime").agg(
        count=("metric_regime", "count"),
        mean_energy_density=("energy_density_proxy", "mean"),
        mean_pressure=("pressure_proxy", "mean"),
        mean_flux=("flux_proxy", "mean"),
        mean_stress_energy_coupling=("stress_energy_coupling", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_metric


def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("stress_regime"):
        plt.scatter(
            sub["energy_density_proxy"],
            sub["stress_energy_coupling"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("energy_density_proxy")
    plt.ylabel("stress_energy_coupling")
    plt.title("BST stress-energy coupling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "stress_energy_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("stress_regime"):
        plt.scatter(
            sub["effective_curvature_score"],
            sub["energy_density_proxy"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("effective_curvature_score")
    plt.ylabel("energy_density_proxy")
    plt.title("BST curvature to energy density proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "energy_curvature_coupling.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(
        valid["cluster_class"].astype(str),
        valid["mean_stress_energy_coupling"],
    )
    plt.xlabel("cluster_class")
    plt.ylabel("mean_stress_energy_coupling")
    plt.title("BST stress-energy by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "stress_energy_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="stress_regime",
        values="stress_energy_coupling",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="stress_energy_coupling")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("stress_regime")
    plt.ylabel("cluster_class")
    plt.title("BST stress-energy tensor proxy heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "stress_energy_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 63 WAVE STRESS–ENERGY TENSOR PROXY TEST ===\n")

    base = load_input()
    samples = compute_tensor_proxy(base)

    summary = summarize(samples)
    by_regime, by_cluster, by_metric = aggregate(samples)

    samples.to_csv(OUT / "wave_stress_energy_samples.csv", index=False)
    summary.to_csv(OUT / "wave_stress_energy_summary.csv", index=False)
    by_regime.to_csv(OUT / "wave_stress_energy_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "wave_stress_energy_by_cluster.csv", index=False)
    by_metric.to_csv(OUT / "wave_stress_energy_by_metric_regime.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nStress-energy by regime:")
    print(by_regime.to_string(index=False))

    print("\nStress-energy by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nStress-energy by metric regime:")
    print(by_metric.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'wave_stress_energy_samples.csv'}")
    print(f"[OK] wrote {OUT / 'wave_stress_energy_summary.csv'}")
    print(f"[OK] wrote {OUT / 'wave_stress_energy_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'wave_stress_energy_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'wave_stress_energy_by_metric_regime.csv'}")
    print(f"[OK] wrote {OUT / 'stress_energy_projection.png'}")
    print(f"[OK] wrote {OUT / 'energy_curvature_coupling.png'}")
    print(f"[OK] wrote {OUT / 'stress_energy_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'stress_energy_heatmap.png'}")
    print("[DONE] wave stress-energy tensor proxy test complete")


if __name__ == "__main__":
    main()