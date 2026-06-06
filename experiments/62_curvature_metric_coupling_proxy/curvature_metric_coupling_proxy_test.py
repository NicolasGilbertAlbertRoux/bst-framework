#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 62 — Curvature / Metric Coupling Proxy

Purpose:
    Test whether BST effective fields and gauge-like invariants induce
    a metric / curvature proxy.

Canonical chain:
    W -> C -> field -> gauge-like invariants -> metric proxy -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no particle identity;
    - no claim of full General Relativity;
    - curvature means effective BST curvature proxy only.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


SEED = 1729
rng = np.random.default_rng(SEED)

FIELD_DIR = Path("results/research_final/field_emergence_test")
GAUGE_DIR = Path("results/research_final/gauge_like_symmetry_emergence_test")

FIELD_SAMPLES = FIELD_DIR / "field_emergence_samples.csv"
GAUGE_BASE = GAUGE_DIR / "gauge_like_base_invariants.csv"

OUT = Path("results/research_final/curvature_metric_coupling_proxy_test")
OUT.mkdir(parents=True, exist_ok=True)


FIELD_REQUIRED = [
    "field_regime",
    "cluster_class",
    "is_valid_field",
    "area",
    "compactness",
    "mean_density",
    "mean_overlap",
    "contact_capacity",
    "field_density",
    "field_coherence",
    "field_range",
    "field_gradient",
    "field_flux",
    "field_order",
    "global_field_stability",
    "local_field_score",
    "extended_field_score",
    "field_presence_score",
]

GAUGE_REQUIRED = [
    "gauge_amplitude",
    "gauge_charge_proxy",
    "field_action_proxy",
    "gauge_invariant_norm",
    "valid_gauge_candidate",
]


def read_required(path, cols):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}\n"
            "Run tests 60 and 61 first."
        )

    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    return df


def load_inputs():
    field = read_required(FIELD_SAMPLES, FIELD_REQUIRED)
    gauge = read_required(GAUGE_BASE, GAUGE_REQUIRED)

    if len(field) != len(gauge):
        raise ValueError(
            f"Field and gauge sample sizes differ: {len(field)} vs {len(gauge)}"
        )

    merged = field.copy()

    for col in GAUGE_REQUIRED:
        merged[col] = gauge[col].values

    return merged


def classify_metric_regime(row):
    regime = str(row["field_regime"])
    if regime == "invalid_control":
        return "flat_invalid"
    if regime == "local_field":
        return "local_metric_well"
    if regime == "extended_field":
        return "extended_curvature_basin"
    return "unclassified_metric"


def compute_metric_curvature(df):
    df = df.copy()

    density = df["field_density"].astype(float)
    coherence = df["field_coherence"].astype(float)
    field_range = df["field_range"].astype(float)
    gradient = df["field_gradient"].astype(float)
    flux = df["field_flux"].astype(float)
    action = df["field_action_proxy"].astype(float)
    gauge_norm = df["gauge_invariant_norm"].astype(float)
    charge = df["gauge_charge_proxy"].astype(float)
    presence = df["field_presence_score"].astype(float)
    compactness = df["compactness"].astype(float)
    area = df["area"].astype(float)

    df["metric_regime"] = df.apply(classify_metric_regime, axis=1)

    df["metric_potential"] = np.clip(
        0.28 * presence
        + 0.22 * gauge_norm
        + 0.18 * action
        + 0.14 * coherence
        + 0.10 * density
        + 0.08 * np.tanh(charge),
        0,
        3,
    )

    df["metric_scale_factor"] = np.clip(
        1.0 / (1.0 + df["metric_potential"]),
        0.05,
        1.0,
    )

    df["metric_anisotropy"] = np.clip(
        np.abs(df["local_field_score"] - df["extended_field_score"])
        * (0.5 + coherence),
        0,
        2,
    )

    df["curvature_density"] = np.clip(
        0.32 * gradient
        + 0.24 * density
        + 0.18 * flux
        + 0.14 * action
        + 0.12 * gauge_norm,
        0,
        4,
    )

    df["curvature_radius_proxy"] = np.clip(
        1.0 / (1e-6 + df["curvature_density"]),
        0,
        20,
    )

    df["metric_coupling_strength"] = np.clip(
        df["metric_potential"]
        * (0.4 + coherence)
        * (0.5 + np.tanh(area / 1000.0)),
        0,
        4,
    )

    df["curvature_coherence"] = np.clip(
        0.40 * coherence
        + 0.25 * presence
        + 0.20 * gauge_norm / (1.0 + gauge_norm)
        + 0.15 * compactness,
        0,
        1,
    )

    df["effective_curvature_score"] = np.clip(
        0.35 * df["metric_coupling_strength"]
        + 0.30 * df["curvature_density"]
        + 0.20 * df["curvature_coherence"]
        + 0.15 * df["metric_anisotropy"],
        0,
        4,
    )

    df["metric_supported"] = (
        (df["is_valid_field"] == 1)
        & (df["effective_curvature_score"] >= 0.45)
    ).astype(int)

    df["extended_metric_supported"] = (
        (df["metric_regime"] == "extended_curvature_basin")
        & (df["effective_curvature_score"] >= 0.85)
    ).astype(int)

    return df


def make_metric_controls(df):
    controls = df[df["field_regime"] == "invalid_control"].copy()
    controls = controls.copy()
    controls["metric_regime"] = "flat_invalid"
    return controls


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
        "metric_scale_factor",
        "metric_anisotropy",
        "curvature_density",
        "curvature_coherence",
        "metric_coupling_strength",
        "effective_curvature_score",
    ]

    if target not in df.columns:
        return np.nan

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


def safe_silhouette(df, labels, cols):
    try:
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(silhouette_score(df[cols], labels))
    except Exception:
        return np.nan


def summarize(df):
    invalid = df[df["metric_regime"] == "flat_invalid"]
    local = df[df["metric_regime"] == "local_metric_well"]
    extended = df[df["metric_regime"] == "extended_curvature_basin"]

    metric_accuracy = supervised_accuracy(df, "metric_supported")

    extended_subset = df[df["metric_regime"].isin(["extended_curvature_basin", "flat_invalid"])].copy()
    extended_metric_accuracy = supervised_accuracy(extended_subset, "extended_metric_supported")

    cols = [
        "metric_potential",
        "metric_anisotropy",
        "curvature_density",
        "curvature_coherence",
        "metric_coupling_strength",
        "effective_curvature_score",
    ]

    metric_silhouette = safe_silhouette(df, df["metric_regime"], cols)

    # Local metric wells are expected to have moderate curvature,
    # not extended-curvature values.

    local_support = float(
        (local["effective_curvature_score"] >= 0.30).mean()
    ) if len(local) else 0.0

    extended_support = float(
        (extended["effective_curvature_score"] >= 0.80).mean()
    ) if len(extended) else 0.0

    invalid_suppression = float(
        (invalid["effective_curvature_score"] < 0.30).mean()
    ) if len(invalid) else 0.0

    mean_local = float(local["effective_curvature_score"].mean()) if len(local) else 0.0
    mean_extended = float(extended["effective_curvature_score"].mean()) if len(extended) else 0.0
    mean_invalid = float(invalid["effective_curvature_score"].mean()) if len(invalid) else 0.0

    local_contrast = mean_local / max(mean_invalid, 1e-9)
    extended_contrast = mean_extended / max(mean_invalid, 1e-9)

    density_curvature_corr = float(
        np.corrcoef(df["field_density"], df["effective_curvature_score"])[0, 1]
    )

    gauge_metric_corr = float(
        np.corrcoef(df["gauge_invariant_norm"], df["metric_potential"])[0, 1]
    )

    coherence_curvature_corr = float(
        np.corrcoef(df["field_coherence"], df["curvature_coherence"])[0, 1]
    )

    verdict = (
        "curvature_metric_coupling_supported"
        if (
            metric_accuracy >= 0.98
            and local_support >= 0.95
            and extended_support >= 0.85
            and invalid_suppression >= 0.95
            and local_contrast > 2.0
            and extended_contrast > 5.0
            and gauge_metric_corr > 0.90
            and coherence_curvature_corr > 0.90
        )
        else "curvature_metric_coupling_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_local_metric": len(local),
        "num_extended_metric": len(extended),
        "num_invalid_controls": len(invalid),
        "metric_accuracy": metric_accuracy,
        "extended_metric_accuracy": extended_metric_accuracy,
        "metric_silhouette": metric_silhouette,
        "local_metric_support": local_support,
        "extended_metric_support": extended_support,
        "invalid_suppression": invalid_suppression,
        "mean_local_curvature_score": mean_local,
        "mean_extended_curvature_score": mean_extended,
        "mean_invalid_curvature_score": mean_invalid,
        "local_curvature_contrast": local_contrast,
        "extended_curvature_contrast": extended_contrast,
        "density_curvature_corr": density_curvature_corr,
        "gauge_metric_corr": gauge_metric_corr,
        "coherence_curvature_corr": coherence_curvature_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "general_relativity_claim": False,
        "safe_hierarchy": "W -> C -> field -> gauge-like invariants -> metric proxy -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(df):
    by_regime = df.groupby("metric_regime").agg(
        count=("metric_regime", "count"),
        mean_metric_potential=("metric_potential", "mean"),
        mean_metric_scale_factor=("metric_scale_factor", "mean"),
        mean_metric_anisotropy=("metric_anisotropy", "mean"),
        mean_curvature_density=("curvature_density", "mean"),
        mean_curvature_coherence=("curvature_coherence", "mean"),
        mean_metric_coupling_strength=("metric_coupling_strength", "mean"),
        mean_effective_curvature_score=("effective_curvature_score", "mean"),
        support_ratio=("metric_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        metric_regime=("metric_regime", lambda x: x.mode().iloc[0]),
        mean_metric_potential=("metric_potential", "mean"),
        mean_metric_scale_factor=("metric_scale_factor", "mean"),
        mean_curvature_density=("curvature_density", "mean"),
        mean_curvature_coherence=("curvature_coherence", "mean"),
        mean_effective_curvature_score=("effective_curvature_score", "mean"),
    ).reset_index()

    by_field_regime = df.groupby("field_regime").agg(
        count=("field_regime", "count"),
        mean_metric_potential=("metric_potential", "mean"),
        mean_curvature_density=("curvature_density", "mean"),
        mean_effective_curvature_score=("effective_curvature_score", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_field_regime


def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("metric_regime"):
        plt.scatter(
            sub["metric_potential"],
            sub["effective_curvature_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("metric_potential")
    plt.ylabel("effective_curvature_score")
    plt.title("BST metric potential to curvature proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "metric_curvature_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("metric_regime"):
        plt.scatter(
            sub["gauge_invariant_norm"],
            sub["metric_potential"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("gauge_invariant_norm")
    plt.ylabel("metric_potential")
    plt.title("BST gauge invariant to metric potential")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "gauge_metric_coupling.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(
        valid["cluster_class"].astype(str),
        valid["mean_effective_curvature_score"],
    )
    plt.xlabel("cluster_class")
    plt.ylabel("mean_effective_curvature_score")
    plt.title("BST curvature proxy by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "curvature_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="field_regime",
        values="effective_curvature_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="effective_curvature_score")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("field_regime")
    plt.ylabel("cluster_class")
    plt.title("BST curvature proxy heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "curvature_metric_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 62 CURVATURE / METRIC COUPLING PROXY TEST ===\n")

    base = load_inputs()
    samples = compute_metric_curvature(base)

    summary = summarize(samples)
    by_regime, by_cluster, by_field_regime = aggregate(samples)

    samples.to_csv(OUT / "curvature_metric_samples.csv", index=False)
    summary.to_csv(OUT / "curvature_metric_summary.csv", index=False)
    by_regime.to_csv(OUT / "curvature_metric_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "curvature_metric_by_cluster.csv", index=False)
    by_field_regime.to_csv(OUT / "curvature_metric_by_field_regime.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nCurvature / metric by regime:")
    print(by_regime.to_string(index=False))

    print("\nCurvature / metric by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nCurvature / metric by field regime:")
    print(by_field_regime.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'curvature_metric_samples.csv'}")
    print(f"[OK] wrote {OUT / 'curvature_metric_summary.csv'}")
    print(f"[OK] wrote {OUT / 'curvature_metric_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'curvature_metric_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'curvature_metric_by_field_regime.csv'}")
    print(f"[OK] wrote {OUT / 'metric_curvature_projection.png'}")
    print(f"[OK] wrote {OUT / 'gauge_metric_coupling.png'}")
    print(f"[OK] wrote {OUT / 'curvature_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'curvature_metric_heatmap.png'}")
    print("[DONE] curvature / metric coupling proxy test complete")


if __name__ == "__main__":
    main()