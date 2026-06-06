#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 61 — Gauge-Like Symmetry Emergence

Purpose:
    Test whether effective fields from BST wave-clusters exhibit gauge-like
    internal invariants.

Input:
    results/research_final/field_emergence_test/field_emergence_samples.csv

Canonical hierarchy:
    W -> C -> local/extended field -> gauge-like invariants -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no particle identity;
    - no Standard Model gauge claim;
    - "gauge-like" means internal transformation preserving effective field invariants.
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

SRC = Path("results/research_final/field_emergence_test")
SAMPLES = SRC / "field_emergence_samples.csv"

OUT = Path("results/research_final/gauge_like_symmetry_emergence_test")
OUT.mkdir(parents=True, exist_ok=True)


REQUIRED_COLUMNS = [
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


def load_samples():
    if not SAMPLES.exists():
        raise FileNotFoundError(
            f"Missing required input: {SAMPLES}\n"
            "Run experiments/60_field_emergence/field_emergence_test.py first."
        )

    df = pd.read_csv(SAMPLES)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {SAMPLES}: {missing}")

    return df


def base_invariants(df):
    df = df.copy()

    local = df["local_field_score"].astype(float)
    extended = df["extended_field_score"].astype(float)
    density = df["field_density"].astype(float)
    coherence = df["field_coherence"].astype(float)
    field_range = df["field_range"].astype(float)
    flux = df["field_flux"].astype(float)
    order = df["field_order"].astype(float)
    capacity = df["contact_capacity"].astype(float)

    df["gauge_amplitude"] = np.sqrt(local ** 2 + extended ** 2)
    df["gauge_phase"] = np.arctan2(extended, local)
    df["gauge_charge_proxy"] = np.clip(
        df["gauge_amplitude"] * (0.5 + np.tanh(capacity / 4.0)),
        0,
        2,
    )

    df["field_action_proxy"] = np.clip(
        density ** 2
        + 0.7 * coherence ** 2
        + 0.5 * field_range ** 2
        + 0.3 * flux ** 2
        + 0.2 * order ** 2,
        0,
        4,
    )

    df["gauge_invariant_norm"] = np.sqrt(
        df["gauge_amplitude"] ** 2
        + 0.35 * df["field_action_proxy"]
        + 0.15 * df["gauge_charge_proxy"] ** 2
    )

    df["valid_gauge_candidate"] = (
        (df["is_valid_field"] == 1)
        & (df["field_presence_score"] >= 0.32)
    ).astype(int)

    return df


def make_gauge_transformed(df):
    """
    Internal rotation of local/extended field components.
    It preserves amplitude-like invariants.
    """
    g = df.copy()

    theta = rng.uniform(-np.pi, np.pi, len(g))
    local = g["local_field_score"].values
    extended = g["extended_field_score"].values

    new_local = local * np.cos(theta) - extended * np.sin(theta)
    new_extended = local * np.sin(theta) + extended * np.cos(theta)

    g["transform_type"] = "gauge_like_rotation"
    g["gauge_angle"] = theta
    g["local_transformed"] = new_local
    g["extended_transformed"] = new_extended

    g["transformed_amplitude"] = np.sqrt(new_local ** 2 + new_extended ** 2)
    g["transformed_phase"] = np.arctan2(new_extended, new_local)

    g["amplitude_error"] = np.abs(g["transformed_amplitude"] - g["gauge_amplitude"])

    g["charge_error"] = np.abs(
        g["gauge_charge_proxy"]
        - np.clip(
            g["transformed_amplitude"] * (0.5 + np.tanh(g["contact_capacity"] / 4.0)),
            0,
            2,
        )
    )

    g["action_error"] = 0.0
    g["invariant_error"] = (
        0.55 * g["amplitude_error"]
        + 0.30 * g["charge_error"]
        + 0.15 * g["action_error"]
    )

    g["expected_invariant"] = 1
    return g


def make_broken_transformed(df):
    """
    Non-gauge perturbation that breaks amplitude-like invariants.
    """
    b = df.copy()

    scale_local = rng.uniform(0.15, 1.85, len(b))
    scale_extended = rng.uniform(0.15, 1.85, len(b))
    additive = rng.normal(0.0, 0.08, len(b))

    local = b["local_field_score"].values
    extended = b["extended_field_score"].values

    new_local = local * scale_local + additive
    new_extended = extended * scale_extended - additive

    b["transform_type"] = "broken_internal_transform"
    b["gauge_angle"] = np.nan
    b["local_transformed"] = new_local
    b["extended_transformed"] = new_extended

    b["transformed_amplitude"] = np.sqrt(new_local ** 2 + new_extended ** 2)
    b["transformed_phase"] = np.arctan2(new_extended, new_local)

    b["amplitude_error"] = np.abs(b["transformed_amplitude"] - b["gauge_amplitude"])

    b["charge_error"] = np.abs(
        b["gauge_charge_proxy"]
        - np.clip(
            b["transformed_amplitude"] * (0.5 + np.tanh(b["contact_capacity"] / 4.0)),
            0,
            2,
        )
    )

    b["action_error"] = np.clip(
        np.abs(scale_local - 1.0) * b["field_density"]
        + np.abs(scale_extended - 1.0) * b["field_range"],
        0,
        2,
    )

    b["invariant_error"] = (
        0.55 * b["amplitude_error"]
        + 0.30 * b["charge_error"]
        + 0.15 * b["action_error"]
    )

    b["expected_invariant"] = 0
    return b


def classify_invariance(transforms):
    features = [
        "gauge_amplitude",
        "gauge_charge_proxy",
        "field_action_proxy",
        "gauge_invariant_norm",
        "transformed_amplitude",
        "amplitude_error",
        "charge_error",
        "action_error",
        "invariant_error",
        "field_presence_score",
        "field_coherence",
        "field_density",
        "field_range",
    ]

    X = transforms[features].values
    y = transforms["expected_invariant"].values

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


def summarize(base, transforms):
    gauge = transforms[transforms["transform_type"] == "gauge_like_rotation"]
    broken = transforms[transforms["transform_type"] == "broken_internal_transform"]

    valid_gauge = gauge[gauge["valid_gauge_candidate"] == 1]
    invalid_gauge = gauge[gauge["valid_gauge_candidate"] == 0]

    invariant_threshold = 1e-9
    relaxed_threshold = 1e-6
    broken_threshold = 0.03

    invariance_accuracy = classify_invariance(transforms)

    gauge_invariance_support = float((gauge["invariant_error"] <= relaxed_threshold).mean())
    valid_gauge_support = float((valid_gauge["invariant_error"] <= relaxed_threshold).mean()) if len(valid_gauge) else 0.0

    broken_rejection = float((broken["invariant_error"] > broken_threshold).mean())
    invalid_suppression = float((invalid_gauge["field_presence_score"] < 0.32).mean()) if len(invalid_gauge) else 0.0

    mean_gauge_error = float(gauge["invariant_error"].mean())
    mean_broken_error = float(broken["invariant_error"].mean())
    error_contrast = mean_broken_error / max(mean_gauge_error, 1e-12)

    phase_dispersion = float(np.std(gauge["transformed_phase"]))
    amplitude_conservation = float(1.0 - np.clip(gauge["amplitude_error"].mean(), 0, 1))

    inv_cols = [
        "gauge_amplitude",
        "gauge_charge_proxy",
        "field_action_proxy",
        "gauge_invariant_norm",
    ]

    regime_silhouette = safe_silhouette(base, base["field_regime"], inv_cols)

    invariant_transform_silhouette = safe_silhouette(
        transforms,
        transforms["expected_invariant"],
        [
            "amplitude_error",
            "charge_error",
            "action_error",
            "invariant_error",
            "transformed_amplitude",
        ],
    )

    verdict = (
        "gauge_like_symmetry_emergence_supported"
        if (
            invariance_accuracy >= 0.99
            and gauge_invariance_support >= 0.999
            and valid_gauge_support >= 0.999
            and broken_rejection >= 0.80
            and invalid_suppression >= 0.95
            and amplitude_conservation >= 0.999
            and error_contrast > 1e6
        )
        else "gauge_like_symmetry_emergence_not_supported"
    )

    return pd.DataFrame([{
        "num_base_samples": len(base),
        "num_transform_samples": len(transforms),
        "num_valid_gauge_candidates": int(base["valid_gauge_candidate"].sum()),
        "invariance_accuracy": invariance_accuracy,
        "gauge_invariance_support": gauge_invariance_support,
        "valid_gauge_support": valid_gauge_support,
        "broken_rejection": broken_rejection,
        "invalid_suppression": invalid_suppression,
        "mean_gauge_invariant_error": mean_gauge_error,
        "mean_broken_invariant_error": mean_broken_error,
        "invariant_error_contrast": error_contrast,
        "amplitude_conservation": amplitude_conservation,
        "phase_dispersion": phase_dispersion,
        "regime_invariant_silhouette": regime_silhouette,
        "invariant_transform_silhouette": invariant_transform_silhouette,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "standard_model_gauge_claim": False,
        "safe_hierarchy": "W -> C -> field -> gauge-like invariants -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(base, transforms):
    by_regime = base.groupby("field_regime").agg(
        count=("field_regime", "count"),
        mean_gauge_amplitude=("gauge_amplitude", "mean"),
        mean_gauge_charge_proxy=("gauge_charge_proxy", "mean"),
        mean_field_action_proxy=("field_action_proxy", "mean"),
        mean_gauge_invariant_norm=("gauge_invariant_norm", "mean"),
        mean_field_presence_score=("field_presence_score", "mean"),
    ).reset_index()

    by_cluster = base.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        regime=("field_regime", lambda x: x.mode().iloc[0]),
        mean_gauge_amplitude=("gauge_amplitude", "mean"),
        mean_gauge_charge_proxy=("gauge_charge_proxy", "mean"),
        mean_field_action_proxy=("field_action_proxy", "mean"),
        mean_gauge_invariant_norm=("gauge_invariant_norm", "mean"),
    ).reset_index()

    by_transform = transforms.groupby("transform_type").agg(
        count=("transform_type", "count"),
        mean_amplitude_error=("amplitude_error", "mean"),
        mean_charge_error=("charge_error", "mean"),
        mean_action_error=("action_error", "mean"),
        mean_invariant_error=("invariant_error", "mean"),
        invariant_support=("invariant_error", lambda x: float((x <= 1e-6).mean())),
    ).reset_index()

    return by_regime, by_cluster, by_transform


def make_plots(base, transforms, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in base.groupby("field_regime"):
        plt.scatter(
            sub["gauge_amplitude"],
            sub["gauge_charge_proxy"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("gauge_amplitude")
    plt.ylabel("gauge_charge_proxy")
    plt.title("BST gauge-like invariant landscape")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "gauge_invariant_landscape.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in transforms.groupby("transform_type"):
        plt.hist(
            sub["invariant_error"],
            bins=60,
            alpha=0.55,
            label=name,
        )
    plt.xlabel("invariant_error")
    plt.ylabel("count")
    plt.title("Gauge-like invariance vs broken transforms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "gauge_invariant_error_histogram.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(
        valid["cluster_class"].astype(str),
        valid["mean_gauge_invariant_norm"],
    )
    plt.xlabel("cluster_class")
    plt.ylabel("mean_gauge_invariant_norm")
    plt.title("Gauge-like invariant norm by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "gauge_invariant_by_cluster.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    gauge = transforms[transforms["transform_type"] == "gauge_like_rotation"]
    plt.scatter(
        gauge["gauge_angle"],
        gauge["transformed_phase"],
        s=6,
        alpha=0.35,
    )
    plt.xlabel("internal gauge angle")
    plt.ylabel("transformed phase")
    plt.title("Gauge-like internal phase rotation")
    plt.tight_layout()
    plt.savefig(OUT / "gauge_phase_rotation.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 61 GAUGE-LIKE SYMMETRY EMERGENCE TEST ===\n")

    raw = load_samples()
    base = base_invariants(raw)

    gauge = make_gauge_transformed(base)
    broken = make_broken_transformed(base)

    transforms = pd.concat([gauge, broken], ignore_index=True)

    summary = summarize(base, transforms)
    by_regime, by_cluster, by_transform = aggregate(base, transforms)

    base.to_csv(OUT / "gauge_like_base_invariants.csv", index=False)
    transforms.to_csv(OUT / "gauge_like_transform_samples.csv", index=False)
    summary.to_csv(OUT / "gauge_like_symmetry_summary.csv", index=False)
    by_regime.to_csv(OUT / "gauge_like_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "gauge_like_by_cluster.csv", index=False)
    by_transform.to_csv(OUT / "gauge_like_by_transform.csv", index=False)

    make_plots(base, transforms, by_cluster)

    print(summary.to_string(index=False))

    print("\nGauge-like invariants by regime:")
    print(by_regime.to_string(index=False))

    print("\nGauge-like invariants by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nGauge-like transform checks:")
    print(by_transform.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'gauge_like_base_invariants.csv'}")
    print(f"[OK] wrote {OUT / 'gauge_like_transform_samples.csv'}")
    print(f"[OK] wrote {OUT / 'gauge_like_symmetry_summary.csv'}")
    print(f"[OK] wrote {OUT / 'gauge_like_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'gauge_like_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'gauge_like_by_transform.csv'}")
    print(f"[OK] wrote {OUT / 'gauge_invariant_landscape.png'}")
    print(f"[OK] wrote {OUT / 'gauge_invariant_error_histogram.png'}")
    print(f"[OK] wrote {OUT / 'gauge_invariant_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'gauge_phase_rotation.png'}")
    print("[DONE] gauge-like symmetry emergence test complete")


if __name__ == "__main__":
    main()