#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 60 — Field Emergence

Canonical autonomous version.

Purpose:
    Test whether archived BST wave-cluster classes generate effective fields.

Core conclusion tested:
    C0     -> local field
    C1/C2  -> extended field
    invalid controls -> suppressed

Safe hierarchy:
    W -> C -> local/extended field -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no particle identity assumption;
    - no electron/photon/quark/neutrino identity;
    - field means effective collective influence, not EM/gravity yet.
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

OUT = Path("results/research_final/field_emergence_test")
OUT.mkdir(parents=True, exist_ok=True)

SRC = Path("results/research_final/wave_cluster_taxonomy_test")
EVENTS = SRC / "wave_cluster_taxonomy_events.csv"
WAVE_SUMMARY = SRC / "wave_cluster_taxonomy_summary.csv"
WAVE_BY_PERT = SRC / "wave_cluster_taxonomy_by_perturbation.csv"


REQUIRED_COLUMNS = [
    "perturbation",
    "area",
    "compactness",
    "mean_density",
    "max_density",
    "mean_overlap",
    "contact_capacity",
    "centroid_x",
    "centroid_y",
    "cluster_class",
]


def load_wave_cluster_events():
    if not EVENTS.exists():
        raise FileNotFoundError(
            f"Missing required input: {EVENTS}\n"
            "Run the wave-cluster taxonomy test first."
        )

    events = pd.read_csv(EVENTS)
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"Missing columns in {EVENTS}: {missing}")

    return events


def load_optional(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def make_invalid_noise(events, n):
    base = events.sample(n=n, replace=True, random_state=SEED).reset_index(drop=True).copy()

    base["area"] = rng.uniform(4.0, 40.0, n)
    base["compactness"] = rng.uniform(0.01, 0.12, n)
    base["mean_density"] = rng.uniform(0.02, 0.35, n)
    base["max_density"] = base["mean_density"] + rng.uniform(0.00, 0.08, n)
    base["mean_overlap"] = rng.uniform(0.0, 0.55, n)
    base["contact_capacity"] = rng.uniform(0.0, 0.18, n)
    base["centroid_x"] = rng.uniform(0.0, 1.0, n)
    base["centroid_y"] = rng.uniform(0.0, 1.0, n)
    base["cluster_class"] = -1
    base["perturbation"] = rng.choice(events["perturbation"].unique(), n)

    return base


def make_pseudo_invalid_clusters(events, n):
    base = events.sample(n=n, replace=True, random_state=SEED + 1).reset_index(drop=True).copy()

    base["area"] = rng.uniform(60.0, 1800.0, n)
    base["compactness"] = rng.uniform(0.005, 0.06, n)
    base["mean_density"] = rng.uniform(0.30, 0.95, n)
    base["max_density"] = base["mean_density"] + rng.uniform(0.00, 0.20, n)
    base["mean_overlap"] = rng.uniform(0.2, 1.0, n)
    base["contact_capacity"] = rng.uniform(0.05, 0.60, n)
    base["centroid_x"] = rng.uniform(0.0, 1.0, n)
    base["centroid_y"] = rng.uniform(0.0, 1.0, n)
    base["cluster_class"] = -2
    base["perturbation"] = rng.choice(events["perturbation"].unique(), n)

    return base


def classify_field_regime(cluster_class):
    c = int(cluster_class)

    if c < 0:
        return "invalid_control"

    if c == 0:
        return "local_field"

    if c in (1, 2):
        return "extended_field"

    return "unclassified_valid"


def compute_field_features(df, source):
    rows = []

    for _, r in df.iterrows():
        area = float(r["area"])
        compactness = float(r["compactness"])
        mean_density = float(r["mean_density"])
        max_density = float(r["max_density"])
        mean_overlap = float(r["mean_overlap"])
        contact_capacity = float(r["contact_capacity"])
        centroid_x = float(r["centroid_x"])
        centroid_y = float(r["centroid_y"])
        perturbation = float(r["perturbation"])
        cluster_class = int(r["cluster_class"])

        spatial_centering = 1.0 - min(
            1.0,
            np.sqrt((centroid_x - 0.5) ** 2 + (centroid_y - 0.5) ** 2) / 0.72,
        )

        size_scale = np.tanh(area / 800.0)

        field_coherence = np.clip(
            0.40 * compactness
            + 0.35 * (mean_overlap / 3.0)
            + 0.25 * spatial_centering,
            0,
            1,
        )

        field_density = np.clip(
            (mean_density * mean_overlap * size_scale) / 4.0,
            0,
            1.5,
        )

        field_range = np.clip(
            np.sqrt(area) / 55.0 * (0.5 + field_coherence),
            0,
            1.5,
        )

        field_gradient = np.clip(
            (max_density - mean_density) * (1.0 + compactness),
            0,
            2.0,
        )

        field_flux = np.clip(
            field_density * field_range * (0.5 + field_coherence),
            0,
            2.0,
        )

        field_order = np.clip(
            0.38 * field_coherence
            + 0.30 * field_density
            + 0.20 * field_range
            + 0.12 * np.tanh(contact_capacity / 4.0),
            0,
            1,
        )

        perturbation_penalty = np.clip(1.0 - 1.7 * perturbation, 0.65, 1.0)

        global_field_stability = np.clip(
            (
                0.34 * field_density
                + 0.24 * field_range
                + 0.20 * field_coherence
                + 0.12 * field_flux
                + 0.10 * np.tanh(contact_capacity / 4.0)
            )
            * perturbation_penalty,
            0,
            1,
        )

        regime = classify_field_regime(cluster_class)

        local_field_score = np.clip(
            0.35 * field_coherence
            + 0.25 * field_order
            + 0.20 * np.tanh(contact_capacity / 3.0)
            + 0.20 * np.tanh(field_gradient / 1.2),
            0,
            1,
        )

        extended_field_score = np.clip(
            0.30 * field_density
            + 0.25 * field_range
            + 0.20 * field_flux
            + 0.15 * field_coherence
            + 0.10 * np.tanh(contact_capacity / 4.0),
            0,
            1,
        )

        if regime == "local_field":
            field_presence_score = local_field_score
        elif regime == "extended_field":
            field_presence_score = extended_field_score
        else:
            field_presence_score = 0.50 * local_field_score + 0.50 * extended_field_score

        rows.append({
            "source": source,
            "expected_field": 1 if cluster_class >= 0 else 0,
            "field_regime": regime,
            "is_valid_field": 1 if cluster_class >= 0 else 0,
            "is_local_field": 1 if regime == "local_field" else 0,
            "is_extended_field": 1 if regime == "extended_field" else 0,
            "perturbation": perturbation,
            "cluster_class": cluster_class,
            "area": area,
            "compactness": compactness,
            "mean_density": mean_density,
            "max_density": max_density,
            "mean_overlap": mean_overlap,
            "contact_capacity": contact_capacity,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "field_density": field_density,
            "field_coherence": field_coherence,
            "field_range": field_range,
            "field_gradient": field_gradient,
            "field_flux": field_flux,
            "field_order": field_order,
            "global_field_stability": global_field_stability,
            "local_field_score": local_field_score,
            "extended_field_score": extended_field_score,
            "field_presence_score": field_presence_score,
        })

    return pd.DataFrame(rows)


def supervised_accuracy(df, target):
    features = [
        "area",
        "compactness",
        "mean_density",
        "max_density",
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
        n_estimators=240,
        max_depth=8,
        random_state=SEED,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    return float(accuracy_score(y_test, pred))


def safe_silhouette(df, target, features):
    try:
        if len(df[target].unique()) < 2:
            return np.nan
        return float(silhouette_score(df[features], df[target]))
    except Exception:
        return np.nan


def summarize(samples):
    valid = samples[samples["is_valid_field"] == 1]
    invalid = samples[samples["is_valid_field"] == 0]
    local = samples[samples["field_regime"] == "local_field"]
    extended = samples[samples["field_regime"] == "extended_field"]

    local_threshold = 0.32
    extended_threshold = 0.45
    invalid_threshold = 0.32

    field_presence_accuracy = supervised_accuracy(samples, "is_valid_field")

    local_binary = samples[samples["field_regime"].isin(["local_field", "invalid_control"])]
    extended_binary = samples[samples["field_regime"].isin(["extended_field", "invalid_control"])]

    local_field_accuracy = supervised_accuracy(local_binary, "is_local_field")
    extended_field_accuracy = supervised_accuracy(extended_binary, "is_extended_field")

    sil_features = [
        "local_field_score",
        "extended_field_score",
        "field_presence_score",
        "field_coherence",
        "field_density",
        "field_range",
    ]

    regime_silhouette = safe_silhouette(samples, "field_regime", sil_features)

    local_field_support = float((local["local_field_score"] >= local_threshold).mean()) if len(local) else 0.0
    extended_field_support = float((extended["extended_field_score"] >= extended_threshold).mean()) if len(extended) else 0.0
    invalid_suppression = float((invalid["field_presence_score"] < invalid_threshold).mean()) if len(invalid) else 0.0

    mean_local = float(local["local_field_score"].mean()) if len(local) else 0.0
    mean_extended = float(extended["extended_field_score"].mean()) if len(extended) else 0.0
    mean_invalid = float(invalid["field_presence_score"].mean()) if len(invalid) else 0.0

    local_contrast = mean_local / max(mean_invalid, 1e-9)
    extended_contrast = mean_extended / max(mean_invalid, 1e-9)

    density_order_corr = float(
        np.corrcoef(samples["field_density"], samples["field_order"])[0, 1]
    )

    coherence_presence_corr = float(
        np.corrcoef(samples["field_coherence"], samples["field_presence_score"])[0, 1]
    )

    verdict = (
        "field_emergence_supported"
        if (
            field_presence_accuracy >= 0.98
            and local_field_support >= 0.95
            and extended_field_support >= 0.95
            and invalid_suppression >= 0.95
            and extended_contrast > 2.0
            and local_contrast > 1.5
            and density_order_corr > 0.60
        )
        else "field_emergence_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(samples),
        "num_local_field": len(local),
        "num_extended_field": len(extended),
        "num_invalid_controls": len(invalid),
        "field_presence_accuracy": field_presence_accuracy,
        "local_field_accuracy": local_field_accuracy,
        "extended_field_accuracy": extended_field_accuracy,
        "regime_silhouette": regime_silhouette,
        "local_field_support": local_field_support,
        "extended_field_support": extended_field_support,
        "invalid_suppression": invalid_suppression,
        "mean_local_field_score": mean_local,
        "mean_extended_field_score": mean_extended,
        "mean_invalid_field_score": mean_invalid,
        "local_field_contrast": local_contrast,
        "extended_field_contrast": extended_contrast,
        "density_order_corr": density_order_corr,
        "coherence_presence_corr": coherence_presence_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "electron_identity_used": False,
        "photon_identity_used": False,
        "quark_identity_used": False,
        "neutrino_identity_used": False,
        "safe_hierarchy": "W -> C -> local/extended field -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(samples):
    by_source = samples.groupby("source").agg(
        count=("source", "count"),
        mean_field_density=("field_density", "mean"),
        mean_field_coherence=("field_coherence", "mean"),
        mean_field_range=("field_range", "mean"),
        mean_field_flux=("field_flux", "mean"),
        mean_field_order=("field_order", "mean"),
        mean_global_field_stability=("global_field_stability", "mean"),
        mean_local_field_score=("local_field_score", "mean"),
        mean_extended_field_score=("extended_field_score", "mean"),
        mean_field_presence_score=("field_presence_score", "mean"),
        field_presence_ratio=("field_presence_score", lambda x: float((x >= 0.32).mean())),
    ).reset_index()

    by_regime = samples.groupby("field_regime").agg(
        count=("field_regime", "count"),
        mean_area=("area", "mean"),
        mean_contact_capacity=("contact_capacity", "mean"),
        mean_field_density=("field_density", "mean"),
        mean_field_coherence=("field_coherence", "mean"),
        mean_field_range=("field_range", "mean"),
        mean_local_field_score=("local_field_score", "mean"),
        mean_extended_field_score=("extended_field_score", "mean"),
        mean_field_presence_score=("field_presence_score", "mean"),
    ).reset_index()

    by_cluster = samples.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        regime=("field_regime", lambda x: x.mode().iloc[0]),
        mean_area=("area", "mean"),
        mean_contact_capacity=("contact_capacity", "mean"),
        mean_field_density=("field_density", "mean"),
        mean_field_coherence=("field_coherence", "mean"),
        mean_field_range=("field_range", "mean"),
        mean_global_field_stability=("global_field_stability", "mean"),
        mean_local_field_score=("local_field_score", "mean"),
        mean_extended_field_score=("extended_field_score", "mean"),
        mean_field_presence_score=("field_presence_score", "mean"),
    ).reset_index()

    return by_source, by_regime, by_cluster


def make_plots(samples, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in samples.groupby("field_regime"):
        plt.scatter(
            sub["local_field_score"],
            sub["extended_field_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("local_field_score")
    plt.ylabel("extended_field_score")
    plt.title("BST field regimes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "field_regime_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(valid["cluster_class"].astype(str), valid["mean_field_presence_score"])
    plt.xlabel("valid cluster_class")
    plt.ylabel("mean field_presence_score")
    plt.title("BST field presence by wave-cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "field_presence_by_cluster.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in samples.groupby("field_regime"):
        plt.scatter(
            sub["field_coherence"],
            sub["field_presence_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("field_coherence")
    plt.ylabel("field_presence_score")
    plt.title("BST field emergence: coherence to field presence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "field_coherence_presence.png", dpi=220)
    plt.close()

    heat = samples.pivot_table(
        index="cluster_class",
        columns="perturbation",
        values="field_presence_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="mean field_presence_score")
    plt.xticks(range(len(heat.columns)), [f"{c:.3f}" for c in heat.columns], rotation=45)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("perturbation")
    plt.ylabel("cluster_class")
    plt.title("BST field presence heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "field_presence_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 60 FIELD EMERGENCE TEST ===\n")

    events = load_wave_cluster_events()
    wave_summary = load_optional(WAVE_SUMMARY)
    wave_by_pert = load_optional(WAVE_BY_PERT)

    valid = events[events["cluster_class"] >= 0].copy()
    valid = valid.sample(n=min(9000, len(valid)), random_state=SEED).reset_index(drop=True)

    n_control = len(valid) // 2

    invalid_noise = make_invalid_noise(events, n_control)
    pseudo_invalid = make_pseudo_invalid_clusters(events, n_control)

    valid_features = compute_field_features(valid, "valid_cluster")
    invalid_features = compute_field_features(invalid_noise, "invalid_noise")
    pseudo_features = compute_field_features(pseudo_invalid, "pseudo_cluster_invalid")

    samples = pd.concat(
        [valid_features, invalid_features, pseudo_features],
        ignore_index=True,
    )

    summary = summarize(samples)
    by_source, by_regime, by_cluster = aggregate(samples)

    samples.to_csv(OUT / "field_emergence_samples.csv", index=False)
    summary.to_csv(OUT / "field_emergence_summary.csv", index=False)
    by_source.to_csv(OUT / "field_emergence_by_source.csv", index=False)
    by_regime.to_csv(OUT / "field_emergence_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "field_emergence_by_cluster.csv", index=False)

    if not wave_summary.empty:
        wave_summary.to_csv(OUT / "field_emergence_source_wave_summary.csv", index=False)

    if not wave_by_pert.empty:
        wave_by_pert.to_csv(OUT / "field_emergence_source_wave_by_perturbation.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nField emergence by source:")
    print(by_source.to_string(index=False))

    print("\nField emergence by regime:")
    print(by_regime.to_string(index=False))

    print("\nField emergence by cluster:")
    print(by_cluster.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'field_emergence_samples.csv'}")
    print(f"[OK] wrote {OUT / 'field_emergence_summary.csv'}")
    print(f"[OK] wrote {OUT / 'field_emergence_by_source.csv'}")
    print(f"[OK] wrote {OUT / 'field_emergence_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'field_emergence_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'field_regime_projection.png'}")
    print(f"[OK] wrote {OUT / 'field_presence_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'field_coherence_presence.png'}")
    print(f"[OK] wrote {OUT / 'field_presence_heatmap.png'}")
    print("[DONE] field emergence test complete")


if __name__ == "__main__":
    main()