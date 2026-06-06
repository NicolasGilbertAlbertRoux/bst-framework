#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 66 — Weak-Like Transition Channels

Purpose:
    Test whether BST bound states exhibit weak-like transition channels:
        instability -> family/regime transition -> decay proxy -> re-stabilization.

Important:
    This is NOT a Standard Model weak interaction claim.
    It tests an effective BST transition-channel proxy.

Canonical hierarchy:
    W -> C -> field -> gauge-like invariants -> metric proxy
      -> stress-energy proxy -> electromagnetic-like interaction
      -> nuclear-like binding -> weak-like transition channels
      -> P/T -> matter

Strict rules:
    - no exact W->T mapping;
    - no particle identity;
    - no quark identity;
    - no neutrino identity;
    - no Standard Model weak claim.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


SEED = 1729
rng = np.random.default_rng(SEED)

INP = Path("results/research_final/nuclear_like_binding_test/nuclear_like_binding_samples.csv")
OUT = Path("results/research_final/weak_like_transition_channels_test")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
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
    "field_gradient",
    "field_flux",
    "field_presence_score",
    "gauge_invariant_norm",
    "metric_potential",
    "curvature_density",
    "curvature_coherence",
    "effective_curvature_score",
    "energy_density_proxy",
    "pressure_proxy",
    "flux_proxy",
    "stress_energy_coupling",
    "conservation_proxy",
    "polarity_proxy",
    "charge_separation_proxy",
    "em_field_strength_proxy",
    "screening_proxy",
    "em_interaction_score",
    "cohesion_proxy",
    "confinement_proxy",
    "binding_glue_proxy",
    "overload_proxy",
    "fragmentation_proxy",
    "binding_energy_proxy",
    "binding_stability_proxy",
    "island_stability_proxy",
]


def load_input():
    if not INP.exists():
        raise FileNotFoundError(
            f"Missing required input: {INP}\n"
            "Run experiments/65_nuclear_like_binding/nuclear_like_binding_test.py first."
        )

    df = pd.read_csv(INP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INP}: {missing}")

    return df


def classify_transition_regime(row):
    if row["binding_regime"] == "unbound_screened":
        return "suppressed_transition"

    if row["binding_regime"] == "localized_bound_state":
        return "local_reconfiguration_channel"

    if row["binding_regime"] == "extended_binding_basin":
        return "basin_decay_channel"

    return "unclassified_transition"


def compute_transition_features(df):
    df = df.copy()

    fc = df["field_coherence"].astype(float)
    fr = df["field_range"].astype(float)
    fg = df["field_gradient"].astype(float)
    ff = df["field_flux"].astype(float)
    fp = df["field_presence_score"].astype(float)

    gn = df["gauge_invariant_norm"].astype(float)

    mp = df["metric_potential"].astype(float)
    cd = df["curvature_density"].astype(float)
    cc = df["curvature_coherence"].astype(float)
    ecs = df["effective_curvature_score"].astype(float)

    ed = df["energy_density_proxy"].astype(float)
    pr = df["pressure_proxy"].astype(float)
    fl = df["flux_proxy"].astype(float)
    se = df["stress_energy_coupling"].astype(float)
    cons = df["conservation_proxy"].astype(float)

    pol = df["polarity_proxy"].astype(float)
    charge = df["charge_separation_proxy"].astype(float)
    ems = df["em_field_strength_proxy"].astype(float)
    screen = df["screening_proxy"].astype(float)
    emi = df["em_interaction_score"].astype(float)

    coh = df["cohesion_proxy"].astype(float)
    conf = df["confinement_proxy"].astype(float)
    glue = df["binding_glue_proxy"].astype(float)
    overload = df["overload_proxy"].astype(float)
    frag = df["fragmentation_proxy"].astype(float)
    be = df["binding_energy_proxy"].astype(float)
    bs = df["binding_stability_proxy"].astype(float)
    island = df["island_stability_proxy"].astype(float)

    df["transition_regime"] = df.apply(classify_transition_regime, axis=1)

    df["instability_drive"] = np.clip(
        0.28 * frag
        + 0.22 * overload
        + 0.18 * np.maximum(emi - glue, 0)
        + 0.14 * np.abs(pol)
        + 0.10 * (1.0 - cons)
        + 0.08 * screen,
        0,
        5,
    )

    df["channel_opening_proxy"] = np.clip(
        0.24 * df["instability_drive"]
        + 0.20 * fg
        + 0.18 * charge
        + 0.14 * cd
        + 0.12 * se
        + 0.12 * np.maximum(0.75 - bs, 0),
        0,
        5,
    )

    df["family_transition_proxy"] = np.clip(
        0.26 * df["channel_opening_proxy"]
        + 0.22 * np.abs(fl - pr)
        + 0.18 * np.abs(fp - fc)
        + 0.14 * np.abs(be - bs)
        + 0.10 * np.tanh(fr)
        + 0.10 * np.tanh(mp),
        0,
        5,
    )

    df["decay_proxy"] = np.clip(
        0.30 * df["channel_opening_proxy"]
        + 0.25 * df["instability_drive"]
        + 0.20 * df["family_transition_proxy"]
        + 0.15 * frag
        + 0.10 * np.maximum(overload - glue, 0),
        0,
        5,
    )

    df["restabilization_proxy"] = np.clip(
        0.30 * coh
        + 0.24 * conf
        + 0.18 * island
        + 0.14 * cons
        + 0.14 * np.maximum(glue - overload, 0),
        0,
        5,
    )

    df["transition_selectivity_proxy"] = np.clip(
        df["channel_opening_proxy"]
        * (0.45 + 0.25 * gn + 0.20 * cc + 0.10 * fc)
        / (1.0 + screen),
        0,
        5,
    )

    df["weak_like_transition_score"] = np.clip(
        0.28 * df["channel_opening_proxy"]
        + 0.24 * df["family_transition_proxy"]
        + 0.20 * df["decay_proxy"]
        + 0.16 * df["transition_selectivity_proxy"]
        + 0.12 * df["restabilization_proxy"],
        0,
        5,
    )

    df["transition_balance_proxy"] = np.clip(
        df["restabilization_proxy"] / (1e-9 + df["decay_proxy"] + df["restabilization_proxy"]),
        0,
        1,
    )

    df["suppression_proxy"] = np.clip(
        1.0
        - 0.24 * df["weak_like_transition_score"]
        - 0.22 * df["channel_opening_proxy"]
        - 0.18 * df["decay_proxy"]
        - 0.16 * charge
        - 0.10 * cd
        - 0.10 * se,
        0,
        1,
    )

    df["transition_supported"] = (
        (df["is_valid_field"] == 1)
        & (
            ((df["transition_regime"] == "local_reconfiguration_channel") & (df["weak_like_transition_score"] >= 0.35))
            | ((df["transition_regime"] == "basin_decay_channel") & (df["weak_like_transition_score"] >= 0.65))
        )
    ).astype(int)

    df["local_transition_supported"] = (
        (df["transition_regime"] == "local_reconfiguration_channel")
        & (df["weak_like_transition_score"] >= 0.35)
    ).astype(int)

    df["basin_transition_supported"] = (
        (df["transition_regime"] == "basin_decay_channel")
        & (df["weak_like_transition_score"] >= 0.65)
    ).astype(int)

    return df


def supervised_accuracy(df, target):
    features = [
        "field_coherence",
        "field_range",
        "field_gradient",
        "field_flux",
        "field_presence_score",
        "gauge_invariant_norm",
        "metric_potential",
        "curvature_density",
        "curvature_coherence",
        "effective_curvature_score",
        "energy_density_proxy",
        "pressure_proxy",
        "flux_proxy",
        "stress_energy_coupling",
        "conservation_proxy",
        "polarity_proxy",
        "charge_separation_proxy",
        "em_field_strength_proxy",
        "screening_proxy",
        "em_interaction_score",
        "cohesion_proxy",
        "confinement_proxy",
        "binding_glue_proxy",
        "overload_proxy",
        "fragmentation_proxy",
        "binding_energy_proxy",
        "binding_stability_proxy",
        "island_stability_proxy",
        "instability_drive",
        "channel_opening_proxy",
        "family_transition_proxy",
        "decay_proxy",
        "restabilization_proxy",
        "transition_selectivity_proxy",
        "weak_like_transition_score",
        "transition_balance_proxy",
        "suppression_proxy",
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
        n_estimators=300,
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
    suppressed = df[df["transition_regime"] == "suppressed_transition"]
    local = df[df["transition_regime"] == "local_reconfiguration_channel"]
    basin = df[df["transition_regime"] == "basin_decay_channel"]

    transition_accuracy = supervised_accuracy(df, "transition_supported")

    local_subset = df[df["transition_regime"].isin(["local_reconfiguration_channel", "suppressed_transition"])]
    basin_subset = df[df["transition_regime"].isin(["basin_decay_channel", "suppressed_transition"])]

    local_transition_accuracy = supervised_accuracy(local_subset, "local_transition_supported")
    basin_transition_accuracy = supervised_accuracy(basin_subset, "basin_transition_supported")

    local_support = float((local["weak_like_transition_score"] >= 0.35).mean()) if len(local) else 0.0
    basin_support = float((basin["weak_like_transition_score"] >= 0.65).mean()) if len(basin) else 0.0
    suppressed_rejection = float((suppressed["weak_like_transition_score"] < 0.35).mean()) if len(suppressed) else 0.0

    mean_local = float(local["weak_like_transition_score"].mean()) if len(local) else 0.0
    mean_basin = float(basin["weak_like_transition_score"].mean()) if len(basin) else 0.0
    mean_suppressed = float(suppressed["weak_like_transition_score"].mean()) if len(suppressed) else 0.0

    local_contrast = mean_local / max(mean_suppressed, 1e-9)
    basin_contrast = mean_basin / max(mean_suppressed, 1e-9)

    transition_silhouette = safe_silhouette(
        df,
        "transition_regime",
        [
            "instability_drive",
            "channel_opening_proxy",
            "family_transition_proxy",
            "decay_proxy",
            "restabilization_proxy",
            "transition_selectivity_proxy",
            "weak_like_transition_score",
        ],
    )

    instability_decay_corr = safe_corr(df["instability_drive"], df["decay_proxy"])
    channel_transition_corr = safe_corr(df["channel_opening_proxy"], df["family_transition_proxy"])
    binding_transition_corr = safe_corr(df["binding_stability_proxy"], df["weak_like_transition_score"])
    fragmentation_decay_corr = safe_corr(df["fragmentation_proxy"], df["decay_proxy"])
    restabilization_balance_corr = safe_corr(df["restabilization_proxy"], df["transition_balance_proxy"])

    verdict = (
        "weak_like_transition_channels_supported"
        if (
            transition_accuracy >= 0.98
            and suppressed_rejection >= 0.90
            and mean_suppressed < mean_local < mean_basin
            and local_contrast > 2.0
            and basin_contrast > 3.0
            and instability_decay_corr > 0.75
            and channel_transition_corr > 0.75
            and binding_transition_corr > 0.75
        )
        else "weak_like_transition_channels_not_supported"
    )

    return pd.DataFrame([{
        "num_samples": len(df),
        "num_suppressed_transition": len(suppressed),
        "num_local_reconfiguration_channel": len(local),
        "num_basin_decay_channel": len(basin),
        "transition_accuracy": transition_accuracy,
        "local_transition_accuracy": local_transition_accuracy,
        "basin_transition_accuracy": basin_transition_accuracy,
        "transition_silhouette": transition_silhouette,
        "local_transition_support": local_support,
        "basin_transition_support": basin_support,
        "suppressed_rejection": suppressed_rejection,
        "mean_local_transition": mean_local,
        "mean_basin_transition": mean_basin,
        "mean_suppressed_transition": mean_suppressed,
        "local_transition_contrast": local_contrast,
        "basin_transition_contrast": basin_contrast,
        "instability_decay_corr": instability_decay_corr,
        "channel_transition_corr": channel_transition_corr,
        "binding_transition_corr": binding_transition_corr,
        "fragmentation_decay_corr": fragmentation_decay_corr,
        "restabilization_balance_corr": restabilization_balance_corr,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "quark_identity_used": False,
        "neutrino_identity_used": False,
        "standard_model_weak_claim": False,
        "safe_hierarchy": "W -> C -> field -> gauge-like invariants -> metric proxy -> stress-energy proxy -> electromagnetic-like interaction -> nuclear-like binding -> weak-like transition channels -> P/T -> matter",
        "verdict": verdict,
    }])


def aggregate(df):
    by_regime = df.groupby("transition_regime").agg(
        count=("transition_regime", "count"),
        mean_instability_drive=("instability_drive", "mean"),
        mean_channel_opening=("channel_opening_proxy", "mean"),
        mean_family_transition=("family_transition_proxy", "mean"),
        mean_decay=("decay_proxy", "mean"),
        mean_restabilization=("restabilization_proxy", "mean"),
        mean_transition_selectivity=("transition_selectivity_proxy", "mean"),
        mean_weak_like_transition=("weak_like_transition_score", "mean"),
        mean_transition_balance=("transition_balance_proxy", "mean"),
        mean_suppression=("suppression_proxy", "mean"),
        support_ratio=("transition_supported", "mean"),
    ).reset_index()

    by_cluster = df.groupby("cluster_class").agg(
        count=("cluster_class", "count"),
        transition_regime=("transition_regime", lambda x: x.mode().iloc[0]),
        mean_instability_drive=("instability_drive", "mean"),
        mean_channel_opening=("channel_opening_proxy", "mean"),
        mean_family_transition=("family_transition_proxy", "mean"),
        mean_decay=("decay_proxy", "mean"),
        mean_restabilization=("restabilization_proxy", "mean"),
        mean_weak_like_transition=("weak_like_transition_score", "mean"),
        mean_transition_balance=("transition_balance_proxy", "mean"),
    ).reset_index()

    by_binding = df.groupby("binding_regime").agg(
        count=("binding_regime", "count"),
        mean_weak_like_transition=("weak_like_transition_score", "mean"),
        mean_decay=("decay_proxy", "mean"),
        mean_restabilization=("restabilization_proxy", "mean"),
        mean_transition_balance=("transition_balance_proxy", "mean"),
    ).reset_index()

    return by_regime, by_cluster, by_binding


def make_plots(df, by_cluster):
    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("transition_regime"):
        plt.scatter(
            sub["channel_opening_proxy"],
            sub["weak_like_transition_score"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("channel_opening_proxy")
    plt.ylabel("weak_like_transition_score")
    plt.title("BST weak-like transition channels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "weak_like_transition_projection.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, sub in df.groupby("transition_regime"):
        plt.scatter(
            sub["decay_proxy"],
            sub["restabilization_proxy"],
            s=8,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("decay_proxy")
    plt.ylabel("restabilization_proxy")
    plt.title("BST decay / restabilization proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "weak_like_decay_restabilization.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    valid = by_cluster[by_cluster["cluster_class"] >= 0]
    plt.bar(
        valid["cluster_class"].astype(str),
        valid["mean_weak_like_transition"],
    )
    plt.xlabel("cluster_class")
    plt.ylabel("mean_weak_like_transition")
    plt.title("BST weak-like transitions by cluster class")
    plt.tight_layout()
    plt.savefig(OUT / "weak_like_by_cluster.png", dpi=220)
    plt.close()

    heat = df.pivot_table(
        index="cluster_class",
        columns="transition_regime",
        values="weak_like_transition_score",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 4.8))
    plt.imshow(heat.values, aspect="auto")
    plt.colorbar(label="weak_like_transition_score")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=25)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xlabel("transition_regime")
    plt.ylabel("cluster_class")
    plt.title("BST weak-like transition heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "weak_like_transition_heatmap.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 66 WEAK-LIKE TRANSITION CHANNELS TEST ===\n")

    base = load_input()
    samples = compute_transition_features(base)

    summary = summarize(samples)
    by_regime, by_cluster, by_binding = aggregate(samples)

    samples.to_csv(OUT / "weak_like_transition_samples.csv", index=False)
    summary.to_csv(OUT / "weak_like_transition_summary.csv", index=False)
    by_regime.to_csv(OUT / "weak_like_transition_by_regime.csv", index=False)
    by_cluster.to_csv(OUT / "weak_like_transition_by_cluster.csv", index=False)
    by_binding.to_csv(OUT / "weak_like_transition_by_binding_regime.csv", index=False)

    make_plots(samples, by_cluster)

    print(summary.to_string(index=False))

    print("\nWeak-like transition by regime:")
    print(by_regime.to_string(index=False))

    print("\nWeak-like transition by cluster:")
    print(by_cluster.to_string(index=False))

    print("\nWeak-like transition by binding regime:")
    print(by_binding.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'weak_like_transition_samples.csv'}")
    print(f"[OK] wrote {OUT / 'weak_like_transition_summary.csv'}")
    print(f"[OK] wrote {OUT / 'weak_like_transition_by_regime.csv'}")
    print(f"[OK] wrote {OUT / 'weak_like_transition_by_cluster.csv'}")
    print(f"[OK] wrote {OUT / 'weak_like_transition_by_binding_regime.csv'}")
    print(f"[OK] wrote {OUT / 'weak_like_transition_projection.png'}")
    print(f"[OK] wrote {OUT / 'weak_like_decay_restabilization.png'}")
    print(f"[OK] wrote {OUT / 'weak_like_by_cluster.png'}")
    print(f"[OK] wrote {OUT / 'weak_like_transition_heatmap.png'}")
    print("[DONE] weak-like transition channels test complete")


if __name__ == "__main__":
    main()