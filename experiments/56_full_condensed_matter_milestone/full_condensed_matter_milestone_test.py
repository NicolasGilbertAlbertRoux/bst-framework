#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 56 — Full Condensed Matter Milestone

Place this file at:
    experiments/56_full_condensed_matter_milestone/full_condensed_matter_milestone_test.py

Run:
    python experiments/56_full_condensed_matter_milestone/full_condensed_matter_milestone_test.py

Purpose
-------
Aggregate the validated BST condensed-matter chain:

    51  Lattice Exchange Networks
    52  Magnon Emergence
    53  Phonon Emergence
    54  Electron–Phonon Coupling
    55  Superconductivity Precursors

and test whether the whole hierarchy remains coherent as a single emergent
pipeline:

    exchange lattice
        -> spin collective modes / magnons
        -> lattice collective modes / phonons
        -> electron–phonon coupling
        -> superconductivity precursors

This script does NOT rerun tests 51→55. It reads their CSV outputs from:
    results/research_final/

If a required result is missing, rerun the corresponding test first.

Outputs
-------
results/research_final/full_condensed_matter_milestone_test/
    condensed_matter_summary.csv
    condensed_matter_chain.csv
    condensed_matter_stage_metrics.csv
    condensed_matter_material_synthesis.csv
    condensed_matter_pipeline.png
    condensed_matter_stage_scores.png
    condensed_matter_material_map.png
"""

from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


BASE = Path("results/research_final")
OUT = BASE / "full_condensed_matter_milestone_test"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


REQUIRED = {
    "lattice": {
        "dir": "lattice_exchange_networks_test",
        "summary": "lattice_exchange_summary.csv",
        "detail": "lattice_exchange_by_family.csv",
        "expected_verdict": "lattice_exchange_networks_supported",
    },
    "magnon": {
        "dir": "magnon_emergence_test",
        "summary": "magnon_summary.csv",
        "detail": "magnon_by_ordering.csv",
        "material": "magnon_by_material.csv",
        "expected_verdict": "magnon_emergence_supported",
    },
    "phonon": {
        "dir": "phonon_emergence_test",
        "summary": "phonon_summary.csv",
        "detail": "phonon_by_class.csv",
        "material": "phonon_by_material.csv",
        "expected_verdict": "phonon_emergence_supported",
    },
    "electron_phonon": {
        "dir": "electron_phonon_coupling_test",
        "summary": "electron_phonon_summary.csv",
        "detail": "electron_phonon_by_class.csv",
        "material": "electron_phonon_by_material.csv",
        "expected_verdict": "electron_phonon_coupling_supported",
    },
    "superconductivity": {
        "dir": "superconductivity_precursors_test",
        "summary": "superconductivity_summary.csv",
        "detail": "superconductivity_by_class.csv",
        "material": "superconductivity_by_material.csv",
        "expected_verdict": "superconductivity_precursors_supported",
    },
}


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Please rerun the corresponding previous BST experiment first."
        )
    return pd.read_csv(path)


def scalar(df: pd.DataFrame, col: str, default: float | None = None) -> float:
    if col not in df.columns:
        if default is None:
            raise KeyError(f"Missing column '{col}' in dataframe columns={list(df.columns)}")
        return float(default)
    return float(df[col].iloc[0])


def text_scalar(df: pd.DataFrame, col: str, default: str = "") -> str:
    if col not in df.columns:
        return default
    return str(df[col].iloc[0])


def clamp01(x: float) -> float:
    if np.isnan(x):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def contrast_score(x: float, scale: float = 10.0) -> float:
    # Smoothly maps contrast 1->~0.095, 10->0.632, 100->0.99995
    x = max(float(x), 0.0)
    return float(1.0 - math.exp(-x / scale))


def geometric_mean(values: List[float]) -> float:
    arr = np.array([max(v, EPS) for v in values], dtype=float)
    return float(np.exp(np.mean(np.log(arr))))


def harmonic_mean(values: List[float]) -> float:
    arr = np.array([max(v, EPS) for v in values], dtype=float)
    return float(len(arr) / np.sum(1.0 / arr))


def load_all() -> Dict[str, Dict[str, pd.DataFrame]]:
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for stage, cfg in REQUIRED.items():
        stage_dir = BASE / cfg["dir"]
        data[stage] = {}
        data[stage]["summary"] = read_csv_required(stage_dir / cfg["summary"])
        data[stage]["detail"] = read_csv_required(stage_dir / cfg["detail"])
        if "material" in cfg:
            data[stage]["material"] = read_csv_required(stage_dir / cfg["material"])
    return data


def compute_stage_scores(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    # 51 — lattice
    s = data["lattice"]["summary"]
    lattice_score = geometric_mean([
        scalar(s, "family_accuracy"),
        scalar(s, "block_accuracy"),
        clamp01(scalar(s, "community_ARI")),
        clamp01(scalar(s, "community_NMI")),
        clamp01(scalar(s, "lattice_stability") / 0.65),
        scalar(s, "transition_recovery"),
        scalar(s, "lanthanide_recovery"),
        scalar(s, "actinide_recovery"),
        scalar(s, "noble_isolation"),
    ])
    rows.append({
        "stage": "51_lattice_exchange_networks",
        "emergence_depth": 1,
        "primary_accuracy": scalar(s, "family_accuracy"),
        "stability": scalar(s, "lattice_stability"),
        "separation": clamp01(scalar(s, "community_NMI")),
        "suppression_or_control": scalar(s, "noble_isolation"),
        "support_score": lattice_score,
        "verdict": text_scalar(s, "verdict"),
    })

    # 52 — magnons
    s = data["magnon"]["summary"]
    magnon_score = geometric_mean([
        scalar(s, "ordering_accuracy"),
        scalar(s, "stability_accuracy"),
        scalar(s, "stable_ratio_valid"),
        1.0 - scalar(s, "stable_ratio_invalid"),
        clamp01(scalar(s, "ordering_silhouette")),
        scalar(s, "ferro_support"),
        scalar(s, "antiferro_support"),
        scalar(s, "ferri_support"),
        scalar(s, "frustrated_support"),
        scalar(s, "nonmagnetic_suppression"),
        scalar(s, "none_suppression"),
        contrast_score(scalar(s, "stability_contrast"), scale=20.0),
    ])
    rows.append({
        "stage": "52_magnon_emergence",
        "emergence_depth": 2,
        "primary_accuracy": scalar(s, "ordering_accuracy"),
        "stability": scalar(s, "mean_valid_stability"),
        "separation": clamp01(scalar(s, "ordering_silhouette")),
        "suppression_or_control": geometric_mean([
            scalar(s, "nonmagnetic_suppression"),
            scalar(s, "none_suppression"),
        ]),
        "support_score": magnon_score,
        "verdict": text_scalar(s, "verdict"),
    })

    # 53 — phonons
    s = data["phonon"]["summary"]
    phonon_score = geometric_mean([
        scalar(s, "class_accuracy"),
        scalar(s, "lattice_accuracy"),
        scalar(s, "stability_accuracy"),
        scalar(s, "stable_ratio_valid"),
        1.0 - scalar(s, "stable_ratio_invalid"),
        clamp01(scalar(s, "class_silhouette")),
        scalar(s, "acoustic_support"),
        scalar(s, "optical_branch_recovery"),
        scalar(s, "invalid_suppression"),
        scalar(s, "vdw_low_velocity"),
        contrast_score(scalar(s, "stability_contrast"), scale=25.0),
    ])
    rows.append({
        "stage": "53_phonon_emergence",
        "emergence_depth": 3,
        "primary_accuracy": scalar(s, "class_accuracy"),
        "stability": scalar(s, "mean_valid_stability"),
        "separation": clamp01(scalar(s, "class_silhouette")),
        "suppression_or_control": geometric_mean([
            scalar(s, "invalid_suppression"),
            scalar(s, "vdw_low_velocity"),
        ]),
        "support_score": phonon_score,
        "verdict": text_scalar(s, "verdict"),
    })

    # 54 — electron-phonon
    s = data["electron_phonon"]["summary"]
    ep_score = geometric_mean([
        scalar(s, "class_accuracy"),
        scalar(s, "coupling_class_accuracy"),
        scalar(s, "stability_accuracy"),
        scalar(s, "stable_ratio_valid"),
        1.0 - scalar(s, "stable_ratio_invalid"),
        clamp01(scalar(s, "coupling_silhouette")),
        scalar(s, "weak_recovery"),
        scalar(s, "moderate_recovery"),
        scalar(s, "strong_recovery"),
        scalar(s, "precursor_recovery"),
        scalar(s, "insulator_suppression"),
        scalar(s, "invalid_suppression"),
        contrast_score(scalar(s, "stability_contrast"), scale=15.0),
    ])
    rows.append({
        "stage": "54_electron_phonon_coupling",
        "emergence_depth": 4,
        "primary_accuracy": geometric_mean([
            scalar(s, "class_accuracy"),
            scalar(s, "coupling_class_accuracy"),
        ]),
        "stability": scalar(s, "mean_valid_stability"),
        "separation": clamp01(scalar(s, "coupling_silhouette")),
        "suppression_or_control": geometric_mean([
            scalar(s, "insulator_suppression"),
            scalar(s, "invalid_suppression"),
        ]),
        "support_score": ep_score,
        "verdict": text_scalar(s, "verdict"),
    })

    # 55 — superconductivity
    s = data["superconductivity"]["summary"]
    sc_score = geometric_mean([
        scalar(s, "family_accuracy"),
        scalar(s, "precursor_class_accuracy"),
        scalar(s, "stability_accuracy"),
        scalar(s, "stable_ratio_valid"),
        1.0 - scalar(s, "stable_ratio_invalid"),
        clamp01(scalar(s, "precursor_silhouette")),
        scalar(s, "weak_precursor_recovery"),
        scalar(s, "strong_precursor_recovery"),
        scalar(s, "phonon_sc_recovery"),
        scalar(s, "unconventional_recovery"),
        scalar(s, "non_precursor_suppression"),
        scalar(s, "insulator_suppression"),
        scalar(s, "semiconductor_suppression"),
        scalar(s, "invalid_suppression"),
        contrast_score(scalar(s, "stability_contrast"), scale=2.0),
    ])
    rows.append({
        "stage": "55_superconductivity_precursors",
        "emergence_depth": 5,
        "primary_accuracy": geometric_mean([
            scalar(s, "family_accuracy"),
            scalar(s, "precursor_class_accuracy"),
        ]),
        "stability": scalar(s, "mean_valid_stability"),
        "separation": clamp01(scalar(s, "precursor_silhouette")),
        "suppression_or_control": geometric_mean([
            scalar(s, "non_precursor_suppression"),
            scalar(s, "insulator_suppression"),
            scalar(s, "semiconductor_suppression"),
            scalar(s, "invalid_suppression"),
        ]),
        "support_score": sc_score,
        "verdict": text_scalar(s, "verdict"),
    })

    return pd.DataFrame(rows)


def build_material_synthesis(data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []

    # Magnon materials.
    m = data["magnon"]["material"].copy()
    for _, r in m.iterrows():
        rows.append({
            "material": r.get("material", ""),
            "module": "magnon",
            "family_or_class": r.get("expected_ordering", ""),
            "valid_or_precursor": r.get("expected_stable", np.nan),
            "collective_coherence": r.get("mean_spin_coherence", np.nan),
            "transport_or_velocity": r.get("mean_magnon_velocity", np.nan),
            "coupling_or_glue": r.get("mean_exchange_strength", np.nan),
            "stability": r.get("mean_magnon_stability", np.nan),
        })

    # Phonon materials.
    m = data["phonon"]["material"].copy()
    for _, r in m.iterrows():
        rows.append({
            "material": r.get("material", ""),
            "module": "phonon",
            "family_or_class": r.get("material_class", ""),
            "valid_or_precursor": r.get("expected_stable", np.nan),
            "collective_coherence": r.get("mean_dispersion_coherence", np.nan),
            "transport_or_velocity": r.get("mean_sound_velocity_proxy", np.nan),
            "coupling_or_glue": r.get("mean_bond_stiffness", np.nan),
            "stability": r.get("mean_phonon_stability", np.nan),
        })

    # Electron-phonon materials.
    m = data["electron_phonon"]["material"].copy()
    for _, r in m.iterrows():
        rows.append({
            "material": r.get("material", ""),
            "module": "electron_phonon",
            "family_or_class": r.get("coupling_class", ""),
            "valid_or_precursor": r.get("expected_stable", np.nan),
            "collective_coherence": r.get("mean_phase_locking", np.nan),
            "transport_or_velocity": 1.0 - r.get("mean_transport_scattering", np.nan),
            "coupling_or_glue": r.get("mean_lambda_ep", np.nan),
            "stability": r.get("mean_coupling_stability", np.nan),
        })

    # Superconductivity materials.
    m = data["superconductivity"]["material"].copy()
    for _, r in m.iterrows():
        rows.append({
            "material": r.get("material", ""),
            "module": "superconductivity",
            "family_or_class": r.get("precursor_class", ""),
            "valid_or_precursor": r.get("expected_precursor", np.nan),
            "collective_coherence": r.get("mean_phase_rigidity", np.nan),
            "transport_or_velocity": r.get("mean_zero_resistance_proxy", np.nan),
            "coupling_or_glue": r.get("mean_pair_glue", np.nan),
            "stability": r.get("mean_superconductivity_stability", np.nan),
        })

    synth = pd.DataFrame(rows)
    numeric_cols = [
        "valid_or_precursor",
        "collective_coherence",
        "transport_or_velocity",
        "coupling_or_glue",
        "stability",
    ]
    for col in numeric_cols:
        synth[col] = pd.to_numeric(synth[col], errors="coerce")

    synth["emergent_material_score"] = synth[
        ["collective_coherence", "transport_or_velocity", "coupling_or_glue", "stability"]
    ].fillna(0.0).clip(lower=0.0).apply(lambda row: geometric_mean(row.tolist()), axis=1)

    return synth


def compute_cross_stage_consistency(data: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[float, pd.DataFrame]:
    checks = []

    # Magnon: magnetic classes should beat none/nonmagnetic.
    mag = data["magnon"]["detail"]
    magnetic = mag[mag["expected_ordering"].isin(["ferromagnetic", "antiferromagnetic", "ferrimagnetic", "frustrated"])]
    controls = mag[mag["expected_ordering"].isin(["none", "nonmagnetic"])]
    mag_gap = float(magnetic["mean_magnon_stability"].mean() - controls["mean_magnon_stability"].mean())
    checks.append({
        "check": "magnon_modes_above_controls",
        "value": mag_gap,
        "passed": float(mag_gap > 0.25),
    })

    # Phonon: valid crystals should beat invalid.
    ph = data["phonon"]["detail"]
    valid_ph = ph[ph["material_class"] != "invalid"]
    invalid_ph = ph[ph["material_class"] == "invalid"]
    ph_gap = float(valid_ph["mean_phonon_stability"].mean() - invalid_ph["mean_phonon_stability"].mean())
    checks.append({
        "check": "phonon_crystals_above_invalid",
        "value": ph_gap,
        "passed": float(ph_gap > 0.30),
    })

    # E-ph: precursor/strong should beat suppressed/invalid.
    ep = data["electron_phonon"]["detail"]
    ep_high = ep[ep["material_class"].isin(["strong_coupling_metal", "phonon_superconductor_precursor"])]
    ep_low = ep[ep["material_class"].isin(["invalid", "vdw_insulator", "ionic_insulator"])]
    ep_gap = float(ep_high["mean_coupling_stability"].mean() - ep_low["mean_coupling_stability"].mean())
    checks.append({
        "check": "electron_phonon_high_coupling_above_suppressed",
        "value": ep_gap,
        "passed": float(ep_gap > 0.35),
    })

    # SC: true precursor classes should beat non/suppressed/invalid.
    sc = data["superconductivity"]["detail"]
    sc_high = sc[sc["precursor_class"].isin([
        "weak_precursor",
        "strong_precursor",
        "phonon_superconductor",
        "unconventional_precursor",
    ])]
    sc_low = sc[sc["precursor_class"].isin([
        "invalid",
        "suppressed",
        "semiconducting",
        "non_precursor",
    ])]
    sc_gap = float(sc_high["mean_superconductivity_stability"].mean() - sc_low["mean_superconductivity_stability"].mean())
    checks.append({
        "check": "superconductivity_precursors_above_controls",
        "value": sc_gap,
        "passed": float(sc_gap > 0.08),
    })

    # E-ph to SC alignment: phonon superconductors should have high EP stability and SC stability.
    ep_cls = ep[ep["material_class"] == "phonon_superconductor_precursor"]
    sc_cls = sc[sc["precursor_class"] == "phonon_superconductor"]
    ep_sc_alignment = float(
        0.5 * ep_cls["mean_coupling_stability"].mean()
        + 0.5 * sc_cls["mean_superconductivity_stability"].mean()
    )
    checks.append({
        "check": "electron_phonon_to_sc_alignment",
        "value": ep_sc_alignment,
        "passed": float(ep_sc_alignment > 0.65),
    })

    check_df = pd.DataFrame(checks)
    consistency = float(check_df["passed"].mean())
    return consistency, check_df


def plot_outputs(stage_df: pd.DataFrame, synth: pd.DataFrame, summary: pd.DataFrame) -> None:
    # Pipeline diagram.
    names = [
        "Proto-cores",
        "Lattice\nexchange",
        "Magnons",
        "Phonons",
        "Electron–phonon\ncoupling",
        "SC\nprecursors",
    ]
    xs = np.arange(len(names), dtype=float)
    ys = np.zeros_like(xs)

    plt.figure(figsize=(12, 3.8))
    for i, (x, name) in enumerate(zip(xs, names)):
        plt.scatter([x], [0], s=900)
        plt.text(x, 0, name, ha="center", va="center", fontsize=9)
        if i < len(xs) - 1:
            plt.annotate(
                "",
                xy=(x + 0.78, 0),
                xytext=(x + 0.22, 0),
                arrowprops=dict(arrowstyle="->", lw=1.6),
            )

    score = float(summary["condensed_matter_chain_score"].iloc[0])
    verdict = str(summary["verdict"].iloc[0])
    plt.text(
        2.5,
        -0.55,
        f"BST condensed-matter chain score = {score:.3f}   |   verdict = {verdict}",
        ha="center",
        fontsize=10,
    )
    plt.ylim(-0.8, 0.5)
    plt.axis("off")
    plt.title("BST full condensed matter milestone — emergence pipeline")
    plt.tight_layout()
    plt.savefig(OUT / "condensed_matter_pipeline.png", dpi=220)
    plt.close()

    # Stage scores.
    plt.figure(figsize=(10, 5))
    plt.bar(stage_df["stage"], stage_df["support_score"])
    plt.axhline(0.80, linestyle="--", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("support score")
    plt.ylim(0, 1.05)
    plt.title("BST condensed matter stage support scores")
    plt.tight_layout()
    plt.savefig(OUT / "condensed_matter_stage_scores.png", dpi=220)
    plt.close()

    # Material synthesis PCA map.
    plot_df = synth.dropna(subset=[
        "collective_coherence",
        "transport_or_velocity",
        "coupling_or_glue",
        "stability",
    ]).copy()

    if len(plot_df) >= 3:
        X = plot_df[[
            "collective_coherence",
            "transport_or_velocity",
            "coupling_or_glue",
            "stability",
        ]].to_numpy(float)
        Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)
        proj = PCA(n_components=2).fit_transform(Xn)
        plot_df["pc1"] = proj[:, 0]
        plot_df["pc2"] = proj[:, 1]

        plt.figure(figsize=(11, 8))
        for module in sorted(plot_df["module"].unique()):
            sub = plot_df[plot_df["module"] == module]
            plt.scatter(sub["pc1"], sub["pc2"], s=34, label=module)
        # Label only high-scoring or representative points to keep it readable.
        lab = plot_df.sort_values("emergent_material_score", ascending=False).head(18)
        for _, r in lab.iterrows():
            plt.text(r["pc1"], r["pc2"], str(r["material"]), fontsize=7, ha="center", va="center")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("BST condensed matter material synthesis map")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT / "condensed_matter_material_map.png", dpi=220)
        plt.close()


def main() -> int:
    try:
        data = load_all()
    except Exception as exc:
        print("\n=== BST FULL CONDENSED MATTER MILESTONE ===\n")
        print("[ERROR] Could not load required previous test outputs.\n")
        print(str(exc))
        print("\nExpected previous result folders:")
        for stage, cfg in REQUIRED.items():
            print(f" - {BASE / cfg['dir']}")
        return 2

    stage_df = compute_stage_scores(data)
    synth = build_material_synthesis(data)
    cross_consistency, chain_checks = compute_cross_stage_consistency(data)

    support_scores = stage_df["support_score"].to_numpy(float)
    chain_score = geometric_mean(support_scores.tolist())
    chain_harmonic_score = harmonic_mean(support_scores.tolist())
    weakest_stage_score = float(np.min(support_scores))
    weakest_stage = str(stage_df.iloc[int(np.argmin(support_scores))]["stage"])

    expected_verdicts_ok = []
    for stage, cfg in REQUIRED.items():
        actual = text_scalar(data[stage]["summary"], "verdict")
        expected_verdicts_ok.append(float(actual == cfg["expected_verdict"]))

    verdict_integrity = float(np.mean(expected_verdicts_ok))

    emergence_depth = int(stage_df["emergence_depth"].max())
    mean_stage_stability = float(stage_df["stability"].mean())
    mean_stage_separation = float(stage_df["separation"].mean())
    mean_control_suppression = float(stage_df["suppression_or_control"].mean())

    # Strict but not brittle: all previous verdicts must be supported, chain score high,
    # no stage collapse, and cross-stage consistency must pass.
    if (
        verdict_integrity == 1.0
        and chain_score >= 0.80
        and chain_harmonic_score >= 0.78
        and weakest_stage_score >= 0.70
        and cross_consistency >= 0.80
    ):
        verdict = "full_condensed_matter_milestone_supported"
    elif (
        verdict_integrity >= 0.80
        and chain_score >= 0.65
        and weakest_stage_score >= 0.55
        and cross_consistency >= 0.60
    ):
        verdict = "weak_full_condensed_matter_milestone"
    else:
        verdict = "full_condensed_matter_milestone_not_supported"

    summary = pd.DataFrame([{
        "num_stages": len(stage_df),
        "emergence_depth": emergence_depth,
        "condensed_matter_chain_score": chain_score,
        "condensed_matter_harmonic_score": chain_harmonic_score,
        "weakest_stage": weakest_stage,
        "weakest_stage_score": weakest_stage_score,
        "verdict_integrity": verdict_integrity,
        "cross_stage_consistency": cross_consistency,
        "mean_stage_stability": mean_stage_stability,
        "mean_stage_separation": mean_stage_separation,
        "mean_control_suppression": mean_control_suppression,
        "num_synthesis_rows": len(synth),
        "mean_material_synthesis_score": float(synth["emergent_material_score"].mean()),
        "top_material_synthesis_score": float(synth["emergent_material_score"].max()),
        "verdict": verdict,
    }])

    chain_df = pd.DataFrame([
        {
            "from_stage": "proto_cores",
            "to_stage": "51_lattice_exchange_networks",
            "transition": "proto-core signatures organize into exchange networks",
            "depth": 1,
        },
        {
            "from_stage": "51_lattice_exchange_networks",
            "to_stage": "52_magnon_emergence",
            "transition": "exchange networks support collective spin waves",
            "depth": 2,
        },
        {
            "from_stage": "51_lattice_exchange_networks",
            "to_stage": "53_phonon_emergence",
            "transition": "exchange networks support collective lattice vibrations",
            "depth": 3,
        },
        {
            "from_stage": "53_phonon_emergence",
            "to_stage": "54_electron_phonon_coupling",
            "transition": "phonon modes couple to electronic transport channels",
            "depth": 4,
        },
        {
            "from_stage": "54_electron_phonon_coupling",
            "to_stage": "55_superconductivity_precursors",
            "transition": "coherent electron–phonon coupling seeds condensation channels",
            "depth": 5,
        },
    ])

    # Optional global silhouette across synthesized module-level rows.
    synth_numeric = synth[[
        "collective_coherence",
        "transport_or_velocity",
        "coupling_or_glue",
        "stability",
    ]].fillna(0.0).to_numpy(float)
    if len(synth) > 5 and len(set(synth["module"])) > 1:
        Xn = (synth_numeric - synth_numeric.mean(axis=0)) / (synth_numeric.std(axis=0) + EPS)
        try:
            module_sil = float(silhouette_score(Xn, synth["module"]))
        except Exception:
            module_sil = np.nan
    else:
        module_sil = np.nan
    summary["module_silhouette"] = module_sil

    stage_df.to_csv(OUT / "condensed_matter_stage_metrics.csv", index=False)
    chain_df.to_csv(OUT / "condensed_matter_chain.csv", index=False)
    chain_checks.to_csv(OUT / "condensed_matter_chain_checks.csv", index=False)
    synth.to_csv(OUT / "condensed_matter_material_synthesis.csv", index=False)
    summary.to_csv(OUT / "condensed_matter_summary.csv", index=False)

    plot_outputs(stage_df, synth, summary)

    print("\n=== BST FULL CONDENSED MATTER MILESTONE TEST ===\n")
    print(summary.to_string(index=False))

    print("\nCondensed matter stage metrics:")
    print(stage_df.to_string(index=False))

    print("\nCross-stage consistency checks:")
    print(chain_checks.to_string(index=False))

    print("\nTop material synthesis scores:")
    top = synth.sort_values("emergent_material_score", ascending=False).head(20)
    print(top.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'condensed_matter_summary.csv'}")
    print(f"[OK] wrote {OUT / 'condensed_matter_stage_metrics.csv'}")
    print(f"[OK] wrote {OUT / 'condensed_matter_chain.csv'}")
    print(f"[OK] wrote {OUT / 'condensed_matter_chain_checks.csv'}")
    print(f"[OK] wrote {OUT / 'condensed_matter_material_synthesis.csv'}")
    print(f"[OK] wrote {OUT / 'condensed_matter_pipeline.png'}")
    print(f"[OK] wrote {OUT / 'condensed_matter_stage_scores.png'}")
    print(f"[OK] wrote {OUT / 'condensed_matter_material_map.png'}")
    print("[DONE] full condensed matter milestone test complete")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
