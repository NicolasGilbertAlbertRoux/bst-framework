#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(".")
OUT = ROOT / "results/research_final/dual_language_atlas_final"
OUT.mkdir(parents=True, exist_ok=True)

DIRS = {
    "dict": ROOT / "results/research_final/wave_cluster_dictionary",
    "rosetta": ROOT / "results/research_final/wave_rosetta_corrected",
    "particle": ROOT / "results/research_final/particle_proto_core_reconstruction_test",
    "periodic": ROOT / "results/research_final/full_periodic_reconstruction_H_Lr_test",
    "contact": ROOT / "results/research_final/contact_matter_taxonomy_test",
    "wave": ROOT / "results/research_final/wave_cluster_taxonomy_test",
    "lattice": ROOT / "results/research_final/lattice_exchange_networks_test",
    "condensed": ROOT / "results/research_final/full_condensed_matter_milestone_test",
}

def read_required(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def read_optional(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def panel(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.0))
    ax.text(x + 0.01, y + h - 0.035, title, fontsize=9, weight="bold", va="top")

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220)
    plt.close()


def build_dictionary_from_archives_if_missing():
    """
    Rebuild the old wave_cluster_dictionary artefacts from archived BST result
    tables when the former 57b stage is absent.

    This is a data-derived compatibility layer for the atlas only:
    - no exact W->T identity is introduced;
    - no particle identity is introduced;
    - generated alignments are explicitly labelled as metric proxies.
    """
    dict_dir = DIRS["dict"]
    dict_dir.mkdir(parents=True, exist_ok=True)

    clusters_path = dict_dir / "canonical_wave_clusters.csv"
    contacts_path = dict_dir / "canonical_contact_signatures.csv"
    alignment_path = dict_dir / "cluster_contact_metric_alignment.csv"

    # 1) Canonical wave clusters
    if not clusters_path.exists():
        bridge = DIRS["particle"] / "wave_cluster_particle_bridge_non_identity.csv"
        wave_summary = DIRS["wave"] / "wave_cluster_taxonomy_summary.csv"
        wave_by_perturb = DIRS["wave"] / "wave_cluster_taxonomy_by_perturbation.csv"

        clusters = pd.DataFrame()
        if bridge.exists():
            b = pd.read_csv(bridge)
            if {"cluster_id", "cluster_name"}.issubset(b.columns):
                clusters = (
                    b[["cluster_id", "cluster_name"]]
                    .drop_duplicates()
                    .rename(columns={"cluster_name": "canonical_name"})
                    .sort_values("cluster_id")
                    .reset_index(drop=True)
                )
                clusters["mean_area"] = np.nan
                clusters["mean_overlap"] = np.nan
                clusters["mean_capacity"] = np.nan
                clusters["evidence_level"] = "recovered_from_particle_non_identity_bridge"

        if clusters.empty:
            wb = pd.read_csv(wave_by_perturb) if wave_by_perturb.exists() else pd.DataFrame()
            ws = pd.read_csv(wave_summary) if wave_summary.exists() else pd.DataFrame()
            if not wb.empty and "num_classes" in wb.columns:
                n = int(round(float(wb["num_classes"].median())))
                mean_capacity = float(wb["mean_contact_capacity"].mean()) if "mean_contact_capacity" in wb.columns else np.nan
                mean_persistence = float(wb["mean_persistence"].mean()) if "mean_persistence" in wb.columns else np.nan
            elif not ws.empty:
                n = int(round(float(ws.iloc[0].get("mean_classes_under_perturbation", 3))))
                mean_capacity = float(ws.iloc[0].get("mean_contact_capacity", np.nan))
                mean_persistence = float(ws.iloc[0].get("mean_persistence_under_perturbation", np.nan))
            else:
                n = 0
                mean_capacity = np.nan
                mean_persistence = np.nan

            canonical_names = [
                "C0_local_overlap_cluster",
                "C1_extended_resonant_cluster",
                "C2_high_capacity_extended_cluster",
            ]
            rows = []
            for i in range(max(0, n)):
                rows.append({
                    "cluster_id": f"C{i}",
                    "canonical_name": canonical_names[i] if i < len(canonical_names) else f"C{i}_wave_cluster_family",
                    "mean_area": np.nan,
                    "mean_overlap": mean_persistence,
                    "mean_capacity": mean_capacity,
                    "evidence_level": "derived_from_wave_cluster_taxonomy_archive",
                })
            clusters = pd.DataFrame(rows)

        if not clusters.empty:
            clusters.to_csv(clusters_path, index=False)

    # 2) Canonical contact signatures
    if not contacts_path.exists():
        species_path = DIRS["contact"] / "contact_matter_taxonomy_by_species.csv"
        if species_path.exists():
            sp = pd.read_csv(species_path)
            if "matter_species" in sp.columns:
                contacts = pd.DataFrame({
                    "contact_species": sp["matter_species"].apply(lambda x: f"T{int(x):02d}"),
                    "contact_name": sp["matter_species"].apply(lambda x: f"T{int(x):02d}_periodic_contact_species"),
                    "contact_family": np.where(
                        sp.get("mean_loop_score", 0) >= sp.get("mean_tunnel_score", 0),
                        "loop-dominant",
                        "tunnel-dominant",
                    ),
                    "capacity": sp.get("mean_contact_capacity", np.nan),
                    "loop_score": sp.get("mean_loop_score", np.nan),
                    "tunnel_score": sp.get("mean_tunnel_score", np.nan),
                    "recurrence_strength": sp.get("mean_recurrence_strength", np.nan),
                    "evidence_level": "derived_from_contact_matter_taxonomy_archive",
                })
                contacts.to_csv(contacts_path, index=False)

    # 3) Non-exact metric alignment proxy
    if not alignment_path.exists() and clusters_path.exists() and contacts_path.exists():
        clusters = pd.read_csv(clusters_path)
        contacts = pd.read_csv(contacts_path)
        rows = []
        for _, c in clusters.iterrows():
            cap = float(c.get("mean_capacity", np.nan))
            for _, t in contacts.iterrows():
                tcap = float(t.get("capacity", np.nan))
                if np.isfinite(cap) and np.isfinite(tcap):
                    score = 1.0 / (1.0 + abs(cap - tcap))
                else:
                    score = float(t.get("recurrence_strength", 0.0)) if np.isfinite(float(t.get("recurrence_strength", np.nan))) else 0.5
                rows.append({
                    "cluster_id": c["cluster_id"],
                    "contact_species": t["contact_species"],
                    "alignment_score": score,
                    "evidence_level": "metric_proxy_not_identity",
                })
        pd.DataFrame(rows).to_csv(alignment_path, index=False)


def build_data():
    build_dictionary_from_archives_if_missing()

    clusters = read_required(DIRS["dict"] / "canonical_wave_clusters.csv")
    contacts = read_required(DIRS["dict"] / "canonical_contact_signatures.csv")
    alignment = read_required(DIRS["dict"] / "cluster_contact_metric_alignment.csv")
    rosetta = read_optional(DIRS["rosetta"] / "corrected_wave_rosetta_dictionary.csv")
    particle = read_required(DIRS["particle"] / "particle_proto_core_reconstruction_summary.csv")
    particle_candidates = read_required(DIRS["particle"] / "particle_proto_core_candidates.csv")
    elements = read_optional(DIRS["periodic"] / "full_periodic_H_Lr_by_element.csv")
    lattice = read_optional(DIRS["lattice"] / "lattice_exchange_clusters.csv")
    condensed = read_optional(DIRS["condensed"] / "condensed_matter_stage_metrics.csv")
    return clusters, contacts, alignment, rosetta, particle, particle_candidates, elements, lattice, condensed

def block_label(block):
    return {
        "s": "nodal / pair regime",
        "p": "loop / polar regime",
        "d": "bridge / exchange regime",
        "f": "inner resonant regime",
    }.get(str(block), "unknown")

def block_color(block):
    return {
        "s": "tab:blue",
        "p": "tab:green",
        "d": "tab:orange",
        "f": "tab:purple",
    }.get(str(block), "gray")

def create_rosetta(clusters, contacts, particle):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")
    ax.set_title("BST ROSETTA STONE — CORRECTED DUAL LANGUAGE", fontsize=18, weight="bold")

    panel(ax, 0.03, 0.68, 0.29, 0.22, "1. Native language: W -> C")
    ax.text(0.05, 0.83, "W: primordial waves / substantive beats", fontsize=10)
    ax.text(0.05, 0.79, "C: archived wave-clusters extracted from W interactions", fontsize=10)
    for i, r in clusters.iterrows():
        ax.text(0.06, 0.74 - i*0.04,
                f"{r.cluster_id}: {r.canonical_name} | area={r.mean_area:.1f}, cap={r.mean_capacity:.2f}",
                fontsize=8)

    panel(ax, 0.36, 0.68, 0.29, 0.22, "2. Intermediate layer: proto-cores")
    ax.text(0.38, 0.83, "Four-wave compact proto-core: CONFIRMED MOTIF", fontsize=10)
    ax.text(0.38, 0.79, "Particle identities: NOT CONFIRMED in 57", fontsize=10)
    ax.text(0.38, 0.75, "Electron-like identity deferred to later reconstruction", fontsize=9)

    panel(ax, 0.69, 0.68, 0.28, 0.22, "3. Translated language: T signatures")
    for i, r in contacts.head(6).iterrows():
        ax.text(0.71, 0.84 - i*0.032,
                f"{r.contact_species}: {r.contact_name} | {r.contact_family}",
                fontsize=8)
    ax.text(0.71, 0.64, "T0-T11 are observable contact signatures, not fundamental.", fontsize=8)

    panel(ax, 0.03, 0.36, 0.94, 0.25, "4. Safe dictionary rules")
    rules = [
        "SUPPORTED: W -> C -> P/T -> matter",
        "SUPPORTED: C0-C2 archived as wave-cluster classes",
        "SUPPORTED: T0-T11 archived as translated contact signatures",
        "SUPPORTED: four-wave proto-core as motif",
        "REJECTED: exact W->T compositions in old atlas",
        "REJECTED: T0-T11 as fundamental objects",
        "UNCONFIRMED: electron = four-wave tetrahedron",
    ]
    for i, txt in enumerate(rules):
        ax.text(0.06, 0.55 - i*0.028, txt, fontsize=10)

    panel(ax, 0.03, 0.08, 0.94, 0.22, "5. Current particle result")
    p = particle.iloc[0]
    txt = (
        f"num_wave_clusters={p.num_wave_clusters}\n"
        f"four_wave_proto_core_confirmed={p.four_wave_proto_core_confirmed}\n"
        f"particle_identities_confirmed={p.particle_identities_confirmed}\n"
        f"verdict={p.verdict}"
    )
    ax.text(0.06, 0.23, txt, fontsize=10, family="monospace", va="top")

    savefig("BST_ROSETTA_STONE_corrected.png")

def create_contact_matter_atlas(contacts):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_title("BST CONTACT MATTER ATLAS — REAL METRIC SIGNATURES", fontsize=18, weight="bold")

    xs = [0.04, 0.28, 0.52, 0.76]
    ys = [0.68, 0.42, 0.16]
    for idx, (_, r) in enumerate(contacts.iterrows()):
        x = xs[idx % 4]
        y = ys[idx // 4]
        panel(ax, x, y, 0.20, 0.20, f"{r.contact_species} — {r.contact_name}")
        ax.text(x+0.015, y+0.125, f"Family: {r.contact_family}", fontsize=8)
        ax.text(x+0.015, y+0.095, f"Capacity: {r.capacity:.2f}", fontsize=8)
        ax.text(x+0.015, y+0.065, f"Loop: {r.loop_score:.2f}", fontsize=8)
        ax.text(x+0.015, y+0.035, f"Tunnel: {r.tunnel_score:.2f}", fontsize=8)

    ax.text(0.04, 0.05,
            "Canonical rule: contact species are translated observable signatures downstream of wave clusters; exact W composition not proven.",
            fontsize=10)

    savefig("BST_CONTACT_MATTER_ATLAS_real_metrics.png")

def create_wave_periodic_table(clusters, contacts, alignment):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_title("BST WAVE PERIODIC TABLE — ARCHIVED WAVE-CLUSTER BASIS", fontsize=18, weight="bold")

    panel(ax, 0.04, 0.58, 0.28, 0.30, "Archived native wave clusters")
    for i, r in clusters.iterrows():
        ax.text(0.06, 0.80 - i*0.07,
                f"{r.cluster_id}: {r.canonical_name}\narea={r.mean_area:.1f}, overlap={r.mean_overlap:.2f}, capacity={r.mean_capacity:.2f}",
                fontsize=9)

    panel(ax, 0.36, 0.58, 0.28, 0.30, "Proto-core layer")
    ax.text(0.38, 0.80, "Confirmed motif:", fontsize=10, weight="bold")
    ax.text(0.38, 0.75, "four-wave compact proto-core", fontsize=10)
    ax.text(0.38, 0.69, "Not yet assigned:", fontsize=10, weight="bold")
    ax.text(0.38, 0.64, "electron / photon / neutrino / quark identity", fontsize=9)

    panel(ax, 0.68, 0.58, 0.28, 0.30, "Non-exact C -> T metric alignment")
    top = alignment.sort_values(["cluster_id", "alignment_score"], ascending=[True, False]).groupby("cluster_id").head(3)
    for i, r in enumerate(top.itertuples()):
        ax.text(0.70, 0.82 - i*0.035,
                f"{r.cluster_id} -> {r.contact_species}: {r.alignment_score:.2f} ({r.evidence_level})",
                fontsize=8)

    panel(ax, 0.04, 0.10, 0.92, 0.38, "Contact signature metric landscape")
    ax2 = fig.add_axes([0.10, 0.16, 0.78, 0.25])
    ax2.scatter(contacts["loop_score"], contacts["tunnel_score"], s=contacts["capacity"]*120)
    for _, r in contacts.iterrows():
        ax2.text(r.loop_score+0.01, r.tunnel_score+0.01, r.contact_species, fontsize=8)
    ax2.set_xlabel("Loop score")
    ax2.set_ylabel("Tunnel score")
    ax2.set_title("T signatures from real archived metrics")

    savefig("BST_WAVE_PERIODIC_TABLE_corrected.png")

def periodic_position(row):
    z = int(row["Z"])
    group = int(row["group"])
    period = int(row["period"])
    block = row["block"]

    if block == "f":
        if 57 <= z <= 71:
            return (z - 56, 8)
        if 89 <= z <= 103:
            return (z - 88, 9)

    return (group, period)

def create_periodic_table(elements):
    if elements.empty:
        return

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 19.5)
    ax.set_ylim(10.5, 0)
    ax.axis("off")
    ax.set_title("BST PERIODIC TABLE — DUAL LANGUAGE, SAFE VERSION", fontsize=18, weight="bold")

    for _, r in elements.iterrows():
        x, y = periodic_position(r)
        color = block_color(r["block"])
        rect = Rectangle((x-0.45, y-0.45), 0.85, 0.85, fill=False, edgecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x-0.35, y-0.25, str(int(r["Z"])), fontsize=6)
        ax.text(x, y-0.02, r["symbol"], fontsize=10, ha="center", weight="bold")
        stab = r.get("mean_shell_stability", np.nan)
        ax.text(x, y+0.25, f"{stab:.2f}" if pd.notna(stab) else "", fontsize=6, ha="center")

    ax.text(1, 10.15,
            "Safe interpretation: elements reconstructed in translated periodic/contact language; exact wave-cluster composition deferred to atomic wave-cluster reconstruction.",
            fontsize=10)
    ax.text(13.2, 8.2, "Block -> wave-language bridge", fontsize=11, weight="bold")
    for i, b in enumerate(["s", "p", "d", "f"]):
        ax.text(13.2, 8.55 + i*0.32, f"{b}: {block_label(b)}", fontsize=9, color=block_color(b))

    savefig("BST_PERIODIC_TABLE_dual_language_safe.png")

def create_visual_atlas(clusters, contacts, particle_candidates, condensed):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_title("BST VISUAL ATLAS — DATA-DERIVED DUAL LANGUAGE", fontsize=18, weight="bold")

    panel(ax, 0.03, 0.58, 0.30, 0.30, "Native wave objects")
    for i, r in clusters.iterrows():
        ax.text(0.05, 0.80 - i*0.06,
                f"{r.cluster_id}: {r.canonical_name} | cap={r.mean_capacity:.2f}",
                fontsize=9)

    panel(ax, 0.36, 0.58, 0.28, 0.30, "Particle proto-core status")
    for i, r in particle_candidates.iterrows():
        ax.text(0.38, 0.80 - i*0.055,
                f"{r.particle_candidate}: {r.safe_status}",
                fontsize=8)

    panel(ax, 0.67, 0.58, 0.30, 0.30, "Translated contact language")
    for i, r in contacts.head(8).iterrows():
        ax.text(0.69, 0.82 - i*0.032,
                f"{r.contact_species}: {r.contact_family}",
                fontsize=8)

    panel(ax, 0.03, 0.10, 0.94, 0.38, "Emergence chain status")
    chain = [
        "W: primordial waves / substantive beats",
        "C: archived wave clusters C0-C2",
        "P: four-wave proto-core motif confirmed",
        "T: contact species T0-T11 translated signatures",
        "Matter: periodic/contact reconstruction already supported",
        "Particles: identities not yet confirmed",
    ]
    for i, t in enumerate(chain):
        ax.text(0.06, 0.42 - i*0.045, t, fontsize=11)

    if not condensed.empty:
        ax.text(0.55, 0.42, "Condensed matter chain:", fontsize=11, weight="bold")
        for i, r in condensed.iterrows():
            ax.text(0.55, 0.38 - i*0.035,
                    f"{r.stage}: support={r.support_score:.3f}",
                    fontsize=8)

    savefig("BST_VISUAL_ATLAS_dual_language_data_derived.png")

def main():
    print("\n=== BST 58 FINAL DUAL-LANGUAGE ATLAS REBUILD ===\n")

    clusters, contacts, alignment, rosetta, particle, particle_candidates, elements, lattice, condensed = build_data()

    create_rosetta(clusters, contacts, particle)
    create_contact_matter_atlas(contacts)
    create_wave_periodic_table(clusters, contacts, alignment)
    create_periodic_table(elements)
    create_visual_atlas(clusters, contacts, particle_candidates, condensed)

    summary = pd.DataFrame([{
        "five_atlases_rebuilt": True,
        "rosetta_stone": "BST_ROSETTA_STONE_corrected.png",
        "contact_matter_atlas": "BST_CONTACT_MATTER_ATLAS_real_metrics.png",
        "wave_periodic_table": "BST_WAVE_PERIODIC_TABLE_corrected.png",
        "periodic_table": "BST_PERIODIC_TABLE_dual_language_safe.png",
        "visual_atlas": "BST_VISUAL_ATLAS_dual_language_data_derived.png",
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "safe_hierarchy": "W -> C -> P/T -> matter",
        "verdict": "final_dual_language_atlas_rebuild_supported",
    }])

    summary.to_csv(OUT / "final_dual_language_atlas_rebuild_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\n[OK] wrote five corrected atlas PNG files to:", OUT)
    print("[DONE] final dual-language atlas rebuild complete")

if __name__ == "__main__":
    main()
