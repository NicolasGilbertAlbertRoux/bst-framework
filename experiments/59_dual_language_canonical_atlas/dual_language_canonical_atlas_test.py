#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/research_final"
OUT = R / "dual_language_canonical_atlas"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "wave_summary": R / "wave_cluster_taxonomy_test/wave_cluster_taxonomy_summary.csv",
    "wave_by_perturbation": R / "wave_cluster_taxonomy_test/wave_cluster_taxonomy_by_perturbation.csv",
    "contact_summary": R / "contact_matter_taxonomy_test/contact_matter_taxonomy_summary.csv",
    "contact_species": R / "contact_matter_taxonomy_test/contact_matter_taxonomy_by_species.csv",
    "particle_summary": R / "particle_proto_core_reconstruction_test/particle_proto_core_reconstruction_summary.csv",
    "particle_candidates": R / "particle_proto_core_reconstruction_test/particle_proto_core_candidates.csv",
    "elements": R / "full_periodic_reconstruction_H_Lr_test/full_periodic_H_Lr_by_element.csv",
    "condensed": R / "full_condensed_matter_milestone_test/condensed_matter_stage_metrics.csv",
}

CONTACT_NAMES = {
    0: "Null / isolated contact",
    1: "Simple bridge",
    2: "Simple loop",
    3: "Simple tunnel",
    4: "Double bridge",
    5: "Double loop",
    6: "Double tunnel",
    7: "Triple bridge",
    8: "Triple loop",
    9: "Triple tunnel",
    10: "Resonant node",
    11: "Quasi-node / transient",
}

def read(path, required=True):
    if path.exists():
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.DataFrame()

def wrap(s, n=34):
    return "\n".join(textwrap.wrap(str(s), n))

def txt(ax, x, y, s, size=8, width=40, **kw):
    ax.text(x, y, wrap(s, width), fontsize=size, va=kw.pop("va", "top"), **kw)

def panel(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.0))
    ax.text(x + 0.01, y + h - 0.025, wrap(title, 36), fontsize=9, weight="bold", va="top")

def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)

def block_color(b):
    return {"s": "tab:blue", "p": "tab:green", "d": "tab:orange", "f": "tab:purple"}.get(str(b), "gray")

def block_wave_label(b):
    return {
        "s": "nodal / radial shell regime",
        "p": "polar / loop-sensitive regime",
        "d": "exchange / bridge-sensitive regime",
        "f": "inner resonant cluster regime",
    }.get(str(b), "unknown wave-shell regime")

def load():
    return {k: read(v, required=True) for k, v in FILES.items()}

def draw_contact(ax, x, y, species, scale=0.035):
    s = int(species)
    # --------------------------------------------------
    # Nodes
    # --------------------------------------------------
    if s in [0, 10, 11]:
        ax.add_patch(
            Circle((x, y), scale, fill=False, lw=1.1)
        )
    # --------------------------------------------------
    # Bridges
    # --------------------------------------------------
    elif s == 1:  # Simple bridge
        ax.add_patch(Circle((x-scale, y), scale*0.45, fill=False, lw=1))
        ax.add_patch(Circle((x+scale, y), scale*0.45, fill=False, lw=1))
        ax.plot([x-scale*0.6, x+scale*0.6], [y, y], lw=1)
    elif s == 4:  # Double bridge
        ax.add_patch(Circle((x-scale, y), scale*0.45, fill=False, lw=1))
        ax.add_patch(Circle((x+scale, y), scale*0.45, fill=False, lw=1))
        ax.plot(
            [x-scale*0.6, x+scale*0.6],
            [y+scale*0.12, y+scale*0.12],
            lw=1
        )
        ax.plot(
            [x-scale*0.6, x+scale*0.6],
            [y-scale*0.12, y-scale*0.12],
            lw=1
        )
    elif s == 7:  # Triple bridge
        pts = np.array([
            [x, y+scale*0.8],
            [x-scale*0.9, y-scale*0.6],
            [x+scale*0.9, y-scale*0.6],
        ])
        ax.plot(
            [pts[0,0], pts[1,0]],
            [pts[0,1], pts[1,1]],
            lw=1
        )
        ax.plot(
            [pts[1,0], pts[2,0]],
            [pts[1,1], pts[2,1]],
            lw=1
        )
        ax.plot(
            [pts[2,0], pts[0,0]],
            [pts[2,1], pts[0,1]],
            lw=1
        )
    # --------------------------------------------------
    # Loops
    # --------------------------------------------------
    elif s == 2:  # Simple loop
        ax.add_patch(
            Circle(
                (x, y),
                scale*1.1,
                fill=False,
                lw=1.1
            )
        )
    elif s == 5:  # Double loop
        ax.add_patch(
            Circle(
                (x-scale*0.55, y),
                scale*0.75,
                fill=False,
                lw=1.1
            )
        )
        ax.add_patch(
            Circle(
                (x+scale*0.55, y),
                scale*0.75,
                fill=False,
                lw=1.1
            )
        )
    elif s == 8:  # Triple loop
        ax.add_patch(
            Circle(
                (x-scale*0.55, y-scale*0.35),
                scale*0.65,
                fill=False,
                lw=1.1
            )
        )
        ax.add_patch(
            Circle(
                (x+scale*0.55, y-scale*0.35),
                scale*0.65,
                fill=False,
                lw=1.1
            )
        )
        ax.add_patch(
            Circle(
                (x, y+scale*0.45),
                scale*0.65,
                fill=False,
                lw=1.1
            )
        )
    # --------------------------------------------------
    # Tunnels
    # --------------------------------------------------
    elif s == 3:  # Simple tunnel
        ax.add_patch(
            Circle(
                (x, y),
                scale*1.15,
                fill=False,
                lw=1.1
            )
        )
        ax.add_patch(
            Circle(
                (x, y),
                scale*0.45,
                fill=False,
                lw=1.0
            )
        )
    elif s == 6:  # Double tunnel
        for dx in (-scale*0.6, scale*0.6):
            ax.add_patch(
                Circle(
                    (x+dx, y),
                    scale*0.7,
                    fill=False,
                    lw=1.1
                )
            )
            ax.add_patch(
                Circle(
                    (x+dx, y),
                    scale*0.28,
                    fill=False,
                    lw=1.0
                )
            )
    elif s == 9:  # Triple tunnel
        positions = [
            (-scale*0.55, -scale*0.30),
            ( scale*0.55, -scale*0.30),
            (0, scale*0.45),
        ]
        for dx, dy in positions:
            ax.add_patch(
                Circle(
                    (x+dx, y+dy),
                    scale*0.55,
                    fill=False,
                    lw=1.1
                )
            )
            ax.add_patch(
                Circle(
                    (x+dx, y+dy),
                    scale*0.22,
                    fill=False,
                    lw=1.0
                )
            )
    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------
    else:
        ax.plot(
            [x-scale, x, x+scale, x-scale],
            [y-scale, y+scale, y-scale, y-scale],
            lw=1
        )

def draw_wave(ax, x, y, kind=0, scale=0.055):
    if kind % 3 == 0:
        for r in [0.35, 0.65, 1.0]:
            ax.add_patch(Circle((x, y), scale*r, fill=False, lw=.9))
    elif kind % 3 == 1:
        xs = np.linspace(x-scale, x+scale, 80)
        ys = y + scale*.25*np.sin(np.linspace(0, 4*np.pi, 80))
        ax.plot(xs, ys, lw=1)
    else:
        theta = np.linspace(0, 2*np.pi, 120)
        ax.plot(x + scale*np.cos(theta), y + scale*.55*np.sin(theta), lw=1)

def contact_rosetta(d):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title("BST ROSETTA — CONTACT LANGUAGE", fontsize=18, weight="bold")
    panel(ax, .03, .64, .30, .26, "Translated observable language")
    txt(ax, .05, .82, "T0–T11 are observable contact signatures. They are not the fundamental wave anastomos.", 10)
    panel(ax, .36, .64, .28, .26, "Safe hierarchy")
    txt(ax, .38, .82, "W → C → P/T → matter", 14)
    txt(ax, .38, .74, "Contact language describes what the material observer sees.", 9)
    panel(ax, .67, .64, .30, .26, "Forbidden shortcuts")
    txt(ax, .69, .82, "No exact W→T recipe. No particle identity assumed. No old Rosetta claim reused.", 9)

    panel(ax, .03, .05, .94, .51, "Canonical contact species")
    ax.text(.05, .505, "Source: contact_matter_taxonomy_by_species.csv", fontsize=7.5, va="top")
    y = .47

    for _, r in d["contact_species"].iterrows():
        s = int(r["matter_species"])
        line = (
            f"T{s:02d} — {CONTACT_NAMES.get(s, 'Contact species')} | "
            f"periodicity={r.mean_periodicity_score:.3f}, recurrence={r.mean_recurrence_strength:.3f}, "
            f"loop={r.mean_loop_score:.3f}, tunnel={r.mean_tunnel_score:.3f}, capacity={r.mean_contact_capacity:.1f}"
        )
        txt(ax, .05, y, line, 7.2, 150)
        y -= .035
    save(fig, "CONTACT_01_ROSETTA.png")

def wave_rosetta(d):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title("BST ROSETTA — WAVE LANGUAGE", fontsize=18, weight="bold")
    ws = d["wave_summary"].iloc[0]
    ps = d["particle_summary"].iloc[0]

    panel(ax, .03, .64, .30, .26, "Native wave language")
    txt(ax, .05, .82, "W are primordial waves / substantive beats. Archived wave-cluster taxonomy confirms stable C-classes under perturbation.", 9)
    txt(ax, .05, .72, f"Mean classes under perturbation: {ws.mean_classes_under_perturbation:.2f}", 9)
    txt(ax, .05, .67, f"Mean persistence: {ws.mean_persistence_under_perturbation:.3f}", 9)

    panel(ax, .36, .64, .28, .26, "Proto-core audit")
    txt(ax, .38, .82, f"Four-wave proto-core motif confirmed: {ps.four_wave_proto_core_confirmed}", 9)
    txt(ax, .38, .76, f"Particle identities confirmed: {ps.particle_identities_confirmed}", 9)

    panel(ax, .67, .64, .30, .26, "Translation rule")
    txt(ax, .69, .82, "Wave language may project into contact language only through archived metrics and hierarchy, not by exact W→T assertion.", 9)

    panel(ax, .03, .08, .94, .48, "Wave-cluster evidence from wave_cluster_taxonomy_by_perturbation.csv")
    y = .51
    for _, r in d["wave_by_perturbation"].iterrows():
        line = (
            f"perturbation={r.perturbation:.3f} | classes={int(r.num_classes)} | "
            f"silhouette={r.silhouette_score:.3f} | persistence={r.mean_persistence:.3f} | "
            f"capacity={r.mean_contact_capacity:.3f}"
        )
        txt(ax, .05, y, line, 7.5, 150)
        y -= .045
    save(fig, "WAVE_01_ROSETTA.png")

def contact_matter_atlas(d):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title("BST CONTACT MATTER ATLAS — CANONICAL", fontsize=18, weight="bold")
    xs, ys = [.04, .28, .52, .76], [.68, .42, .16]
    for idx, (_, r) in enumerate(d["contact_species"].iterrows()):
        s = int(r["matter_species"]); x, y = xs[idx % 4], ys[idx // 4]
        panel(ax, x, y, .20, .20, f"T{s:02d} — {CONTACT_NAMES.get(s)}")
        draw_contact(ax, x+.055, y+.095, s, scale=0.026)
        txt(ax, x+.125, y+.145, f"Period {r.mean_periodicity_score:.2f}", 7, 24)
        txt(ax, x+.125, y+.110, f"Recur {r.mean_recurrence_strength:.2f}", 7, 24)
        txt(ax, x+.125, y+.075, f"L/T {r.mean_loop_score:.2f}/{r.mean_tunnel_score:.2f}", 7, 24)
        txt(ax, x+.125, y+.040, f"Cap {r.mean_contact_capacity:.1f}", 7, 24)
    save(fig, "CONTACT_02_MATTER_ATLAS.png")

def wave_matter_atlas(d):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title("BST WAVE MATTER ATLAS — CANONICAL", fontsize=18, weight="bold")
    panel(ax, .03, .56, .30, .34, "Wave cluster taxonomy")
    for i, r in d["wave_by_perturbation"].iterrows():
        draw_wave(ax, .07, .80-i*.06, i)
        txt(ax, .13, .82-i*.06, f"Perturb {r.perturbation:.3f}: {int(r.num_classes)} classes, persistence {r.mean_persistence:.2f}", 7.5, 34)

    panel(ax, .36, .56, .28, .34, "Proto-core / particle caution")
    for i, r in d["particle_candidates"].iterrows():
        txt(ax, .38, .82-i*.055, f"{r.particle_candidate}: {r.safe_status}", 7.5, 36)

    panel(ax, .67, .56, .30, .34, "Wave → matter rule")
    txt(ax, .69, .82, "Matter atlas in wave language must describe W-driven cluster persistence, capacity and proto-core motifs. It must not assign exact W recipes to T species.", 9, 42)

    panel(ax, .03, .10, .94, .36, "Canonical statement")
    txt(ax, .05, .40, "This is the native-language counterpart of the contact atlas: the contact forms are observational projections of stable wave-cluster organization. The code intentionally writes no exact W→T line.", 11, 150)
    save(fig, "WAVE_02_MATTER_ATLAS.png")

def periodic_pos(row):
    z, b = int(row.Z), str(row.block)
    if b == "f" and 57 <= z <= 71: return z - 56, 8
    if b == "f" and 89 <= z <= 103: return z - 88, 9
    return int(row.group), int(row.period)

def periodic_table(d, mode):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 19.5); ax.set_ylim(10.6, 0); ax.axis("off")
    title = "BST PERIODIC TABLE — CONTACT LANGUAGE" if mode == "contact" else "BST PERIODIC TABLE — WAVE LANGUAGE"
    ax.set_title(title, fontsize=18, weight="bold")

    for _, r in d["elements"].iterrows():
        x, y = periodic_pos(r)
        c = block_color(r.block)
        ax.add_patch(Rectangle((x-.43, y-.43), .86, .86, fill=False, ec=c, lw=1.0))
        ax.text(x-.34, y-.27, str(int(r.Z)), fontsize=6)
        ax.text(x, y-.02, r.symbol, fontsize=10, ha="center", weight="bold")
        if mode == "contact":
            lab = f"{r.block} | {r.mean_shell_stability:.2f}"
        else:
            lab = f"{r.block}→W | {r.mean_shell_stability:.2f}"
        ax.text(x, y+.23, lab, fontsize=5.5, ha="center")

    if mode == "contact":
        ax.text(1, 10.15, "Contact view: periodicity expressed as observable block/contact/shell organization.", fontsize=10)
    else:
        ax.text(1, 10.15, "Wave view: blocks are translated into wave-shell regimes; exact element-level W-compositions are not asserted.", fontsize=9)
        ax.text(13, 8.15, "Wave-shell legend", fontsize=11, weight="bold")
        for i, b in enumerate(["s", "p", "d", "f"]):
            ax.text(13, 8.55+i*.35, f"{b}: {block_wave_label(b)}", fontsize=8, color=block_color(b))

    save(fig, "CONTACT_03_PERIODIC_TABLE.png" if mode == "contact" else "WAVE_03_PERIODIC_TABLE.png")

def visual_atlas(d, mode):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title(f"BST VISUAL ATLAS — {mode.upper()} LANGUAGE", fontsize=18, weight="bold")

    if mode == "contact":
        panel(ax, .03, .58, .30, .30, "Contact species")
        for i, r in d["contact_species"].head(6).iterrows():
            s = int(r.matter_species)
            txt(ax, .05, .82-i*.045, f"T{s}: {CONTACT_NAMES.get(s)}", 8)

        panel(ax, .36, .58, .28, .30, "Condensed matter chain")
        for i, r in d["condensed"].iterrows():
            txt(ax, .38, .82-i*.045, f"{r.stage}: {r.support_score:.3f}", 7.5)

        panel(ax, .67, .58, .30, .30, "Interpretation")
        txt(ax, .69, .82, "This view is suitable for matter, chemistry and condensed-matter observables.", 9)
    else:
        panel(ax, .03, .58, .30, .30, "Wave clusters")
        for i, r in d["wave_by_perturbation"].iterrows():
            txt(ax, .05, .82-i*.05, f"Perturb={r.perturbation:.3f}: classes={int(r.num_classes)}, persistence={r.mean_persistence:.2f}", 7.5)

        panel(ax, .36, .58, .28, .30, "Particle proto-core status")
        for i, r in d["particle_candidates"].iterrows():
            txt(ax, .38, .82-i*.045, f"{r.particle_candidate}: {r.safe_status}", 7.5)

        panel(ax, .67, .58, .30, .30, "Interpretation")
        txt(ax, .69, .82, "This view is suitable for native BST ontology: W, C, proto-core motifs and wave-shell regimes.", 9)

    panel(ax, .03, .10, .94, .36, "Common rule")
    txt(ax, .05, .40, "Both atlases describe the same pipeline: W → C → P/T → matter. Contact language is observational; wave language is native. No exact W→T shortcut is used.", 11, 150)
    save(fig, f"{mode.upper()}_04_VISUAL_ATLAS.png")

def dictionary(d, mode):
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    ax.set_title(f"BST DICTIONARY — {mode.upper()} LANGUAGE", fontsize=18, weight="bold")

    y = .88
    if mode == "contact":
        for _, r in d["contact_species"].iterrows():
            s = int(r.matter_species)
            line = f"T{s:02d} | {CONTACT_NAMES.get(s)} | periodicity={r.mean_periodicity_score:.3f} | recurrence={r.mean_recurrence_strength:.3f} | loop={r.mean_loop_score:.3f} | tunnel={r.mean_tunnel_score:.3f} | capacity={r.mean_contact_capacity:.1f}"
            txt(ax, .04, y, line, 7.5, 150); y -= .055
    else:
        ws = d["wave_summary"].iloc[0]
        lines = [
            f"Wave taxonomy verdict: {ws.verdict}",
            f"Mean classes under perturbation: {ws.mean_classes_under_perturbation:.3f}",
            f"Mean persistence under perturbation: {ws.mean_persistence_under_perturbation:.3f}",
            f"Mean contact capacity: {ws.mean_contact_capacity:.3f}",
            "Four-wave compact proto-core: confirmed motif only.",
            "Particle identities: not confirmed in 57.",
            "Forbidden: exact W→T mapping.",
            "Allowed: W→C→P/T→matter hierarchy.",
        ]
        for line in lines:
            txt(ax, .04, y, line, 10, 140); y -= .075

    save(fig, f"{mode.upper()}_05_DICTIONARY.png")

def main():
    print("\n=== BST 59 DUAL-LANGUAGE CANONICAL ATLAS — REAL FILES ===\n")
    d = load()

    contact_rosetta(d)
    wave_rosetta(d)
    contact_matter_atlas(d)
    wave_matter_atlas(d)
    periodic_table(d, "contact")
    periodic_table(d, "wave")
    visual_atlas(d, "contact")
    visual_atlas(d, "wave")
    dictionary(d, "contact")
    dictionary(d, "wave")

    summary = pd.DataFrame([{
        "contact_atlases": 5,
        "wave_atlases": 5,
        "total_atlases": 10,
        "exact_W_to_T_used": False,
        "particle_identity_used": False,
        "source_files_real": True,
        "safe_hierarchy": "W -> C -> P/T -> matter",
        "verdict": "canonical_dual_language_bst_atlas_complete",
    }])
    summary.to_csv(OUT / "dual_language_canonical_atlas_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\n[OK] wrote 10 canonical atlas PNG files to: {OUT}")
    print("[DONE] canonical dual-language atlas complete")

if __name__ == "__main__":
    main()