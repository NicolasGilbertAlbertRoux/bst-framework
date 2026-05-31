#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BST / OMM-SBT — Ring / Loop Emergence Scanner
=============================================

Purpose
-------
This is a conservative, data-grounded scanner.

It does NOT generate decorative figures.
It does NOT assume that rings exist.
It does NOT claim that BST derives string theory or M-theory.

It asks:

    Do the already-produced OMM/SBT/Zeta outputs contain
    statistically nontrivial loop-like / annular / closed motifs?

The script searches the existing project outputs for:
    - branching_zone_projection.csv
    - branching_zone_centers.csv
    - local_36_1_microloop_events.csv
    - zeta_intersector_transition_candidates.csv

Then it performs three tests:

1. Annular geometry test
   On high-branching projected points:
      - radial concentration around candidate centers,
      - central depletion,
      - angular coverage,
      - null comparison against randomized point clouds.

2. Transition loop test
   On local microloop transition logs:
      - repeated mode transition motifs,
      - return / closure indicators,
      - compression / energy / closure consistency.

3. Intersector bridge test
   On intersector transition candidates:
      - whether bridge candidates connect coherent clusters
        with low transition distance and high closure.

Interpretation
--------------
Positive result:
    "Annular / loop-like motifs deserve deeper formal study."

Negative result:
    "The ring hypothesis is not supported by these outputs."

In all cases:
    - no Polyakov or Nambu-Goto action is derived here;
    - no equivalence with string theory or M-theory is claimed;
    - this is a screening test only.

Usage
-----
From a project root:
    python bst_ring_loop_emergence_scanner.py --root ultime-recherche

Or directly from a zip:
    python bst_ring_loop_emergence_scanner.py --zip ultime-recherche.zip

Fast mode:
    python bst_ring_loop_emergence_scanner.py --zip ultime-recherche.zip --fast

Outputs
-------
results/research_final/bst_ring_loop_emergence_scanner/
    summary.md
    annular_geometry_scores.csv
    transition_loop_scores.csv
    intersector_bridge_scores.csv
    annular_geometry_projection.png
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

TARGETS = {
    "projection": "branching_zone_projection.csv",
    "centers": "branching_zone_centers.csv",
    "microloops": "local_36_1_microloop_events.csv",
    "intersectors": "zeta_intersector_transition_candidates.csv",
}

OUTDIR = Path("results/research_final/bst_ring_loop_emergence_scanner")


def log(msg: str) -> None:
    print(msg, flush=True)


def find_file_in_root(root: Path, filename: str) -> Optional[Path]:
    hits = list(root.rglob(filename))
    hits = [h for h in hits if "__MACOSX" not in str(h)]
    return hits[0] if hits else None


def read_csv_from_zip(zip_path: Path, filename: str) -> Optional[pd.DataFrame]:
    with zipfile.ZipFile(zip_path, "r") as z:
        hits = [n for n in z.namelist() if n.endswith(filename) and "__MACOSX" not in n]
        if not hits:
            return None
        # Prefer results/research_final if multiple
        hits.sort(key=lambda n: ("results/research_final" not in n, len(n)))
        with z.open(hits[0]) as f:
            return pd.read_csv(f)


def load_sources(args) -> Dict[str, pd.DataFrame]:
    data = {}

    if args.zip:
        zip_path = Path(args.zip)
        log(f"[LOAD] reading from zip: {zip_path}")
        for key, name in TARGETS.items():
            df = read_csv_from_zip(zip_path, name)
            if df is not None:
                data[key] = df
                log(f"[OK] {name}: {df.shape}")
            else:
                log(f"[MISS] {name}")

    if args.root:
        root = Path(args.root)
        log(f"[LOAD] reading from root: {root}")
        for key, name in TARGETS.items():
            if key in data:
                continue
            p = find_file_in_root(root, name)
            if p:
                data[key] = pd.read_csv(p)
                log(f"[OK] {name}: {data[key].shape} ({p})")
            else:
                log(f"[MISS] {name}")

    if not data:
        raise RuntimeError("No target CSV files found. Provide --root or --zip.")

    return data


# ----------------------------
# Test 1 — annular geometry
# ----------------------------

def angular_coverage(points: np.ndarray, center: np.ndarray, bins: int = 24) -> float:
    if len(points) < 3:
        return 0.0
    ang = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    hist, _ = np.histogram(ang, bins=bins, range=(-np.pi, np.pi))
    return float(np.count_nonzero(hist) / bins)


def annular_score(points: np.ndarray, center: np.ndarray) -> Dict[str, float]:
    """
    A ring-like cloud should have:
      - peak radius away from zero,
      - low central occupancy,
      - moderate radial concentration,
      - broad angular coverage.
    """
    if len(points) < 8:
        return {
            "n": len(points),
            "radius_mean": np.nan,
            "radius_std": np.nan,
            "radial_concentration": 0.0,
            "central_depletion": 0.0,
            "angular_coverage": 0.0,
            "annular_score": 0.0,
        }

    r = np.linalg.norm(points - center[None, :], axis=1)
    r_mean = float(np.mean(r))
    r_std = float(np.std(r))

    radial_concentration = float(1.0 / (1.0 + r_std / (r_mean + 1e-9)))

    central_radius = 0.35 * (r_mean + 1e-9)
    central_count = int(np.sum(r < central_radius))
    central_depletion = float(1.0 - central_count / max(len(points), 1))

    coverage = angular_coverage(points, center, bins=24)

    # Penalize tiny radius: point clusters can fake concentration.
    radius_factor = float(np.tanh(r_mean / 0.35))

    score = radius_factor * radial_concentration * central_depletion * coverage

    return {
        "n": int(len(points)),
        "radius_mean": r_mean,
        "radius_std": r_std,
        "radial_concentration": radial_concentration,
        "central_depletion": central_depletion,
        "angular_coverage": coverage,
        "annular_score": float(score),
    }


def null_annular_scores(points: np.ndarray, center: np.ndarray, n_null: int = 200) -> np.ndarray:
    """
    Null model: uniformly random points inside the same bounding box.
    This does not preserve all structure; it is a conservative first screen.
    """
    if len(points) < 8:
        return np.array([])

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    scores = []
    for _ in range(n_null):
        rand = np.column_stack([
            np.random.uniform(mins[0], maxs[0], len(points)),
            np.random.uniform(mins[1], maxs[1], len(points)),
        ])
        scores.append(annular_score(rand, center)["annular_score"])
    return np.array(scores)


def run_annular_geometry_test(proj: pd.DataFrame, centers: Optional[pd.DataFrame], fast: bool) -> pd.DataFrame:
    log("[TEST 1] Annular geometry test")

    if not {"z1", "z2"}.issubset(proj.columns):
        log("[SKIP] projection CSV lacks z1/z2 columns")
        return pd.DataFrame()

    if "high_branching" in proj.columns:
        pts_df = proj[proj["high_branching"].astype(bool)].copy()
        if len(pts_df) < 8:
            pts_df = proj.copy()
    else:
        # Fallback: top quartile of branching ratio if available
        if "branching_ratio" in proj.columns:
            q = proj["branching_ratio"].quantile(0.75)
            pts_df = proj[proj["branching_ratio"] >= q].copy()
        else:
            pts_df = proj.copy()

    points = pts_df[["z1", "z2"]].to_numpy(float)

    candidate_centers = []
    if centers is not None and {"z1_center", "z2_center"}.issubset(centers.columns):
        for _, row in centers.iterrows():
            candidate_centers.append((float(row["z1_center"]), float(row["z2_center"]), f"cluster_{int(row.get('cluster', len(candidate_centers)))}"))

    # Always include centroid as neutral center
    centroid = points.mean(axis=0)
    candidate_centers.append((float(centroid[0]), float(centroid[1]), "global_centroid"))

    rows = []
    n_null = 50 if fast else 250

    for cx, cy, label in candidate_centers:
        center = np.array([cx, cy])
        scores = annular_score(points, center)
        null = null_annular_scores(points, center, n_null=n_null)
        if len(null):
            p_emp = float((np.sum(null >= scores["annular_score"]) + 1) / (len(null) + 1))
            null_mean = float(np.mean(null))
            null_std = float(np.std(null))
            z = float((scores["annular_score"] - null_mean) / (null_std + 1e-9))
        else:
            p_emp, null_mean, null_std, z = np.nan, np.nan, np.nan, np.nan

        rows.append({
            "center_label": label,
            "center_z1": cx,
            "center_z2": cy,
            **scores,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_z_score": z,
            "empirical_p_value": p_emp,
        })

    return pd.DataFrame(rows).sort_values("annular_score", ascending=False)


def plot_annular_projection(proj: pd.DataFrame, scores: pd.DataFrame, outdir: Path) -> None:
    if scores.empty or not {"z1", "z2"}.issubset(proj.columns):
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    if "branching_ratio" in proj.columns:
        sc = ax.scatter(proj["z1"], proj["z2"], c=proj["branching_ratio"], s=14, alpha=0.75)
        fig.colorbar(sc, ax=ax, label="branching_ratio")
    else:
        ax.scatter(proj["z1"], proj["z2"], s=14, alpha=0.75)

    if "high_branching" in proj.columns:
        hb = proj[proj["high_branching"].astype(bool)]
        ax.scatter(hb["z1"], hb["z2"], s=22, facecolors="none", edgecolors="black", label="high_branching")

    best = scores.iloc[0]
    cx, cy = best["center_z1"], best["center_z2"]
    r = best["radius_mean"]
    circ = plt.Circle((cx, cy), r, fill=False, linewidth=2, linestyle="--", label="best annular radius")
    ax.add_patch(circ)
    ax.scatter([cx], [cy], marker="x", s=80, label="tested center")

    ax.set_title("Annular geometry scan on existing branching-zone projection")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "annular_geometry_projection.png", dpi=180)
    plt.close(fig)


# ----------------------------
# Test 2 — transition loops
# ----------------------------

def run_transition_loop_test(events: pd.DataFrame) -> pd.DataFrame:
    log("[TEST 2] Transition loop test")

    required = {"from_mode", "to_mode"}
    if not required.issubset(events.columns):
        log("[SKIP] microloop CSV lacks from_mode/to_mode columns")
        return pd.DataFrame()

    rows = []
    grouped = events.groupby(["from_mode", "to_mode"], dropna=False)
    total = len(events)

    for (fm, tm), g in grouped:
        reverse = events[(events["from_mode"] == tm) & (events["to_mode"] == fm)]
        recurrence = len(g) / max(total, 1)
        reverse_support = len(reverse) / max(total, 1)

        energy_ok = np.nan
        compression_ok = np.nan
        closure_ok = np.nan

        if {"energy_delta"}.issubset(g.columns):
            energy_ok = float(np.mean(g["energy_delta"] <= 0))
        if {"compression_delta"}.issubset(g.columns):
            compression_ok = float(np.mean(np.abs(g["compression_delta"]) < np.nanmedian(np.abs(events["compression_delta"])) + 1e-12))
        if {"from_closure", "to_closure"}.issubset(g.columns):
            closure_ok = float(np.mean(g["to_closure"] >= 0.95 * g["from_closure"]))

        loop_score = recurrence * (1.0 + reverse_support)
        for v in [energy_ok, compression_ok, closure_ok]:
            if not np.isnan(v):
                loop_score *= (0.5 + 0.5 * v)

        rows.append({
            "from_mode": fm,
            "to_mode": tm,
            "count": int(len(g)),
            "recurrence": float(recurrence),
            "reverse_support": float(reverse_support),
            "energy_nonincreasing_fraction": energy_ok,
            "compression_stable_fraction": compression_ok,
            "closure_preserved_fraction": closure_ok,
            "transition_loop_score": float(loop_score),
        })

    return pd.DataFrame(rows).sort_values("transition_loop_score", ascending=False)


# ----------------------------
# Test 3 — intersector bridge
# ----------------------------

def run_intersector_bridge_test(cand: pd.DataFrame) -> pd.DataFrame:
    log("[TEST 3] Intersector bridge test")

    if cand.empty:
        return pd.DataFrame()

    df = cand.copy()

    # Normalize useful columns when present
    dist = df["transition_distance"] if "transition_distance" in df.columns else pd.Series(np.ones(len(df)))
    score = df["transition_score"] if "transition_score" in df.columns else pd.Series(np.ones(len(df)))
    closure_a = df["closure_a"] if "closure_a" in df.columns else pd.Series(np.ones(len(df)))
    closure_b = df["closure_b"] if "closure_b" in df.columns else pd.Series(np.ones(len(df)))

    df["mean_closure"] = (closure_a + closure_b) / 2
    df["distance_factor"] = 1.0 / (1.0 + dist)
    df["bridge_score"] = df["distance_factor"] * score * df["mean_closure"]

    return df.sort_values("bridge_score", ascending=False)


# ----------------------------
# Summary
# ----------------------------

def write_summary(outdir: Path, ann: pd.DataFrame, loops: pd.DataFrame, bridges: pd.DataFrame) -> None:
    lines = []
    lines.append("# BST Ring / Loop Emergence Scanner\n")
    lines.append("This is a data-grounded screening test using existing OMM/SBT/Zeta outputs.\n")
    lines.append("It does not claim equivalence with string theory or M-theory.\n")

    lines.append("## Main interpretation\n")
    if not ann.empty:
        best = ann.iloc[0]
        lines.append(f"- Best annular geometry score: `{best['annular_score']:.4f}` at `{best['center_label']}`.")
        lines.append(f"- Empirical null p-value: `{best['empirical_p_value']:.4f}`; null z-score: `{best['null_z_score']:.2f}`.")
        if best["empirical_p_value"] < 0.05 and best["annular_score"] > best["null_mean"]:
            lines.append("- Preliminary result: annular geometry is stronger than this simple null model.")
        else:
            lines.append("- Preliminary result: annular geometry is not clearly stronger than this simple null model.")
    else:
        lines.append("- Annular geometry test was not available.")

    if not loops.empty:
        b = loops.iloc[0]
        lines.append(f"- Strongest transition motif: `{b['from_mode']} -> {b['to_mode']}` with score `{b['transition_loop_score']:.4f}`.")
    else:
        lines.append("- Transition loop test was not available.")

    if not bridges.empty:
        b = bridges.iloc[0]
        lines.append(f"- Strongest intersector bridge score: `{b['bridge_score']:.4f}`.")
    else:
        lines.append("- Intersector bridge test was not available.")

    lines.append("\n## String-like correspondence status\n")
    lines.append("Permitted wording if positive: annular / loop-like stabilization motifs are present in the existing reconstruction outputs and may justify deeper comparison with closed-string-like extended objects.")
    lines.append("Forbidden wording: BST derives string theory, M-theory, Polyakov action, or Nambu-Goto action.")
    lines.append("Next mathematical test if results are positive: extract closed contours from simulated fields, parameterize them as worldline/worldsheet candidates, and test whether an area/minimal-surface functional approximates their evolution.\n")

    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="Project root containing results.")
    parser.add_argument("--zip", type=str, default=None, help="Zip archive containing project files.")
    parser.add_argument("--fast", action="store_true", help="Use fewer null samples.")
    args = parser.parse_args()

    if not args.root and not args.zip:

        candidates = []

        for base in [Path("."), Path(".."), Path("research_final")]:

            candidates.extend(base.glob("ultime-recherche*.zip"))

            candidates.extend(base.glob("*.zip"))

        candidates = [p for p in candidates if p.exists()]

        if Path("ultime-recherche").exists():

            args.root = "ultime-recherche"

        elif candidates:

            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            args.zip = str(candidates[0])

            print(f"[AUTO] using zip: {args.zip}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    log("=== BST RING / LOOP EMERGENCE SCANNER ===")
    log(f"[OUT] {OUTDIR}")

    data = load_sources(args)

    ann = pd.DataFrame()
    if "projection" in data:
        ann = run_annular_geometry_test(data["projection"], data.get("centers"), args.fast)
        ann.to_csv(OUTDIR / "annular_geometry_scores.csv", index=False)
        plot_annular_projection(data["projection"], ann, OUTDIR)
        log(f"[WRITE] annular_geometry_scores.csv ({len(ann)} rows)")

    loops = pd.DataFrame()
    if "microloops" in data:
        loops = run_transition_loop_test(data["microloops"])
        loops.to_csv(OUTDIR / "transition_loop_scores.csv", index=False)
        log(f"[WRITE] transition_loop_scores.csv ({len(loops)} rows)")

    bridges = pd.DataFrame()
    if "intersectors" in data:
        bridges = run_intersector_bridge_test(data["intersectors"])
        bridges.to_csv(OUTDIR / "intersector_bridge_scores.csv", index=False)
        log(f"[WRITE] intersector_bridge_scores.csv ({len(bridges)} rows)")

    write_summary(OUTDIR, ann, loops, bridges)
    log("[WRITE] summary.md")
    log("[DONE]")


if __name__ == "__main__":
    main()
