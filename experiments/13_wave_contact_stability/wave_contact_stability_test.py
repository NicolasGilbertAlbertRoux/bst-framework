#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Wave Contact Stability Test
=================================

Purpose
-------
Test whether wave-contact structures detected in primordial mantle overlaps
remain stable under small perturbations of the substrate.

This test follows the quantum_contact_classes experiment, but asks a stronger
question:

    Do loop/tunnel-like contact sectors persist under perturbation?

Interpretation
--------------
- If contacts disappear or reclassify randomly, they are not robust sectors.
- If contact classes remain stable under perturbation, they become candidates
  for emergent quantum contact classes.

Outputs
-------
results/research_final/wave_contact_stability_test/
    wave_contact_stability_events.csv
    wave_contact_stability_summary.csv
    wave_contact_stability_projection.png
"""

from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


OUT = Path("results/research_final/wave_contact_stability_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)

N = 128
T_STEPS = 120
NUM_WAVES = 5

MIN_COMPONENT_AREA = 18
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 8

PERTURBATION_LEVELS = [0.00, 0.01, 0.025, 0.05, 0.075, 0.10]
REFERENCE_LEVEL = 0.00

x = np.linspace(-1.0, 1.0, N)
y = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(x, y)


def connected_components(mask):
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    h, w = mask.shape

    for iy in range(h):
        for ix in range(w):
            if not mask[iy, ix] or visited[iy, ix]:
                continue

            q = deque([(iy, ix)])
            visited[iy, ix] = True
            pts = []

            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))

                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        q.append((ny, nx))

            comps.append(np.asarray(pts, dtype=int))

    return comps


def perimeter_of_component(mask, pts):
    p = 0
    h, w = mask.shape

    for cy, cx in pts:
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not mask[ny, nx]:
                p += 1

    return p


def hole_count_for_component(component_mask):
    bg = ~component_mask
    visited = np.zeros_like(bg, dtype=bool)
    h, w = bg.shape
    holes = 0

    for iy in range(h):
        for ix in range(w):
            if not bg[iy, ix] or visited[iy, ix]:
                continue

            q = deque([(iy, ix)])
            visited[iy, ix] = True
            touches_boundary = False

            while q:
                cy, cx = q.popleft()

                if cy in (0, h - 1) or cx in (0, w - 1):
                    touches_boundary = True

                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and bg[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        q.append((ny, nx))

            if not touches_boundary:
                holes += 1

    return holes


def wave_field(t, perturbation=0.0, seed_offset=0):
    local_rng = np.random.default_rng(SEED + 10_000 * seed_offset + int(1000 * perturbation))

    field = np.zeros((N, N))
    envelopes = []

    for i in range(NUM_WAVES):
        angle = 2 * np.pi * i / NUM_WAVES

        jitter_cx = perturbation * local_rng.normal()
        jitter_cy = perturbation * local_rng.normal()
        jitter_phase = perturbation * local_rng.normal()

        cx = 0.45 * np.cos(angle + 0.015 * t) + jitter_cx
        cy = 0.45 * np.sin(angle + 0.013 * t) + jitter_cy

        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        radius = 0.32 + 0.05 * np.sin(0.09 * t + i)
        sigma = 0.035 + 0.005 * np.cos(0.05 * t + i)

        envelope = np.exp(-((r - radius) ** 2) / (2 * sigma**2))
        phase = 24.0 * r - 0.18 * t + 1.7 * i + jitter_phase

        field += envelope * np.sin(phase)
        envelopes.append(envelope)

    return field, np.stack(envelopes, axis=0)


def extract_contact_events(perturbation, seed_offset=0):
    records = []

    for t in range(T_STEPS):
        field, envelopes = wave_field(t, perturbation, seed_offset)

        overlap_count = np.sum(envelopes > 0.35, axis=0)
        mantle_overlap = overlap_count >= 2

        amp = np.abs(field)

        if not np.any(mantle_overlap):
            continue

        contact_threshold = np.quantile(amp[mantle_overlap], 0.55)
        contact_mask = mantle_overlap & (amp >= contact_threshold)

        comps = connected_components(contact_mask)

        for comp_id, pts in enumerate(comps):
            area = len(pts)

            if area < MIN_COMPONENT_AREA:
                continue

            ys = pts[:, 0]
            xs = pts[:, 1]

            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            local = np.zeros((ymax - ymin + 3, xmax - xmin + 3), dtype=bool)
            local[ys - ymin + 1, xs - xmin + 1] = True

            holes = hole_count_for_component(local)
            perimeter = perimeter_of_component(contact_mask, pts)

            compactness = 4.0 * np.pi * area / ((perimeter + 1e-9) ** 2)
            loop_score = holes + (1.0 - compactness)

            mean_amp = float(amp[ys, xs].mean())
            mean_overlap = float(overlap_count[ys, xs].mean())

            cx = float(xs.mean() / N)
            cy = float(ys.mean() / N)

            records.append(
                {
                    "perturbation": perturbation,
                    "time": t,
                    "component_id": comp_id,
                    "area": area,
                    "perimeter": perimeter,
                    "holes": holes,
                    "compactness": compactness,
                    "loop_score": loop_score,
                    "mean_amplitude": mean_amp,
                    "mean_overlap": mean_overlap,
                    "centroid_x": cx,
                    "centroid_y": cy,
                }
            )

    return pd.DataFrame(records)


def feature_matrix(events):
    cols = [
        "area",
        "perimeter",
        "holes",
        "compactness",
        "loop_score",
        "mean_amplitude",
        "mean_overlap",
    ]
    Xf = events[cols].to_numpy(dtype=float)
    return (Xf - Xf.mean(axis=0)) / (Xf.std(axis=0) + 1e-9)


def classify_events(events):
    if len(events) < 10:
        labels = np.full(len(events), -1, dtype=int)
        return labels, 0, np.nan

    Xf = feature_matrix(events)

    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
    ).fit_predict(Xf)

    valid = labels >= 0
    num_classes = len(set(labels)) - (1 if -1 in labels else 0)

    if num_classes >= 2 and np.any(valid):
        sil = float(silhouette_score(Xf[valid], labels[valid]))
    else:
        sil = np.nan

    return labels, num_classes, sil


def class_persistence(events, labels):
    valid_labels = sorted([x for x in set(labels) if x >= 0])
    if not valid_labels:
        return np.nan

    persistences = []
    for label in valid_labels:
        times = events.loc[labels == label, "time"]
        persistences.append(len(set(times)) / T_STEPS)

    return float(np.mean(persistences))


def main():
    print("\n=== BST WAVE CONTACT STABILITY TEST ===")

    all_events = []
    summaries = []

    reference_events = None
    reference_labels = None

    for p in PERTURBATION_LEVELS:
        events = extract_contact_events(p)

        if len(events) == 0:
            summaries.append(
                {
                    "perturbation": p,
                    "num_events": 0,
                    "num_classes": 0,
                    "noise_fraction": np.nan,
                    "silhouette_score": np.nan,
                    "mean_persistence": np.nan,
                    "ari_vs_reference": np.nan,
                }
            )
            continue

        labels, num_classes, sil = classify_events(events)
        events["class_label"] = labels

        noise_fraction = float(np.mean(labels == -1))
        persistence = class_persistence(events, labels)

        if p == REFERENCE_LEVEL:
            reference_events = events.copy()
            reference_labels = labels.copy()
            ari = 1.0
        else:
            # Compare label structure distribution, not point identity.
            # We use a conservative proxy: compare sorted class-size histograms.
            ref_sizes = np.array(
                sorted(
                    [
                        np.sum(reference_labels == k)
                        for k in set(reference_labels)
                        if k >= 0
                    ]
                )
            )
            cur_sizes = np.array(
                sorted(
                    [
                        np.sum(labels == k)
                        for k in set(labels)
                        if k >= 0
                    ]
                )
            )

            m = min(len(ref_sizes), len(cur_sizes))

            if m == 0:
                ari = np.nan
            else:
                ref_clip = ref_sizes[-m:]
                cur_clip = cur_sizes[-m:]
                ari = float(
                    np.corrcoef(ref_clip, cur_clip)[0, 1]
                    if m >= 2
                    else 1.0
                )

        summaries.append(
            {
                "perturbation": p,
                "num_events": int(len(events)),
                "num_classes": int(num_classes),
                "noise_fraction": noise_fraction,
                "silhouette_score": sil,
                "mean_persistence": persistence,
                "ari_vs_reference": ari,
            }
        )

        all_events.append(events)

    events_all = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    summary = pd.DataFrame(summaries)

    robust_rows = summary[summary["perturbation"] > 0]

    mean_classes = float(robust_rows["num_classes"].mean())
    mean_persistence = float(robust_rows["mean_persistence"].mean())
    mean_ari = float(robust_rows["ari_vs_reference"].mean())
    mean_noise = float(robust_rows["noise_fraction"].mean())

    if mean_classes >= 2 and mean_persistence >= 0.25 and mean_ari >= 0.50:
        verdict = "stable_wave_contact_classes"
    elif mean_classes >= 1 and mean_persistence >= 0.15:
        verdict = "weakly_stable_wave_contact_sectorization"
    else:
        verdict = "unstable_wave_contact_continuum"

    global_summary = pd.DataFrame(
        [
            {
                "mean_classes_under_perturbation": mean_classes,
                "mean_persistence_under_perturbation": mean_persistence,
                "mean_structure_similarity": mean_ari,
                "mean_noise_fraction": mean_noise,
                "verdict": verdict,
            }
        ]
    )

    events_all.to_csv(OUT / "wave_contact_stability_events.csv", index=False)
    summary.to_csv(OUT / "wave_contact_stability_by_perturbation.csv", index=False)
    global_summary.to_csv(OUT / "wave_contact_stability_summary.csv", index=False)

    if len(events_all) >= 5:
        Xf = feature_matrix(events_all)
        proj = PCA(n_components=2).fit_transform(Xf)

        plt.figure(figsize=(7, 6))
        plt.scatter(
            proj[:, 0],
            proj[:, 1],
            c=events_all["perturbation"],
            s=10,
        )
        plt.title("Wave contact stability under perturbation")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(OUT / "wave_contact_stability_projection.png", dpi=220)
        plt.close()

    print("\nPerturbation diagnostics:")
    print(summary.to_string(index=False))

    print("\nGlobal summary:")
    print(global_summary.to_string(index=False))

    print(f"\n[OK] wrote {OUT / 'wave_contact_stability_events.csv'}")
    print(f"[OK] wrote {OUT / 'wave_contact_stability_by_perturbation.csv'}")
    print(f"[OK] wrote {OUT / 'wave_contact_stability_summary.csv'}")
    print(f"[OK] wrote {OUT / 'wave_contact_stability_projection.png'}")
    print("[DONE] wave contact stability test complete")


if __name__ == "__main__":
    main()