#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Periodic Contact Attractor Test

Purpose
-------
Test whether matter-like contact structures are better described as
periodic intensity attractors rather than fixed contact clusters.

Core idea
---------
Because wave mantles are diffuse, contact matter may not persist as a
binary object. Instead, its contact intensity may rise, fade, and return
periodically across breathing cycles.

Outputs
-------
results/research_final/periodic_contact_attractor_test/
    periodic_contact_attractor_timeseries.csv
    periodic_contact_attractor_summary.csv
    periodic_contact_attractor_signal.png
    periodic_contact_attractor_spectrum.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/research_final/periodic_contact_attractor_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)

N = 128
T_STEPS = 360
NUM_CLUSTERS = 4
NUM_WAVES_PER_CLUSTER = 4

x = np.linspace(-1.0, 1.0, N)
y = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(x, y)

EPS = 1e-12


def gaussian_ring(cx, cy, radius, sigma):
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return np.exp(-((r - radius) ** 2) / (2.0 * sigma**2))


def cluster_centers(t):
    phase = 2.0 * np.pi * t / T_STEPS

    # Breathing: contraction -> expansion -> retraction.
    breath = 0.42 + 0.08 * np.sin(2.0 * phase)

    centers = []
    for k in range(NUM_CLUSTERS):
        a = 2.0 * np.pi * k / NUM_CLUSTERS + 0.12 * np.sin(phase)
        centers.append([breath * np.cos(a), breath * np.sin(a)])

    return np.asarray(centers)


def mantle_field(t, perturb=0.0):
    centers = cluster_centers(t)
    mantles = []

    phase = 2.0 * np.pi * t / T_STEPS

    for cid, (cx, cy) in enumerate(centers):
        cluster = np.zeros((N, N))

        for w in range(NUM_WAVES_PER_CLUSTER):
            local_phase = phase + 0.7 * cid + 0.35 * w

            radius = 0.18 + 0.035 * np.sin(3.0 * local_phase)
            sigma = 0.038 + 0.006 * np.cos(2.0 * local_phase)

            jitter_x = perturb * rng.normal()
            jitter_y = perturb * rng.normal()

            cluster += gaussian_ring(
                cx + jitter_x,
                cy + jitter_y,
                radius,
                sigma,
            )

        cluster /= NUM_WAVES_PER_CLUSTER
        mantles.append(cluster)

    return np.asarray(mantles)


def contact_intensity(mantles):
    """
    Diffuse contact intensity, not binary contact existence.

    Pairwise mantle overlaps are summed, then central and peripheral
    contact channels are measured separately.
    """
    pair_sum = np.zeros((N, N))

    for i in range(NUM_CLUSTERS):
        for j in range(i + 1, NUM_CLUSTERS):
            pair_sum += mantles[i] * mantles[j]

    r = np.sqrt(X**2 + Y**2)

    central_mask = r < 0.22
    peripheral_mask = (r >= 0.22) & (r < 0.72)

    central = float(pair_sum[central_mask].mean())
    peripheral = float(pair_sum[peripheral_mask].mean())
    total = float(pair_sum.mean())
    peak = float(pair_sum.max())

    return total, central, peripheral, peak


records = []

for t in range(T_STEPS):
    mantles = mantle_field(t)

    total, central, peripheral, peak = contact_intensity(mantles)

    records.append(
        {
            "time": t,
            "total_contact_intensity": total,
            "central_contact_intensity": central,
            "peripheral_contact_intensity": peripheral,
            "peak_contact_intensity": peak,
        }
    )

df = pd.DataFrame(records)

signal = df["total_contact_intensity"].to_numpy()
signal = signal - signal.mean()

spectrum = np.abs(np.fft.rfft(signal)) ** 2
freqs = np.fft.rfftfreq(len(signal), d=1.0)

# Ignore zero frequency.
dominant_idx = np.argmax(spectrum[1:]) + 1
dominant_frequency = float(freqs[dominant_idx])
dominant_power = float(spectrum[dominant_idx])
total_power = float(spectrum[1:].sum() + EPS)
periodicity_score = dominant_power / total_power

# Recurrence: compare signal to itself after one dominant period.
dominant_period = int(round(1.0 / dominant_frequency)) if dominant_frequency > 0 else T_STEPS

if 1 <= dominant_period < T_STEPS:
    a = signal[:-dominant_period]
    b = signal[dominant_period:]
    recurrence_corr = float(np.corrcoef(a, b)[0, 1])
else:
    recurrence_corr = np.nan

intensity_contrast = float(
    (df["peak_contact_intensity"].max() - df["peak_contact_intensity"].min())
    / (df["peak_contact_intensity"].mean() + EPS)
)

central_peripheral_coupling = float(
    np.corrcoef(
        df["central_contact_intensity"],
        df["peripheral_contact_intensity"],
    )[0, 1]
)

if (
    periodicity_score >= 0.45
    and recurrence_corr >= 0.75
    and intensity_contrast >= 0.25
):
    verdict = "periodic_contact_attractor_supported"
elif periodicity_score >= 0.25 and recurrence_corr >= 0.50:
    verdict = "weak_periodic_contact_attractor"
else:
    verdict = "no_periodic_contact_attractor"

summary = pd.DataFrame(
    [
        {
            "dominant_frequency": dominant_frequency,
            "dominant_period": dominant_period,
            "periodicity_score": periodicity_score,
            "recurrence_corr": recurrence_corr,
            "intensity_contrast": intensity_contrast,
            "central_peripheral_coupling": central_peripheral_coupling,
            "verdict": verdict,
        }
    ]
)

df.to_csv(OUT / "periodic_contact_attractor_timeseries.csv", index=False)
summary.to_csv(OUT / "periodic_contact_attractor_summary.csv", index=False)

plt.figure(figsize=(9, 4))
plt.plot(df["time"], df["total_contact_intensity"], label="total")
plt.plot(df["time"], df["central_contact_intensity"], label="central")
plt.plot(df["time"], df["peripheral_contact_intensity"], label="peripheral")
plt.xlabel("time")
plt.ylabel("diffuse contact intensity")
plt.title("Periodic diffuse contact intensity")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "periodic_contact_attractor_signal.png", dpi=220)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(freqs[1:], spectrum[1:])
plt.xlabel("frequency")
plt.ylabel("power")
plt.title("Contact intensity spectrum")
plt.tight_layout()
plt.savefig(OUT / "periodic_contact_attractor_spectrum.png", dpi=220)
plt.close()

print("\n=== BST PERIODIC CONTACT ATTRACTOR TEST ===\n")
print(summary.to_string(index=False))
print(f"\n[OK] wrote {OUT / 'periodic_contact_attractor_timeseries.csv'}")
print(f"[OK] wrote {OUT / 'periodic_contact_attractor_summary.csv'}")
print(f"[OK] wrote {OUT / 'periodic_contact_attractor_signal.png'}")
print(f"[OK] wrote {OUT / 'periodic_contact_attractor_spectrum.png'}")
print("[DONE] periodic contact attractor test complete")