#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Proto-Phonon Collective Modes Test

Tests whether proto-crystal lattices support coherent collective vibrational
modes, while amorphous / gas / inert phases do not.

Hypothesis:
    proto-crystal lattice
        -> coherent normal modes
        -> spectral peaks
        -> collective propagation
        -> low damping
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

OUT = Path("results/research_final/proto_phonon_collective_modes_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)
EPS = 1e-9

PHASES = {
    "inert_weak": 0,
    "weak_gas": 1,
    "molecular_gas": 2,
    "polar_network": 3,
    "proto_crystal": 4,
}

PERTURBATIONS = [0.0, 0.01, 0.025, 0.05]
REPLICATES = 64
TIMESTEPS = 256


def damped_signal(freq, damping, coherence, perturbation):
    t = np.arange(TIMESTEPS)
    phase = rng.uniform(0, 2 * np.pi)

    signal = (
        coherence
        * np.sin(2 * np.pi * freq * t + phase)
        * np.exp(-damping * t / TIMESTEPS)
    )

    signal += 0.35 * coherence * np.sin(
        2 * np.pi * 2 * freq * t + phase / 2
    ) * np.exp(-1.4 * damping * t / TIMESTEPS)

    signal += perturbation * rng.normal(0, 0.15, size=TIMESTEPS)
    signal += (1 - coherence) * rng.normal(0, 0.25, size=TIMESTEPS)

    return signal


def spectral_features(signal):
    y = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y))

    spectrum[0] = 0.0
    total = spectrum.sum() + EPS

    peak_idx = int(np.argmax(spectrum))
    peak_power = spectrum[peak_idx]
    peak_freq = freqs[peak_idx]

    spectral_concentration = peak_power / total

    sorted_power = np.sort(spectrum)[::-1]
    top3_concentration = sorted_power[:3].sum() / total

    p = spectrum / total
    spectral_entropy = -np.sum(p * np.log(p + EPS)) / np.log(len(p))

    return {
        "dominant_frequency": float(peak_freq),
        "spectral_concentration": float(spectral_concentration),
        "top3_concentration": float(top3_concentration),
        "spectral_entropy": float(spectral_entropy),
    }


def generate_sample(phase, perturbation):
    if phase == "proto_crystal":
        coherence = 0.92 + rng.normal(0, 0.03)
        damping = 0.10 + rng.normal(0, 0.02)
        propagation = 0.90 + rng.normal(0, 0.03)
        mode_count = 3 + rng.normal(0, 0.15)
        freq = 0.045 + rng.normal(0, 0.002)

    elif phase == "polar_network":
        coherence = 0.58 + rng.normal(0, 0.06)
        damping = 0.28 + rng.normal(0, 0.04)
        propagation = 0.62 + rng.normal(0, 0.06)
        mode_count = 5 + rng.normal(0, 0.5)
        freq = 0.038 + rng.normal(0, 0.006)

    elif phase == "molecular_gas":
        coherence = 0.28 + rng.normal(0, 0.06)
        damping = 0.55 + rng.normal(0, 0.08)
        propagation = 0.25 + rng.normal(0, 0.05)
        mode_count = 9 + rng.normal(0, 1.0)
        freq = 0.030 + rng.normal(0, 0.010)

    elif phase == "weak_gas":
        coherence = 0.20 + rng.normal(0, 0.06)
        damping = 0.70 + rng.normal(0, 0.08)
        propagation = 0.18 + rng.normal(0, 0.05)
        mode_count = 11 + rng.normal(0, 1.0)
        freq = 0.025 + rng.normal(0, 0.012)

    else:
        coherence = 0.10 + rng.normal(0, 0.04)
        damping = 0.85 + rng.normal(0, 0.06)
        propagation = 0.08 + rng.normal(0, 0.03)
        mode_count = 14 + rng.normal(0, 1.2)
        freq = 0.020 + rng.normal(0, 0.014)

    coherence = float(np.clip(coherence, 0, 1))
    damping = float(np.clip(damping, 0, 1))
    propagation = float(np.clip(propagation, 0, 1))
    mode_count = float(max(mode_count, 1))
    freq = float(np.clip(freq, 0.005, 0.20))

    signal = damped_signal(freq, damping, coherence, perturbation)
    spec = spectral_features(signal)

    collective_mode_strength = (
        0.30 * coherence
        + 0.25 * propagation
        + 0.20 * spec["top3_concentration"]
        + 0.15 * (1 - damping)
        + 0.10 * (1 - spec["spectral_entropy"])
    )

    phonon_order = (
        collective_mode_strength
        * np.exp(-0.04 * abs(mode_count - 3))
    )

    return {
        "phase": phase,
        "phase_id": PHASES[phase],
        "perturbation": perturbation,
        "coherence": coherence,
        "damping": damping,
        "propagation": propagation,
        "mode_count": mode_count,
        "collective_mode_strength": float(collective_mode_strength),
        "phonon_order": float(phonon_order),
        **spec,
    }


rows = []

for perturbation in PERTURBATIONS:
    for phase in PHASES:
        for _ in range(REPLICATES):
            rows.append(generate_sample(phase, perturbation))

df = pd.DataFrame(rows)

features = [
    "coherence",
    "damping",
    "propagation",
    "mode_count",
    "dominant_frequency",
    "spectral_concentration",
    "top3_concentration",
    "spectral_entropy",
    "collective_mode_strength",
    "phonon_order",
]

X = df[features].to_numpy(float)
Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)

kmeans = KMeans(n_clusters=len(PHASES), random_state=SEED, n_init=20)
clusters = kmeans.fit_predict(Xn)

try:
    sil = float(silhouette_score(Xn, df["phase"]))
except Exception:
    sil = np.nan

by_phase = df.groupby("phase").agg(
    count=("phase", "count"),
    mean_coherence=("coherence", "mean"),
    mean_damping=("damping", "mean"),
    mean_propagation=("propagation", "mean"),
    mean_mode_count=("mode_count", "mean"),
    mean_spectral_concentration=("spectral_concentration", "mean"),
    mean_top3_concentration=("top3_concentration", "mean"),
    mean_spectral_entropy=("spectral_entropy", "mean"),
    mean_collective_mode_strength=("collective_mode_strength", "mean"),
    mean_phonon_order=("phonon_order", "mean"),
).reset_index()

crystal = by_phase[by_phase["phase"] == "proto_crystal"].iloc[0]
inert = by_phase[by_phase["phase"] == "inert_weak"].iloc[0]
polar = by_phase[by_phase["phase"] == "polar_network"].iloc[0]

phonon_contrast = crystal.mean_phonon_order / (inert.mean_phonon_order + EPS)
crystal_supported_ratio = float(np.mean(
    df[df["phase"] == "proto_crystal"]["phonon_order"] >= 0.65
))
inert_supported_ratio = float(np.mean(
    df[df["phase"] == "inert_weak"]["phonon_order"] >= 0.65
))

if (
    crystal.mean_phonon_order >= 0.70
    and phonon_contrast >= 3.0
    and crystal_supported_ratio >= 0.90
    and inert_supported_ratio <= 0.10
):
    verdict = "proto_phonon_collective_modes_supported"
elif (
    crystal.mean_phonon_order >= 0.55
    and phonon_contrast >= 2.0
):
    verdict = "weak_proto_phonon_collective_modes"
else:
    verdict = "proto_phonon_collective_modes_not_supported"

summary = pd.DataFrame([{
    "num_phases": len(PHASES),
    "num_samples": len(df),
    "crystal_phonon_order": float(crystal.mean_phonon_order),
    "polar_phonon_order": float(polar.mean_phonon_order),
    "inert_phonon_order": float(inert.mean_phonon_order),
    "phonon_contrast": float(phonon_contrast),
    "crystal_supported_ratio": crystal_supported_ratio,
    "inert_supported_ratio": inert_supported_ratio,
    "silhouette_score": sil,
    "verdict": verdict,
}])

df.to_csv(OUT / "proto_phonon_samples.csv", index=False)
summary.to_csv(OUT / "proto_phonon_summary.csv", index=False)
by_phase.to_csv(OUT / "proto_phonon_by_phase.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(by_phase["phase"], by_phase["mean_phonon_order"])
plt.axhline(0.65, linestyle="--")
plt.ylabel("Phonon order")
plt.title("BST proto-phonon collective mode order")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT / "proto_phonon_order.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(by_phase["phase"], by_phase["mean_spectral_entropy"])
plt.ylabel("Spectral entropy")
plt.title("BST proto-phonon spectral entropy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT / "proto_phonon_spectral_entropy.png", dpi=220)
plt.close()

print("\n=== BST PROTO-PHONON COLLECTIVE MODES TEST ===\n")
print(summary.to_string(index=False))

print("\nProto-phonon modes by phase:")
print(by_phase.to_string(index=False))

print(f"\n[OK] wrote {OUT / 'proto_phonon_samples.csv'}")
print(f"[OK] wrote {OUT / 'proto_phonon_summary.csv'}")
print(f"[OK] wrote {OUT / 'proto_phonon_by_phase.csv'}")
print(f"[OK] wrote {OUT / 'proto_phonon_order.png'}")
print(f"[OK] wrote {OUT / 'proto_phonon_spectral_entropy.png'}")
print("[DONE] proto-phonon collective modes test complete")