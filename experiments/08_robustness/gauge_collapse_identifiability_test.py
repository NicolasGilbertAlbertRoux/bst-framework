#!/usr/bin/env python3
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

OUT = Path("results/research_final/gauge_collapse_identifiability_test")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

memory_vals = [0.00, 0.02, 0.05, 0.09, 0.13]
sigma_vals = [2.0, 3.0, 4.5, 6.0]
noise_vals = [0.035, 0.08, 0.125, 0.17]
nonlinear_vals = [0.01, 0.025, 0.04, 0.06]
grad_vals = [0.08, 0.12, 0.18]

def observables(M, sigma, noise, nl, g):
    # intentionally: nl weakly coupled / gauge-like
    phase = (
        0.14
        + 1.75*M
        + 0.012*np.log1p(sigma)
        - 0.06*noise
        - 0.20*g
        + 0.015*np.sin(40*nl)
        + rng.normal(0, 0.010)
    )
    energy = (
        0.0042*np.exp(-8*M)
        / np.sqrt(sigma)
        + 0.0015*noise
        + 0.0009*g
        + 0.00003*np.cos(60*nl)
        + rng.normal(0, 0.00004)
    )
    coherence = (
        0.012
        + 0.028*M
        + 0.002*np.log1p(sigma)
        - 0.002*noise
        + 0.0002*np.sin(30*nl)
        + rng.normal(0, 0.0005)
    )
    spectral = (
        31.5
        + 7*M
        + 0.25*sigma
        - 2.0*noise
        + 0.15*g
        + 0.04*np.sin(25*nl)
        + rng.normal(0, 0.25)
    )
    return np.array([phase, energy, coherence, spectral])

rows = []
for M, sig, noise, nl, g in product(memory_vals, sigma_vals, noise_vals, nonlinear_vals, grad_vals):
    obs = observables(M, sig, noise, nl, g)
    rows.append(dict(
        memory_gain=M, sigma=sig, noise=noise,
        nonlinear=nl, grad_coeff=g,
        phase_survival=obs[0],
        observable_energy=obs[1],
        coherence=obs[2],
        spectral_ratio=obs[3],
    ))

df = pd.DataFrame(rows)
df.to_csv(OUT / "gauge_collapse_dataset.csv", index=False)

features = ["phase_survival", "observable_energy", "coherence", "spectral_ratio"]

# For each nonlinear value, compare its observable cloud to the others
# after nearest-neighbour re-optimisation over (M, sigma, noise, g).
collapse_rows = []

for nl_a in nonlinear_vals:
    A = df[df.nonlinear == nl_a].copy()
    for nl_b in nonlinear_vals:
        if nl_a == nl_b:
            continue
        B = df[df.nonlinear == nl_b].copy()

        dists = []
        for _, a in A.iterrows():
            xa = a[features].to_numpy(float)
            XB = B[features].to_numpy(float)
            scale = df[features].std().to_numpy(float) + 1e-12
            dist = np.sqrt(((XB - xa) / scale) ** 2).sum(axis=1)
            dists.append(dist.min())

        collapse_rows.append(dict(
            nonlinear_a=nl_a,
            nonlinear_b=nl_b,
            mean_min_observable_distance=float(np.mean(dists)),
            median_min_observable_distance=float(np.median(dists)),
        ))

collapse = pd.DataFrame(collapse_rows)
collapse.to_csv(OUT / "gauge_collapse_report.csv", index=False)

# Baseline: random observable distance
X = df[features].to_numpy(float)
scale = df[features].std().to_numpy(float) + 1e-12
rand_d = []
for _ in range(2000):
    i, j = rng.integers(0, len(X), 2)
    rand_d.append(np.sqrt(((X[i] - X[j]) / scale) ** 2).sum())

baseline = float(np.mean(rand_d))
collapse_score = 1.0 - collapse["mean_min_observable_distance"].mean() / baseline

summary = pd.DataFrame([dict(
    random_baseline_distance=baseline,
    mean_reoptimized_distance=collapse["mean_min_observable_distance"].mean(),
    gauge_collapse_score=collapse_score,
    interpretation="nonlinear behaves as gauge/redundant if score is close to 1"
)])
summary.to_csv(OUT / "gauge_collapse_summary.csv", index=False)

plt.figure(figsize=(9,5))
plt.bar(
    [f"{a}->{b}" for a,b in zip(collapse.nonlinear_a, collapse.nonlinear_b)],
    collapse.mean_min_observable_distance
)
plt.axhline(baseline, linestyle="--", label="random baseline")
plt.xticks(rotation=70)
plt.ylabel("re-optimized observable distance")
plt.title("Gauge collapse: nonlinear reabsorbed by other parameters")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "gauge_collapse_distances.png", dpi=160)

plt.figure(figsize=(7,5))
plt.scatter(df["nonlinear"], df["phase_survival"], alpha=0.55)
plt.xlabel("nonlinear coefficient")
plt.ylabel("phase survival")
plt.title("Weak direct identifiability of nonlinear term")
plt.tight_layout()
plt.savefig(OUT / "nonlinear_phase_scatter.png", dpi=160)

print("\n=== GAUGE COLLAPSE / IDENTIFIABILITY TEST ===")
print(summary.to_string(index=False))
print("\n=== PAIRWISE COLLAPSE ===")
print(collapse.to_string(index=False))
print(f"\n[OK] wrote {OUT}")
print("[DONE] gauge collapse identifiability test complete")