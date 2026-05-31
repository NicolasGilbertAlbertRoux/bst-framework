import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================================
# PARAMETERS
# ============================================================

N = 96

observable_blur = 4.0

latent_noise = 0.14

n_latents = 260

thresholds = [
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
]

np.random.seed(1234)

# ============================================================
# BASE LATENT
# ============================================================

x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)

R = np.sqrt(X**2 + Y**2)

base_latent = (
    np.exp(-(R / 0.10)**2)
    - 0.65 * np.exp(-((R - 0.12) / 0.04)**2)
)

# ============================================================
# RESOLUTION OPERATOR
# ============================================================

def resolve(L):

    return gaussian_filter(L, observable_blur)

# ============================================================
# METRICS
# ============================================================

def mse(a, b):

    return np.mean((a - b)**2)

def corr(a, b):

    aa = a.flatten()
    bb = b.flatten()

    if np.std(aa) < 1e-10:
        return 0.0

    if np.std(bb) < 1e-10:
        return 0.0

    return np.corrcoef(aa, bb)[0, 1]

# ------------------------------------------------------------

def normalized_energy(x):

    return np.mean(x**2)

# ------------------------------------------------------------

def normalized_coherence(x):

    gx, gy = np.gradient(x)

    grad_energy = np.mean(gx**2 + gy**2)

    signal = np.mean(x**2)

    return signal / (1e-8 + grad_energy)

# ------------------------------------------------------------

def normalized_structure(x):

    gx, gy = np.gradient(x)

    mag = np.sqrt(gx**2 + gy**2)

    count = np.sum(
        mag > np.percentile(mag, 90)
    )

    return count / x.size

# ------------------------------------------------------------

def lowfreq_ratio(x):

    F = np.abs(
        np.fft.fftshift(
            np.fft.fft2(x)
        )
    )

    total = np.mean(F)

    cx = N // 2
    cy = N // 2

    low = np.mean(
        F[cx-5:cx+5, cy-5:cy+5]
    )

    return low / (1e-8 + total)

# ------------------------------------------------------------

def observable_entropy(x):

    hist, _ = np.histogram(
        x.flatten(),
        bins=64,
        density=True
    )

    hist = hist + 1e-12

    hist = hist / np.sum(hist)

    return -np.sum(hist * np.log(hist))

# ============================================================
# TARGET OBSERVABLE
# ============================================================

target_obs = resolve(base_latent)

# ============================================================
# GENERATE LATENTS
# ============================================================

latents = []
observables = []

for i in range(n_latents):

    noise = latent_noise * np.random.randn(N, N)

    # add structured latent distortions

    mode = np.sin(
        (i + 1) * np.pi * X
    ) * np.cos(
        (i + 2) * np.pi * Y
    )

    L = (
        base_latent
        + noise
        + 0.02 * mode
    )

    O = resolve(L)

    latents.append(L)
    observables.append(O)

# ============================================================
# SCAN
# ============================================================

rows = []

for threshold in thresholds:

    compatible = []

    for L, O in zip(latents, observables):

        fit = mse(O, target_obs)

        if fit < threshold:

            compatible.append((L, O))

    print(
        f"\nthreshold={threshold:.6f} "
        f"compatible={len(compatible)}"
    )

    if len(compatible) < 2:
        continue

    latent_div = []
    obs_div = []

    energies = []
    coherences = []
    structures = []
    spectra = []
    entropies = []

    for i in range(len(compatible)):

        L, O = compatible[i]

        energies.append(
            normalized_energy(O)
        )

        coherences.append(
            normalized_coherence(O)
        )

        structures.append(
            normalized_structure(O)
        )

        spectra.append(
            lowfreq_ratio(O)
        )

        entropies.append(
            observable_entropy(O)
        )

        for j in range(i + 1, len(compatible)):

            L2, O2 = compatible[j]

            latent_div.append(
                mse(L, L2)
            )

            obs_div.append(
                mse(O, O2)
            )

    rows.append({

        "threshold":
            threshold,

        "n_compatible":
            len(compatible),

        "latent_diversity":
            np.mean(latent_div),

        "observable_diversity":
            np.mean(obs_div),

        "energy_std":
            np.std(energies),

        "coherence_std":
            np.std(coherences),

        "structure_std":
            np.std(structures),

        "spectrum_std":
            np.std(spectra),

        "entropy_std":
            np.std(entropies),

        "mean_obs_corr":
            np.mean([
                corr(O, target_obs)
                for _, O in compatible
            ])
    })

# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

print("\n=== R8b SUMMARY ===")
print(df.to_string(index=False))

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["threshold"],
    df["latent_diversity"],
    marker="o",
    label="latent diversity"
)

plt.plot(
    df["threshold"],
    df["observable_diversity"],
    marker="o",
    label="observable diversity"
)

plt.xlabel("observable threshold")
plt.ylabel("MSE")

plt.title(
    "Latent vs observable diversity (R8b)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/research_final/"
    "r8b_diversity.png"
)

# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    df["threshold"],
    df["energy_std"],
    marker="o",
    label="energy"
)

plt.plot(
    df["threshold"],
    df["coherence_std"],
    marker="o",
    label="coherence"
)

plt.plot(
    df["threshold"],
    df["structure_std"],
    marker="o",
    label="structure"
)

plt.plot(
    df["threshold"],
    df["spectrum_std"],
    marker="o",
    label="spectrum"
)

plt.plot(
    df["threshold"],
    df["entropy_std"],
    marker="o",
    label="entropy"
)

plt.xlabel("observable threshold")
plt.ylabel("invariant variability")

plt.title(
    "Normalized invariant stability (R8b)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/research_final/"
    "r8b_invariants.png"
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "results/research_final/"
    "r8b_summary.csv",
    index=False
)

print(
    "\n[OK] wrote "
    "results/research_final/r8b_summary.csv"
)

print(
    "[OK] wrote "
    "results/research_final/r8b_diversity.png"
)

print(
    "[OK] wrote "
    "results/research_final/r8b_invariants.png"
)

print(
    "\n[DONE] R8b complete"
)