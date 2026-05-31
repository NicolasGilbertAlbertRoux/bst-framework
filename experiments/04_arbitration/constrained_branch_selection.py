import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================================
# PARAMETERS
# ============================================================

N = 96

n_candidates = 220

observable_blur = 4.0
latent_noise = 0.08

observable_threshold = 0.001

np.random.seed(123)

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

def energy(x):

    return np.mean(x**2)

def asymmetry(x):

    left = np.mean(x[:, :N//2])
    right = np.mean(x[:, N//2:])

    return right - left

def spectral_bias(x):

    F = np.abs(np.fft.fftshift(np.fft.fft2(x)))

    cx = N // 2
    cy = N // 2

    q1 = np.mean(F[:cx, :cy])
    q2 = np.mean(F[cx:, cy:])

    return q2 - q1

# ============================================================
# GLOBAL CONSTRAINT
# ============================================================

# This is the "arbitration field"

constraint_field = (
    0.08 * X
    + 0.05 * np.sin(3 * np.pi * Y)
)

# ============================================================
# TARGET OBSERVABLE
# ============================================================

target_obs = resolve(base_latent)

# ============================================================
# GENERATE COMPATIBLE LATENTS
# ============================================================

compatible = []

for i in range(n_candidates):

    noise = latent_noise * np.random.randn(N, N)

    L = base_latent + noise

    O = resolve(L)

    fit = mse(O, target_obs)

    if fit < observable_threshold:

        compatible.append((L, O))

print(
    f"\ncompatible latent count = "
    f"{len(compatible)}"
)

# ============================================================
# ARBITRATION SCORE
# ============================================================

rows = []

for idx, (L, O) in enumerate(compatible):

    latent_alignment = np.mean(L * constraint_field)

    obs_alignment = np.mean(O * constraint_field)

    score = (
        1.5 * latent_alignment
        + 0.5 * obs_alignment
        + 0.2 * asymmetry(L)
        + 0.2 * spectral_bias(L)
    )

    rows.append({
        "idx": idx,

        "observable_fit":
            mse(O, target_obs),

        "latent_alignment":
            latent_alignment,

        "observable_alignment":
            obs_alignment,

        "latent_energy":
            energy(L),

        "observable_energy":
            energy(O),

        "latent_asymmetry":
            asymmetry(L),

        "latent_spectral_bias":
            spectral_bias(L),

        "selection_score":
            score,
    })

df = pd.DataFrame(rows)

# ============================================================
# SELECTED BRANCH
# ============================================================

df = df.sort_values(
    "selection_score",
    ascending=False
)

selected_idx = int(df.iloc[0]["idx"])

selected_L, selected_O = compatible[selected_idx]

print("\n=== TOP BRANCHES ===")

print(
    df.head(10).to_string(index=False)
)

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["observable_fit"],
    df["selection_score"]
)

plt.xlabel("observable fit")
plt.ylabel("selection score")

plt.title(
    "Branch selection inside observable equivalence class"
)

plt.tight_layout()

plt.savefig(
    "results/research_final/"
    "branch_selection_scatter.png"
)

# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4)
)

axes[0].imshow(selected_L, cmap="coolwarm")
axes[0].set_title("selected latent")

axes[1].imshow(selected_O, cmap="coolwarm")
axes[1].set_title("selected observable")

axes[2].imshow(constraint_field, cmap="coolwarm")
axes[2].set_title("constraint field")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "results/research_final/"
    "selected_branch.png"
)

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "n_compatible":
        len(compatible),

    "best_selection_score":
        df["selection_score"].max(),

    "worst_selection_score":
        df["selection_score"].min(),

    "selection_score_std":
        df["selection_score"].std(),

    "observable_fit_std":
        df["observable_fit"].std(),

    "latent_alignment_std":
        df["latent_alignment"].std(),

    "spectral_bias_std":
        df["latent_spectral_bias"].std(),
}])

print("\n=== SUMMARY ===")
print(summary.to_string(index=False))

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    "results/research_final/"
    "branch_selection_details.csv",
    index=False
)

summary.to_csv(
    "results/research_final/"
    "branch_selection_summary.csv",
    index=False
)

print(
    "\n[OK] wrote "
    "results/research_final/"
    "branch_selection_details.csv"
)

print(
    "[OK] wrote "
    "results/research_final/"
    "branch_selection_summary.csv"
)

print(
    "[OK] wrote "
    "results/research_final/"
    "branch_selection_scatter.png"
)

print(
    "[OK] wrote "
    "results/research_final/"
    "selected_branch.png"
)

print(
    "\n[DONE] constrained branch selection complete"
)