#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

OUT = Path("results/research_final/latent_aliasing_scan")
OUT.mkdir(parents=True, exist_ok=True)

N = 192
SEED = 123

RESOLUTIONS = {
    "fine": {"sigma": 0.0, "down": 1},
    "medium": {"sigma": 2.0, "down": 2},
    "coarse": {"sigma": 5.0, "down": 4},
    "very_coarse": {"sigma": 9.0, "down": 8},
}

MICRO_NOISE_LEVELS = [0.02, 0.05, 0.10, 0.20, 0.35]


def resolve(field, sigma, down):
    y = gaussian_filter(field, sigma=sigma) if sigma > 0 else field.copy()
    if down > 1:
        low = y[::down, ::down]
        y = zoom(low, zoom=down, order=1)
        y = y[:field.shape[0], :field.shape[1]]
    return y


def corr(a, b):
    aa = a.ravel()
    bb = b.ravel()
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def make_base_latent(rng):
    yy, xx = np.indices((N, N))
    cx = cy = N // 2
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2

    core = -1.4 * np.exp(-r2 / (2 * 5.0 ** 2))
    ring = 0.9 * np.exp(-(np.sqrt(r2) - 12.0) ** 2 / (2 * 2.5 ** 2))
    background = 0.08 * gaussian_filter(rng.normal(size=(N, N)), sigma=4.0)

    return core + ring + background


def make_micro_perturbation(rng, level):
    raw = rng.normal(size=(N, N))

    # microstructure high-frequency:
    # remove low-frequency component
    smooth = gaussian_filter(raw, sigma=3.0)
    high = raw - smooth

    high = high / (np.std(high) + 1e-12)
    return level * high


def main():
    print("\n=== LATENT ALIASING SCAN ===")

    rng = np.random.default_rng(SEED)

    base = make_base_latent(rng)

    rows = []
    examples = {}

    for level in MICRO_NOISE_LEVELS:
        perturb = make_micro_perturbation(rng, level)

        L1 = base + perturb
        L2 = base - perturb

        latent_mse = mse(L1, L2)
        latent_corr = corr(L1, L2)

        for name, cfg in RESOLUTIONS.items():
            O1 = resolve(L1, cfg["sigma"], cfg["down"])
            O2 = resolve(L2, cfg["sigma"], cfg["down"])

            rows.append({
                "micro_noise_level": level,
                "resolution": name,
                "sigma": cfg["sigma"],
                "downsample": cfg["down"],
                "latent_mse": latent_mse,
                "latent_corr": latent_corr,
                "observable_mse": mse(O1, O2),
                "observable_corr": corr(O1, O2),
                "aliasing_ratio_mse": mse(O1, O2) / (latent_mse + 1e-12),
            })

            if level == MICRO_NOISE_LEVELS[-1]:
                examples[name] = (O1, O2, O1 - O2)

    df = pd.DataFrame(rows)
    csv_path = OUT / "latent_aliasing_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))

    # Plot aliasing ratio
    plt.figure(figsize=(8, 5))
    for name in RESOLUTIONS:
        sub = df[df["resolution"] == name]
        plt.plot(sub["micro_noise_level"], sub["aliasing_ratio_mse"], marker="o", label=name)
    plt.xlabel("micro noise level")
    plt.ylabel("observable MSE / latent MSE")
    plt.title("Aliasing ratio by resolution")
    plt.legend()
    plt.tight_layout()
    fig1 = OUT / "aliasing_ratio_by_resolution.png"
    plt.savefig(fig1, dpi=260)
    plt.close()

    # Plot observable corr
    plt.figure(figsize=(8, 5))
    for name in RESOLUTIONS:
        sub = df[df["resolution"] == name]
        plt.plot(sub["micro_noise_level"], sub["observable_corr"], marker="o", label=name)
    plt.xlabel("micro noise level")
    plt.ylabel("corr(R(L1), R(L2))")
    plt.title("Observable similarity despite latent difference")
    plt.legend()
    plt.tight_layout()
    fig2 = OUT / "observable_corr_by_resolution.png"
    plt.savefig(fig2, dpi=260)
    plt.close()

    # Example figure for strongest perturbation
    fig, axes = plt.subplots(len(RESOLUTIONS), 3, figsize=(12, 12))
    for i, (name, (O1, O2, D)) in enumerate(examples.items()):
        axes[i, 0].imshow(O1, cmap="coolwarm")
        axes[i, 0].set_title(f"{name}: R(L1)")
        axes[i, 1].imshow(O2, cmap="coolwarm")
        axes[i, 1].set_title(f"{name}: R(L2)")
        axes[i, 2].imshow(D, cmap="coolwarm")
        axes[i, 2].set_title(f"{name}: difference")
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.tight_layout()
    fig3 = OUT / "aliasing_examples.png"
    plt.savefig(fig3, dpi=260)
    plt.close()

    print(f"\n[OK] wrote {csv_path}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print(f"[OK] wrote {fig3}")
    print("[DONE] latent aliasing scan complete")


if __name__ == "__main__":
    main()