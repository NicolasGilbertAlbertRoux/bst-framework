#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

OUT = Path("results/research_final/latent_equivalence_class_scan")
OUT.mkdir(parents=True, exist_ok=True)

N = 160
N_CANDIDATES = 220

SIGMA_OBS = 6.0
DOWN_OBS = 4

SEED = 2026

FIT_THRESHOLDS = [0.0005, 0.001, 0.002, 0.005, 0.010]


def resolve(field):
    y = gaussian_filter(field, sigma=SIGMA_OBS)
    low = y[::DOWN_OBS, ::DOWN_OBS]
    y = zoom(low, zoom=DOWN_OBS, order=1)
    return y[:field.shape[0], :field.shape[1]]


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def corr(a, b):
    aa = a.ravel()
    bb = b.ravel()
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def make_visible_core():
    yy, xx = np.indices((N, N))
    cx = cy = N // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    core = -1.3 * np.exp(-r**2 / (2 * 4.5**2))
    ring = 0.8 * np.exp(-(r - 12.0)**2 / (2 * 2.4**2))
    mantle = 0.12 * np.exp(-r**2 / (2 * 45.0**2))

    return core + ring + mantle


def make_candidate_latent(rng, strength):
    base = make_visible_core()

    raw = rng.normal(size=(N, N))
    smooth = gaussian_filter(raw, sigma=3.5)
    micro = raw - smooth
    micro /= np.std(micro) + 1e-12

    meso_raw = rng.normal(size=(N, N))
    meso = gaussian_filter(meso_raw, sigma=1.2)
    meso -= gaussian_filter(meso, sigma=7.0)
    meso /= np.std(meso) + 1e-12

    phase = rng.uniform(0, 2*np.pi)
    yy, xx = np.indices((N, N))
    wave = np.sin(0.45 * xx + 0.31 * yy + phase)
    wave = wave - gaussian_filter(wave, sigma=6.0)
    wave /= np.std(wave) + 1e-12

    return base + strength * (0.55 * micro + 0.30 * meso + 0.15 * wave)


def latent_diversity(latents):
    if len(latents) < 2:
        return np.nan, np.nan

    mses = []
    corrs = []

    for i in range(len(latents)):
        for j in range(i + 1, len(latents)):
            mses.append(mse(latents[i], latents[j]))
            corrs.append(corr(latents[i], latents[j]))

    return float(np.mean(mses)), float(np.mean(corrs))


def main():
    print("\n=== LATENT EQUIVALENCE CLASS SCAN ===")

    rng = np.random.default_rng(SEED)

    target_latent = make_visible_core()
    target_obs = resolve(target_latent)

    candidates = []

    for i in range(N_CANDIDATES):
        strength = rng.uniform(0.02, 0.45)
        L = make_candidate_latent(rng, strength)
        O = resolve(L)

        candidates.append({
            "idx": i,
            "strength": strength,
            "latent": L,
            "observable": O,
            "obs_mse": mse(O, target_obs),
            "obs_corr": corr(O, target_obs),
            "latent_mse_vs_target": mse(L, target_latent),
            "latent_corr_vs_target": corr(L, target_latent),
        })

    rows = []
    examples = {}

    for threshold in FIT_THRESHOLDS:
        accepted = [c for c in candidates if c["obs_mse"] <= threshold]
        accepted_latents = [c["latent"] for c in accepted]

        div_mse, div_corr = latent_diversity(accepted_latents)

        rows.append({
            "fit_threshold": threshold,
            "n_accepted": len(accepted),
            "accepted_fraction": len(accepted) / N_CANDIDATES,
            "mean_obs_mse": float(np.mean([c["obs_mse"] for c in accepted])) if accepted else np.nan,
            "mean_obs_corr": float(np.mean([c["obs_corr"] for c in accepted])) if accepted else np.nan,
            "mean_latent_mse_vs_target": float(np.mean([c["latent_mse_vs_target"] for c in accepted])) if accepted else np.nan,
            "mean_latent_corr_vs_target": float(np.mean([c["latent_corr_vs_target"] for c in accepted])) if accepted else np.nan,
            "latent_pairwise_diversity_mse": div_mse,
            "latent_pairwise_diversity_corr": div_corr,
        })

        if accepted:
            # keep three most latent-different among good fits
            accepted_sorted = sorted(
                accepted,
                key=lambda c: c["latent_mse_vs_target"],
                reverse=True
            )
            examples[threshold] = accepted_sorted[:3]

    df = pd.DataFrame(rows)
    csv_path = OUT / "latent_equivalence_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.plot(df["fit_threshold"], df["n_accepted"], marker="o")
    plt.xlabel("observable fit threshold (MSE)")
    plt.ylabel("number of compatible latents")
    plt.title("Size of latent equivalence class")
    plt.tight_layout()
    fig1 = OUT / "equivalence_class_size.png"
    plt.savefig(fig1, dpi=260)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["fit_threshold"], df["latent_pairwise_diversity_mse"], marker="o")
    plt.xlabel("observable fit threshold (MSE)")
    plt.ylabel("latent pairwise diversity MSE")
    plt.title("Diversity inside compatible latent class")
    plt.tight_layout()
    fig2 = OUT / "latent_diversity_inside_class.png"
    plt.savefig(fig2, dpi=260)
    plt.close()

    # examples for largest threshold
    threshold = FIT_THRESHOLDS[-1]
    chosen = examples.get(threshold, [])

    if chosen:
        fig, axes = plt.subplots(len(chosen), 3, figsize=(11, 3.5 * len(chosen)))
        if len(chosen) == 1:
            axes = np.array([axes])

        for row_idx, c in enumerate(chosen):
            axes[row_idx, 0].imshow(c["latent"], cmap="coolwarm")
            axes[row_idx, 0].set_title(f"L{c['idx']} latent")

            axes[row_idx, 1].imshow(c["observable"], cmap="coolwarm")
            axes[row_idx, 1].set_title(f"R(L{c['idx']})")

            axes[row_idx, 2].imshow(c["observable"] - target_obs, cmap="coolwarm")
            axes[row_idx, 2].set_title("observable residual")

            for j in range(3):
                axes[row_idx, j].set_xticks([])
                axes[row_idx, j].set_yticks([])

        plt.tight_layout()
        fig3 = OUT / "compatible_latent_examples.png"
        plt.savefig(fig3, dpi=260)
        plt.close()
    else:
        fig3 = None

    print(f"\n[OK] wrote {csv_path}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    if fig3:
        print(f"[OK] wrote {fig3}")
    print("[DONE] latent equivalence class scan complete")


if __name__ == "__main__":
    main()