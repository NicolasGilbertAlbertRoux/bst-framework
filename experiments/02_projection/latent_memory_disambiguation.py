#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

OUT = Path("results/research_final/latent_memory_disambiguation")
OUT.mkdir(parents=True, exist_ok=True)

N = 160
STEPS = 220
DT = 0.05

D = 0.18
GAMMA = 0.015
LAMBDA = 0.010
DAMP = 0.997

SIGMA_OBS = 5.0
DOWN_OBS = 4

MEMORY_WINDOWS = [1, 3, 5, 10, 20, 40]

SEED = 1234


def laplacian(x):
    return (
        np.roll(x, 1, 0) + np.roll(x, -1, 0)
        + np.roll(x, 1, 1) + np.roll(x, -1, 1)
        - 4 * x
    )


def resolve(field):
    y = gaussian_filter(field, sigma=SIGMA_OBS)
    low = y[::DOWN_OBS, ::DOWN_OBS]
    y = zoom(low, zoom=DOWN_OBS, order=1)
    return y[:field.shape[0], :field.shape[1]]


def corr(a, b):
    aa = a.ravel()
    bb = b.ravel()

    sa = np.std(aa)
    sb = np.std(bb)

    if sa < 1e-12 or sb < 1e-12:
        return np.nan

    return float(np.corrcoef(aa, bb)[0, 1])


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def evolve(phi, vel, forcing):
    force = (
        D * laplacian(phi)
        - GAMMA * phi
        - LAMBDA * phi**3
        + forcing
    )

    vel = DAMP * vel + DT * force
    phi = phi + DT * vel

    return phi, vel


def make_initial_pair(rng):
    yy, xx = np.indices((N, N))
    cx = cy = N // 2

    r2 = (xx - cx)**2 + (yy - cy)**2

    base = (
        -1.5 * np.exp(-r2 / (2 * 4.0**2))
        + 0.9 * np.exp(-(np.sqrt(r2) - 12)**2 / (2 * 2.0**2))
    )

    noise = rng.normal(size=(N, N))
    micro = noise - gaussian_filter(noise, sigma=3.0)
    micro /= np.std(micro)

    L1 = base + 0.22 * micro
    L2 = base - 0.22 * micro

    return L1, L2


def memory_signature(history, window):
    if len(history) < window:
        return None

    stack = np.stack(history[-window:], axis=0)

    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)

    return np.concatenate([
        mean.ravel(),
        std.ravel(),
    ])


def main():
    print("\n=== LATENT MEMORY DISAMBIGUATION ===")

    rng = np.random.default_rng(SEED)

    phi1, phi2 = make_initial_pair(rng)

    vel1 = np.zeros_like(phi1)
    vel2 = np.zeros_like(phi2)

    hist_obs1 = []
    hist_obs2 = []

    latent_corrs = []
    observable_corrs = []

    for step in range(STEPS):

        forcing1 = 0.02 * np.sin(0.17 * step)
        forcing2 = 0.02 * np.sin(0.17 * step + 0.4)

        phi1, vel1 = evolve(phi1, vel1, forcing1)
        phi2, vel2 = evolve(phi2, vel2, forcing2)

        O1 = resolve(phi1)
        O2 = resolve(phi2)

        hist_obs1.append(O1.copy())
        hist_obs2.append(O2.copy())

        latent_corrs.append(corr(phi1, phi2))
        observable_corrs.append(corr(O1, O2))

        if step % 40 == 0:
            print(
                f"step={step:03d} "
                f"latent_corr={latent_corrs[-1]:.4f} "
                f"observable_corr={observable_corrs[-1]:.4f}"
            )

    rows = []

    for w in MEMORY_WINDOWS:

        sig1 = memory_signature(hist_obs1, w)
        sig2 = memory_signature(hist_obs2, w)

        if sig1 is None:
            continue

        mem_corr = corr(sig1, sig2)
        mem_mse = mse(sig1, sig2)

        rows.append({
            "memory_window": w,
            "observable_corr_final": observable_corrs[-1],
            "latent_corr_final": latent_corrs[-1],
            "memory_signature_corr": mem_corr,
            "memory_signature_mse": mem_mse,
        })

    df = pd.DataFrame(rows)

    csv_path = OUT / "memory_disambiguation_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== MEMORY SUMMARY ===")
    print(df.to_string(index=False))

    # corr evolution
    plt.figure(figsize=(8, 5))
    plt.plot(latent_corrs, label="latent corr")
    plt.plot(observable_corrs, label="observable corr")
    plt.xlabel("step")
    plt.ylabel("correlation")
    plt.title("Latent vs observable similarity")
    plt.legend()
    plt.tight_layout()

    fig1 = OUT / "latent_vs_observable_corr.png"
    plt.savefig(fig1, dpi=260)
    plt.close()

    # memory curves
    plt.figure(figsize=(8, 5))
    plt.plot(df["memory_window"], df["memory_signature_corr"], marker="o")
    plt.xlabel("memory window")
    plt.ylabel("memory signature correlation")
    plt.title("Memory-based disambiguation")
    plt.tight_layout()

    fig2 = OUT / "memory_disambiguation.png"
    plt.savefig(fig2, dpi=260)
    plt.close()

    print(f"\n[OK] wrote {csv_path}")
    print(f"[OK] wrote {fig1}")
    print(f"[OK] wrote {fig2}")
    print("[DONE] latent memory disambiguation complete")


if __name__ == "__main__":
    main()