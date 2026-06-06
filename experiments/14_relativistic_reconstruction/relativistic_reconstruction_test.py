#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Emergent Relativistic Reconstruction Test
===============================================

Purpose
-------
Test whether BST wave-substrate dynamics supports an effective relativistic
reconstruction.

This test does not claim full Einstein gravity.

It checks four weaker but essential signatures:

1. existence of a Lorentzian-like effective metric signature;
2. finite propagation of local wave perturbations;
3. curvature-dependent trajectory deviation;
4. consistency between closure geometry and geodesic-like motion.

Outputs
-------
results/research_final/relativistic_reconstruction_test/
    relativistic_reconstruction_summary.csv
    propagation_arrivals.csv
    relativistic_wave_field.png
    relativistic_curvature_map.png
    relativistic_ray_deviation.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT = Path("results/research_final/relativistic_reconstruction_test")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1729
rng = np.random.default_rng(SEED)

N = 128
T_STEPS = 180
DX = 1.0
DT = 0.35

C0 = 0.85
ALPHA = 1.20
DAMP = 0.997
EPS = 1e-12

x = np.linspace(-1.0, 1.0, N)
y = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(x, y)


def gaussian(cx, cy, sigma, amp=1.0):
    return amp * np.exp(
        -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma**2)
    )


def laplacian(u):
    return (
        np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1)
        + np.roll(u, -1, axis=1)
        - 4.0 * u
    ) / (DX * DX)


def grad(f):
    gy, gx = np.gradient(f, DX, DX)
    return gx, gy


def bilinear(arr, pos):
    px = float(np.clip(pos[0], 0, N - 1.001))
    py = float(np.clip(pos[1], 0, N - 1.001))

    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, N - 1)
    y1 = min(y0 + 1, N - 1)

    tx = px - x0
    ty = py - y0

    return (
        (1 - tx) * (1 - ty) * arr[y0, x0]
        + tx * (1 - ty) * arr[y0, x1]
        + (1 - tx) * ty * arr[y1, x0]
        + tx * ty * arr[y1, x1]
    )


def sample_vec(gx, gy, pos):
    return np.array(
        [
            bilinear(gx, pos),
            bilinear(gy, pos),
        ],
        dtype=float,
    )


def build_closure_geometry():
    rho = (
        gaussian(-0.25, 0.10, 0.25, 1.00)
        + gaussian(0.35, -0.20, 0.22, 0.85)
        + 0.45 * gaussian(0.10, 0.35, 0.35, 1.00)
    )

    rho = (rho - rho.min()) / (rho.max() - rho.min() + EPS)

    refractive_index = 1.0 + ALPHA * rho
    effective_speed = C0 / (refractive_index + EPS)

    log_n = np.log(refractive_index + EPS)
    curvature_proxy = laplacian(log_n)

    return rho, refractive_index, effective_speed, curvature_proxy


def finite_propagation_test(effective_speed):
    u_prev = gaussian(0.0, 0.0, 0.045, 1.0)
    u = u_prev.copy()

    yy, xx = np.indices((N, N))
    center = np.array([N // 2, N // 2])
    rr = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)

    radii = np.array([8, 16, 24, 32, 40])
    arrivals = {int(r): None for r in radii}

    for step in range(T_STEPS):
        c2 = (effective_speed * DT / DX) ** 2
        u_next = 2.0 * u - u_prev + c2 * laplacian(u)
        u_next *= DAMP

        u_prev, u = u, u_next

        amp = np.abs(u)

        for r in radii:
            if arrivals[int(r)] is not None:
                continue

            shell = (rr >= r - 1.5) & (rr <= r + 1.5)

            if amp[shell].max() > 0.025:
                arrivals[int(r)] = step * DT

    rows = []

    for r, arrival_time in arrivals.items():
        rows.append(
            {
                "radius": r,
                "arrival_time": np.nan if arrival_time is None else arrival_time,
            }
        )

    arrival_df = pd.DataFrame(rows)

    valid = arrival_df.dropna()

    if len(valid) >= 2:
        speed, intercept = np.polyfit(
            valid["arrival_time"],
            valid["radius"],
            1,
        )
        corr = np.corrcoef(
            valid["arrival_time"],
            valid["radius"],
        )[0, 1]
    else:
        speed = np.nan
        corr = np.nan

    return u, arrival_df, float(speed), float(corr)


def integrate_ray(refractive_index, start, vel, steps=360, dt=0.08, gain=0.18):
    gx, gy = grad(np.log(refractive_index + EPS))

    pos = start.astype(float).copy()
    v = vel.astype(float).copy()

    speed = np.linalg.norm(v) + EPS

    traj = []

    for _ in range(steps):
        g = sample_vec(gx, gy, pos)

        vhat = v / (np.linalg.norm(v) + EPS)
        transverse_g = g - np.dot(g, vhat) * vhat

        acc = gain * transverse_g

        v = v + dt * acc
        v = speed * v / (np.linalg.norm(v) + EPS)

        pos = pos + dt * v

        pos[0] = np.clip(pos[0], 2, N - 3)
        pos[1] = np.clip(pos[1], 2, N - 3)

        traj.append(pos.copy())

    return np.asarray(traj)


def curvature_deviation_test(refractive_index, curvature_proxy):
    start = np.array([20.0, 104.0])
    vel = np.array([0.75, -0.27])

    ray_a = integrate_ray(refractive_index, start, vel)
    ray_b = integrate_ray(
        refractive_index,
        start + np.array([0.0, 1.25]),
        vel,
    )

    separation = np.linalg.norm(ray_b - ray_a, axis=1)

    sep_acc = np.gradient(np.gradient(separation))

    curvature_samples = np.array(
        [
            abs(bilinear(curvature_proxy, p))
            for p in ray_a
        ]
    )

    if np.std(sep_acc) > EPS and np.std(curvature_samples) > EPS:
        corr = np.corrcoef(
            np.abs(sep_acc),
            curvature_samples,
        )[0, 1]
    else:
        corr = np.nan

    straight = np.array(
        [
            start + (i + 1) * 0.08 * vel
            for i in range(len(ray_a))
        ]
    )

    straight[:, 0] = np.clip(straight[:, 0], 2, N - 3)
    straight[:, 1] = np.clip(straight[:, 1], 2, N - 3)

    deviation = np.linalg.norm(ray_a - straight, axis=1)

    path_length = np.cumsum(
        np.r_[0.0, np.linalg.norm(np.diff(ray_a, axis=0), axis=1)]
    )

    normalized_deviation = deviation / (1.0 + path_length)

    ray_df = pd.DataFrame(
        {
            "step": np.arange(len(ray_a)),
            "ray_x": ray_a[:, 0],
            "ray_y": ray_a[:, 1],
            "nearby_ray_x": ray_b[:, 0],
            "nearby_ray_y": ray_b[:, 1],
            "separation": separation,
            "separation_acceleration": sep_acc,
            "curvature_proxy": curvature_samples,
            "normalized_deviation": normalized_deviation,
        }
    )

    return ray_a, ray_b, ray_df, float(corr), float(np.mean(normalized_deviation))


def classify(
    metric_signature_valid,
    propagation_speed,
    propagation_corr,
    min_speed,
    max_speed,
    curvature_corr,
    mean_norm_deviation,
):
    finite_speed_valid = (
        np.isfinite(propagation_speed)
        and min_speed <= propagation_speed <= max_speed
        and propagation_corr >= 0.95
    )

    curvature_valid = (
        np.isfinite(curvature_corr)
        and abs(curvature_corr) >= 0.25
    )

    geometry_valid = mean_norm_deviation < 0.05

    if metric_signature_valid and finite_speed_valid and curvature_valid and geometry_valid:
        return "emergent_relativistic_reconstruction_supported"

    if metric_signature_valid and finite_speed_valid and geometry_valid:
        return "weak_effective_relativistic_geometry"

    return "relativistic_reconstruction_not_supported"


def main():
    print("\n=== BST EMERGENT RELATIVISTIC RECONSTRUCTION TEST ===")

    rho, refractive_index, effective_speed, curvature_proxy = build_closure_geometry()

    metric_signature_valid = bool(np.all(effective_speed > 0.0))

    final_wave, arrivals, propagation_speed, propagation_corr = finite_propagation_test(
        effective_speed
    )

    ray_a, ray_b, ray_df, curvature_corr, mean_norm_deviation = curvature_deviation_test(
        refractive_index,
        curvature_proxy,
    )

    min_speed = float(effective_speed.min())
    max_speed = float(effective_speed.max())

    verdict = classify(
        metric_signature_valid,
        propagation_speed,
        propagation_corr,
        min_speed,
        max_speed,
        curvature_corr,
        mean_norm_deviation,
    )

    summary = pd.DataFrame(
        [
            {
                "metric_signature_valid": metric_signature_valid,
                "min_effective_speed": min_speed,
                "max_effective_speed": max_speed,
                "measured_propagation_speed": propagation_speed,
                "propagation_fit_corr": propagation_corr,
                "curvature_deviation_corr": curvature_corr,
                "mean_normalized_curvature_deviation": mean_norm_deviation,
                "verdict": verdict,
            }
        ]
    )

    summary.to_csv(
        OUT / "relativistic_reconstruction_summary.csv",
        index=False,
    )

    arrivals.to_csv(
        OUT / "propagation_arrivals.csv",
        index=False,
    )

    ray_df.to_csv(
        OUT / "ray_deviation_timeseries.csv",
        index=False,
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(final_wave, origin="lower")
    plt.title("Finite wave propagation field")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(
        OUT / "relativistic_wave_field.png",
        dpi=220,
    )
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(curvature_proxy, origin="lower")
    plt.title("Effective curvature proxy")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(
        OUT / "relativistic_curvature_map.png",
        dpi=220,
    )
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(refractive_index, origin="lower")
    plt.plot(ray_a[:, 0], ray_a[:, 1], label="ray")
    plt.plot(ray_b[:, 0], ray_b[:, 1], "--", label="nearby ray")
    plt.title("Curvature-dependent ray deviation")
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(
        OUT / "relativistic_ray_deviation.png",
        dpi=220,
    )
    plt.close()

    print(summary.to_string(index=False))
    print(f"\n[OK] wrote {OUT / 'relativistic_reconstruction_summary.csv'}")
    print(f"[OK] wrote {OUT / 'propagation_arrivals.csv'}")
    print(f"[OK] wrote {OUT / 'ray_deviation_timeseries.csv'}")
    print(f"[OK] wrote {OUT / 'relativistic_wave_field.png'}")
    print(f"[OK] wrote {OUT / 'relativistic_curvature_map.png'}")
    print(f"[OK] wrote {OUT / 'relativistic_ray_deviation.png'}")
    print("[DONE] relativistic reconstruction test complete")


if __name__ == "__main__":
    main()