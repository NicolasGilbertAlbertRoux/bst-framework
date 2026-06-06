#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST — Transverse Closure Equilibrium Dynamics Test
==================================================

Purpose
-------
This experiment probes whether an effective transverse-closure ratio behaves as:

1. a strictly conserved quantity,
2. a bounded equilibrium regime,
3. an attractor-like relaxation toward kappa ~= 1.

The test is intentionally conservative.

It does not assume exact conservation of transverse closure. Instead, it
measures whether future organization remains dynamically balanced with past
organization across the canonical R_lambda / phase / breath signals.

Interpretation
--------------
kappa(t) ~= 1 means that the local future organization has approximately the
same norm as the local past organization.

Large fluctuations exclude strict conservation.

A stable tail around kappa ~= 1 supports bounded closure equilibrium.

Input
-----
results/research_final/Rlambda_functional_role_test/
    Rlambda_functional_role_timeseries.csv

Outputs
-------
results/research_final/transverse_closure_dynamics_test/
    transverse_closure_timeseries.csv
    transverse_closure_summary.csv
    transverse_closure_equilibrium.png
    transverse_closure_distance.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("results/research_final/Rlambda_functional_role_test")
OUT = Path("results/research_final/transverse_closure_dynamics_test")
OUT.mkdir(parents=True, exist_ok=True)

INPUT = BASE / "Rlambda_functional_role_timeseries.csv"

REQUIRED_COLUMNS = ["R_lambda", "phase_error", "breath_error"]

WINDOW = 24
EPS = 1e-9
KAPPA_STAR = 1.0


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {INPUT}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_columns(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds = np.where(stds < EPS, 1.0, stds)
    return (values - means) / stds


def compute_closure_equilibrium(
    signals: np.ndarray,
    window: int = WINDOW,
) -> pd.DataFrame:
    rows = []

    for step in range(window, len(signals) - window):
        past = signals[step - window:step]
        future = signals[step:step + window]

        past_mean = past.mean(axis=0)
        future_mean = future.mean(axis=0)

        past_norm = float(np.linalg.norm(past_mean))
        future_norm = float(np.linalg.norm(future_mean))

        kappa = future_norm / (past_norm + EPS)
        distance = abs(kappa - KAPPA_STAR)

        rows.append({
            "step": step,
            "closure_kappa": kappa,
            "distance_to_equilibrium": distance,
            "past_norm": past_norm,
            "future_norm": future_norm,
        })

    if not rows:
        raise RuntimeError(
            f"Not enough samples for WINDOW={window}. "
            f"Need at least {2 * window + 1} rows."
        )

    return pd.DataFrame(rows)


def summarize(timeseries: pd.DataFrame) -> pd.DataFrame:
    n = len(timeseries)
    q = max(1, int(0.2 * n))

    early = timeseries.iloc[:q]
    tail = timeseries.iloc[-q:]

    mean_kappa = float(timeseries["closure_kappa"].mean())
    std_kappa = float(timeseries["closure_kappa"].std())
    tail_mean = float(tail["closure_kappa"].mean())
    tail_std = float(tail["closure_kappa"].std())

    mean_distance = float(timeseries["distance_to_equilibrium"].mean())
    early_distance = float(early["distance_to_equilibrium"].mean())
    tail_distance = float(tail["distance_to_equilibrium"].mean())
    reduction_ratio = float(early_distance / (tail_distance + EPS))

    min_kappa = float(timeseries["closure_kappa"].min())
    max_kappa = float(timeseries["closure_kappa"].max())

    if std_kappa < 1e-3 and abs(mean_kappa - KAPPA_STAR) < 1e-3:
        verdict = "strong_conservation"
    elif reduction_ratio > 1.5 and tail_distance < early_distance:
        verdict = "attractor_like_relaxation"
    elif abs(tail_mean - KAPPA_STAR) < 0.25:
        verdict = "bounded_closure_equilibrium"
    else:
        verdict = "undetermined"

    return pd.DataFrame([{
        "mean_kappa": mean_kappa,
        "std_kappa": std_kappa,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "min_kappa": min_kappa,
        "max_kappa": max_kappa,
        "mean_distance_to_equilibrium": mean_distance,
        "early_distance_to_equilibrium": early_distance,
        "tail_distance_to_equilibrium": tail_distance,
        "distance_reduction_ratio": reduction_ratio,
        "kappa_star": KAPPA_STAR,
        "window": WINDOW,
        "verdict": verdict,
    }])


def plot_outputs(timeseries: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(timeseries["step"], timeseries["closure_kappa"], label=r"$\kappa(t)$")
    plt.axhline(KAPPA_STAR, linestyle="--", label=r"$\kappa^\star=1$")
    plt.xlabel("step")
    plt.ylabel("closure equilibrium ratio")
    plt.title("Transverse closure equilibrium dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "transverse_closure_equilibrium.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        timeseries["step"],
        timeseries["distance_to_equilibrium"],
        label=r"$|\kappa(t)-1|$",
    )
    plt.xlabel("step")
    plt.ylabel("distance to closure equilibrium")
    plt.title("Distance to transverse closure equilibrium")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "transverse_closure_distance.png", dpi=220)
    plt.close()


def main() -> None:
    print("\n=== BST TRANSVERSE CLOSURE EQUILIBRIUM DYNAMICS TEST ===")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}\n"
            "Run experiments/00_prerequisites/generate_canonical_inputs.py first."
        )

    df = pd.read_csv(INPUT)
    require_columns(df, REQUIRED_COLUMNS)

    raw_signals = df[REQUIRED_COLUMNS].to_numpy(dtype=float)
    signals = normalize_columns(raw_signals)

    timeseries = compute_closure_equilibrium(signals, WINDOW)
    timeseries_path = OUT / "transverse_closure_timeseries.csv"
    timeseries.to_csv(timeseries_path, index=False)

    summary = summarize(timeseries)
    summary_path = OUT / "transverse_closure_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_outputs(timeseries)

    print(summary.to_string(index=False))
    print(f"\n[OK] wrote {timeseries_path}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {OUT / 'transverse_closure_equilibrium.png'}")
    print(f"[OK] wrote {OUT / 'transverse_closure_distance.png'}")
    print("[DONE] transverse closure equilibrium dynamics test complete")


if __name__ == "__main__":
    main()