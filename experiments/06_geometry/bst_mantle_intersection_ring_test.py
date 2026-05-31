#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BST / OMM-SBT — Diffuse Mantle Intersection Ring Test
=====================================================

Question tested
---------------
If two primordial oscillatory diffuse mantles meet, do their equal-threshold
contact regions naturally form point-like contacts or annular loop-like
intersections?

This is NOT a string-theory derivation.
It tests the geometric/oscillatory premise:

    two diffuse oscillatory wave mantles
        -> threshold shells / coherent mantles
        -> intersections
        -> point contact or ring-like loop depending on separation/phase/threshold

Main outputs
------------
results/research_final/bst_mantle_intersection_ring_test/
    summary.md
    mantle_intersection_scores.csv
    best_ring_case.png
    ring_radius_over_time.png
    ring_mode_spectrum.png

Run
---
python bst_mantle_intersection_ring_test.py --fast

or full:
python bst_mantle_intersection_ring_test.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTDIR = Path("results/research_final/bst_mantle_intersection_ring_test")


def log(msg):
    print(msg, flush=True)


def make_grid(n=120, lim=4.0):
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    z = np.linspace(-lim, lim, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


def mantle_field(X, Y, Z, center, phase, k=3.0, sigma=2.0):
    cx, cy, cz = center
    R = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)
    envelope = np.exp(-(R**2)/(2*sigma**2))
    carrier = np.cos(k*R + phase)
    # positive coherent mantle intensity; no imposed ring
    return envelope * carrier


def intersection_points(F1, F2, X, Y, Z, threshold, tol):
    """
    Approximate intersection of two threshold mantles:
        F1 ≈ threshold and F2 ≈ threshold
    """
    mask = (np.abs(F1-threshold) < tol) & (np.abs(F2-threshold) < tol)
    pts = np.column_stack([X[mask], Y[mask], Z[mask]])
    return pts


def ring_metrics(points):
    """
    Given 3D intersection points, test whether they behave like a ring.
    Since two equal spheres centered on x-axis intersect in a plane x≈0,
    we estimate best plane by PCA and measure radial closure in that plane.
    """
    if len(points) < 40:
        return dict(n=len(points), ring_score=0.0, mean_radius=np.nan,
                    radius_cv=np.nan, angular_coverage=0.0,
                    plane_thickness=np.nan, dominant_mode=np.nan,
                    spectral_entropy=np.nan)

    C = points.mean(axis=0)
    P = points - C

    # PCA: ring plane = two largest variance directions; normal = smallest
    cov = np.cov(P.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    u = vecs[:, 0]
    v = vecs[:, 1]
    n = vecs[:, 2]

    a = P @ u
    b = P @ v
    h = P @ n

    r = np.sqrt(a*a + b*b)
    mean_r = float(np.mean(r))
    std_r = float(np.std(r))
    radius_cv = std_r / (mean_r + 1e-12)
    plane_thickness = float(np.std(h) / (mean_r + 1e-12))

    theta = np.arctan2(b, a)
    bins = np.linspace(-np.pi, np.pi, 73)
    counts, _ = np.histogram(theta, bins=bins)
    angular_coverage = float(np.count_nonzero(counts) / (len(bins)-1))

    # Angular radius profile
    rb = []
    th_centers = []
    for i in range(len(bins)-1):
        m = (theta >= bins[i]) & (theta < bins[i+1])
        if np.any(m):
            rb.append(np.median(r[m]))
            th_centers.append(0.5*(bins[i]+bins[i+1]))
        else:
            rb.append(np.nan)
            th_centers.append(0.5*(bins[i]+bins[i+1]))
    rb = np.array(rb)
    idx = np.arange(len(rb))
    good = ~np.isnan(rb)
    if np.sum(good) > 8:
        ext_i = np.r_[idx[good], idx[good]+len(rb)]
        ext_v = np.r_[rb[good], rb[good]]
        rb2 = np.interp(idx, ext_i, ext_v)
        sig = rb2 - np.mean(rb2)
        spec = np.abs(np.fft.rfft(sig))
        specn = spec/(np.sum(spec)+1e-12)
        dominant = int(np.argmax(specn[1:])+1) if len(specn)>1 else 0
        entropy = float(-np.sum(specn*np.log(specn+1e-12)))
    else:
        dominant = np.nan
        entropy = np.nan

    # ring score: broad angular coverage, thin plane, stable radius, nonzero radius
    radius_factor = np.tanh(mean_r/0.25)
    radial_regular = 1/(1+radius_cv)
    planar = 1/(1+plane_thickness)
    score = angular_coverage * radial_regular * planar * radius_factor

    return dict(n=int(len(points)), ring_score=float(score), mean_radius=mean_r,
                radius_cv=float(radius_cv), angular_coverage=angular_coverage,
                plane_thickness=plane_thickness, dominant_mode=dominant,
                spectral_entropy=entropy)


def scan(fast=False):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    n = 90 if fast else 130
    x, y, z, X, Y, Z = make_grid(n=n, lim=4.0)

    seps = np.linspace(0.8, 4.6, 14 if fast else 28)
    thresholds = np.linspace(-0.30, 0.55, 10 if fast else 18)
    phases = np.linspace(0, 2*np.pi, 8 if fast else 20, endpoint=False)
    ks = [2.2, 3.0, 3.8] if fast else [1.8, 2.4, 3.0, 3.6, 4.2]
    sigmas = [1.5, 2.0, 2.5] if fast else [1.3, 1.7, 2.1, 2.6]

    rows = []
    best = None
    total = len(seps)*len(thresholds)*len(phases)*len(ks)*len(sigmas)
    c = 0

    for sep in seps:
        for k in ks:
            for sigma in sigmas:
                for phase in phases:
                    F1 = mantle_field(X, Y, Z, (-sep/2,0,0), phase=phase, k=k, sigma=sigma)
                    F2 = mantle_field(X, Y, Z, ( sep/2,0,0), phase=-phase, k=k, sigma=sigma)
                    scale = max(np.std(F1), np.std(F2), 1e-9)

                    for threshold in thresholds:
                        tol = 0.055 * scale
                        pts = intersection_points(F1, F2, X, Y, Z, threshold, tol)
                        m = ring_metrics(pts)

                        row = dict(sep=float(sep), k=float(k), sigma=float(sigma),
                                   phase=float(phase), threshold=float(threshold),
                                   tol=float(tol), **m)
                        rows.append(row)

                        if best is None or row["ring_score"] > best["ring_score"]:
                            best = row | {"points": pts.copy()}

                    c += len(thresholds)
                    if c % 1000 == 0:
                        log(f"[SCAN] {c}/{total} threshold cases")

    df = pd.DataFrame([{k:v for k,v in r.items() if k!="points"} for r in rows])
    df = df.sort_values("ring_score", ascending=False)
    df.to_csv(OUTDIR/"mantle_intersection_scores.csv", index=False)

    plot_best(best)
    time_modulation(best)
    write_summary(df)

    return df


def plot_best(best):
    pts = best["points"]
    if len(pts) == 0:
        return

    C = pts.mean(axis=0)
    P = pts - C
    cov = np.cov(P.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    u, v, n = vecs[:,0], vecs[:,1], vecs[:,2]
    a = P @ u
    b = P @ v
    h = P @ n

    fig = plt.figure(figsize=(11,4))

    ax1 = fig.add_subplot(1,2,1, projection="3d")
    ax1.scatter(pts[:,0], pts[:,1], pts[:,2], s=2, alpha=0.5)
    ax1.set_title("Best 3D mantle-threshold intersection")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(a, b, s=2, alpha=0.6)
    ax2.set_aspect("equal","box")
    ax2.set_title("Projection onto best-fit intersection plane")
    ax2.set_xlabel("plane axis 1")
    ax2.set_ylabel("plane axis 2")

    fig.suptitle(
        f"score={best['ring_score']:.3f}, sep={best['sep']:.2f}, "
        f"k={best['k']:.2f}, sigma={best['sigma']:.2f}, threshold={best['threshold']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(OUTDIR/"best_ring_case.png", dpi=180)
    plt.close(fig)

    # spectrum for best case
    r = np.sqrt(a*a+b*b)
    th = np.arctan2(b,a)
    bins = np.linspace(-np.pi,np.pi,73)
    rb=[]
    for i in range(len(bins)-1):
        mask=(th>=bins[i])&(th<bins[i+1])
        rb.append(np.median(r[mask]) if np.any(mask) else np.nan)
    rb=np.array(rb)
    idx=np.arange(len(rb)); good=~np.isnan(rb)
    if np.sum(good)>8:
        ext_i=np.r_[idx[good],idx[good]+len(rb)]
        ext_v=np.r_[rb[good],rb[good]]
        rb2=np.interp(idx,ext_i,ext_v)
        sig=rb2-np.mean(rb2)
        spec=np.abs(np.fft.rfft(sig))
        spec=spec/(np.sum(spec)+1e-12)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(np.arange(len(spec)),spec)
        ax.set_title("Angular modulation spectrum of best mantle-intersection ring")
        ax.set_xlabel("mode number")
        ax.set_ylabel("normalized amplitude")
        fig.tight_layout()
        fig.savefig(OUTDIR/"ring_mode_spectrum.png",dpi=180)
        plt.close(fig)


def time_modulation(best):
    sep=float(best["sep"]); k=float(best["k"]); sigma=float(best["sigma"]); threshold=float(best["threshold"])
    n=90
    x,y,z,X,Y,Z=make_grid(n=n,lim=4.0)
    phases=np.linspace(0,2*np.pi,48,endpoint=False)
    rows=[]
    for ph in phases:
        F1=mantle_field(X,Y,Z,(-sep/2,0,0),phase=ph,k=k,sigma=sigma)
        F2=mantle_field(X,Y,Z,( sep/2,0,0),phase=-ph,k=k,sigma=sigma)
        scale=max(np.std(F1),np.std(F2),1e-9)
        pts=intersection_points(F1,F2,X,Y,Z,threshold,0.055*scale)
        m=ring_metrics(pts)
        rows.append(dict(phase=float(ph), **m))
    df=pd.DataFrame(rows)
    df.to_csv(OUTDIR/"ring_radius_over_time.csv",index=False)

    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(df["phase"],df["mean_radius"],marker="o",label="mean radius")
    ax2=ax.twinx()
    ax2.plot(df["phase"],df["ring_score"],marker="x",linestyle="--",label="ring score")
    ax.set_xlabel("relative beating phase")
    ax.set_ylabel("mean ring radius")
    ax2.set_ylabel("ring score")
    ax.set_title("Ring modulation across beating phase")
    fig.tight_layout()
    fig.savefig(OUTDIR/"ring_radius_over_time.png",dpi=180)
    plt.close(fig)


def write_summary(df):
    b=df.iloc[0]
    verdict="positive geometric intersection signal" if b["ring_score"]>0.55 and b["angular_coverage"]>0.65 else "weak or partial signal"
    text=f"""# BST Mantle Intersection Ring Test

This test directly examines intersections of two diffuse oscillatory primordial mantles.

Best case:
- ring_score: {b['ring_score']:.4f}
- n points: {int(b['n'])}
- mean_radius: {b['mean_radius']:.4f}
- radius_cv: {b['radius_cv']:.4f}
- angular_coverage: {b['angular_coverage']:.4f}
- plane_thickness: {b['plane_thickness']:.4f}
- dominant_mode: {b['dominant_mode']}
- spectral_entropy: {b['spectral_entropy']:.4f}
- sep: {b['sep']:.4f}
- k: {b['k']:.4f}
- sigma: {b['sigma']:.4f}
- threshold: {b['threshold']:.4f}

Verdict: **{verdict}**

Interpretation:
- This test is closer to the user's proposed geometry than the previous projection scanners.
- It checks whether two diffuse oscillatory mantles produce point-like contact or loop-like threshold intersections.
- A positive result supports the geometrical plausibility of annular contact motifs.
- It still does not imply equivalence with string theory, M-theory, Polyakov action, or Nambu-Goto action.

Next step if retained:
- Compare the phase-dependent ring-radius spectrum with idealized closed-loop vibration spectra.
- Only then discuss a cautious string-inspired analogy in an appendix.
"""
    (OUTDIR/"summary.md").write_text(text,encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fast",action="store_true")
    args=ap.parse_args()
    OUTDIR.mkdir(parents=True,exist_ok=True)
    log("=== BST DIFFUSE MANTLE INTERSECTION RING TEST ===")
    log(f"[OUT] {OUTDIR}")
    df=scan(fast=args.fast)
    b=df.iloc[0]
    log(f"[BEST] score={b['ring_score']:.4f}, coverage={b['angular_coverage']:.3f}, radius_cv={b['radius_cv']:.3f}")
    log("[DONE]")


if __name__=="__main__":
    main()
