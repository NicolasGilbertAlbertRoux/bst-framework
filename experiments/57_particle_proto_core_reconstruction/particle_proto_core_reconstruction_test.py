#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BST 57 — Particle Proto-Core Reconstruction

Goal:
    Test, without invention, whether particle-like identities can be supported
    from archived BST wave-cluster / proto-core evidence.

Safe hierarchy:
    W -> C -> P/T -> matter

Strict rules:
    - four-wave proto-core may be confirmed as motif if archived;
    - electron/photon/neutrino/quark identities are NOT assumed;
    - identities are confirmed only if archive evidence exists;
    - otherwise they remain hypotheses for later physics reconstruction.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".")
OUT = ROOT / "results/research_final/particle_proto_core_reconstruction_test"
OUT.mkdir(parents=True, exist_ok=True)

DICT = ROOT / "results/research_final/wave_cluster_dictionary"
AUDIT = ROOT / "results/research_final/wave_cluster_archive_audit"
PROTO = ROOT / "results/research_final/proto_core_identification_test"


PARTICLE_TERMS = {
    "electron_like": ["electron", "electron-like", "electronic"],
    "photon_like": ["photon", "photon-like", "radiation", "light"],
    "neutrino_like": ["neutrino", "neutrino-like"],
    "quark_like": ["quark", "quark-like", "baryon", "hadron"],
}


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def scan_archive_terms():
    rows = []
    files = list(ROOT.glob("experiments/**/*.py")) + list(ROOT.glob("results/research_final/**/*.csv"))

    for particle, terms in PARTICLE_TERMS.items():
        hits = []
        for path in files:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            for term in terms:
                if term.lower() in txt:
                    hits.append(str(path))
                    break

        rows.append({
            "particle_candidate": particle,
            "archive_keyword_hits": len(hits),
            "evidence_files": " | ".join(hits[:20]),
            "identity_status": "archive_mentions_found" if hits else "not_found_in_archive",
        })

    return pd.DataFrame(rows)


def four_wave_status(proto_audit):
    if proto_audit.empty:
        return False, "proto_core_audit_missing"

    mask = proto_audit["claim"].astype(str).str.contains(
        "Four-wave compact proto-core", case=False, na=False
    )

    if not mask.any():
        return False, "four_wave_motif_not_found"

    status = str(proto_audit[mask].iloc[0].get("status", "unknown"))
    return status in {"source_code_evidence", "archived_result"}, status


def summarize_proto_core_metrics():
    contacts = read_csv(PROTO / "proto_core_contacts.csv")
    by_element = read_csv(PROTO / "proto_core_by_element.csv")
    summary = read_csv(PROTO / "proto_core_summary.csv")

    rows = []

    if not summary.empty:
        rows.append({
            "metric": "proto_core_summary_verdict",
            "value": str(summary.iloc[0].get("verdict", "unknown")),
            "status": "archived_result",
        })

    if not by_element.empty:
        for col in by_element.columns:
            if "recovery" in col.lower():
                rows.append({
                    "metric": col,
                    "value": float(by_element[col].mean()),
                    "status": "archived_result",
                })

    if not contacts.empty:
        core = contacts[contacts.get("role", "") == "core"] if "role" in contacts.columns else contacts
        for col in ["loop", "tunnel", "capacity", "stability", "recurrence", "coherence"]:
            if col in core.columns:
                rows.append({
                    "metric": f"core_mean_{col}",
                    "value": float(core[col].mean()),
                    "status": "archived_result",
                })

    return pd.DataFrame(rows)


def build_particle_candidate_table(term_scan, proto_audit, proto_metrics):
    four_wave_confirmed, four_wave_evidence = four_wave_status(proto_audit)

    rows = []

    for _, r in term_scan.iterrows():
        candidate = r["particle_candidate"]
        keyword_found = r["archive_keyword_hits"] > 0

        if candidate == "electron_like":
            motif_status = "four_wave_proto_core_confirmed" if four_wave_confirmed else "motif_unconfirmed"
            identity_confirmed = False
            safe_status = "candidate_supported_as_motif_only" if four_wave_confirmed else "unsupported"
            reason = (
                "Four-wave compact proto-core exists, but archive does not yet prove electron identity."
                if four_wave_confirmed else
                "No confirmed four-wave proto-core motif."
            )
        else:
            motif_status = "no_specific_proto_core_motif_confirmed"
            identity_confirmed = False
            safe_status = "archive_mentions_only" if keyword_found else "unsupported"
            reason = (
                "Archive mentions exist, but no direct particle identity reconstruction is proven."
                if keyword_found else
                "No direct archive evidence found for this particle-like identity."
            )

        rows.append({
            "particle_candidate": candidate,
            "motif_status": motif_status,
            "identity_confirmed": identity_confirmed,
            "safe_status": safe_status,
            "archive_keyword_hits": int(r["archive_keyword_hits"]),
            "evidence_basis": four_wave_evidence if candidate == "electron_like" else r["identity_status"],
            "safe_interpretation": reason,
            "next_required_test": "derive identity from W/C/P observables, not from naming",
        })

    return pd.DataFrame(rows)


def build_wave_particle_bridge(clusters, candidates):
    rows = []

    for _, c in clusters.iterrows():
        for _, p in candidates.iterrows():
            rows.append({
                "cluster_id": c["cluster_id"],
                "cluster_name": c["canonical_name"],
                "particle_candidate": p["particle_candidate"],
                "allowed_relation": "possible_motif_substrate",
                "evidence_level": "not_identity_proof",
                "identity_confirmed": False,
                "warning": "C-cluster may support later reconstruction, but particle identity is not assigned in 57.",
            })

    return pd.DataFrame(rows)


def plot_candidate_status(candidates):
    score_map = {
        "candidate_supported_as_motif_only": 0.5,
        "archive_mentions_only": 0.25,
        "unsupported": 0.0,
    }

    vals = [score_map.get(x, 0.0) for x in candidates["safe_status"]]

    plt.figure(figsize=(8, 4.5))
    plt.bar(candidates["particle_candidate"], vals)
    plt.ylim(0, 1)
    plt.ylabel("Evidence score, not identity proof")
    plt.title("BST 57 particle proto-core reconstruction status")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT / "particle_candidate_status.png", dpi=220)
    plt.close()


def main():
    print("\n=== BST 57 PARTICLE PROTO-CORE RECONSTRUCTION TEST ===\n")

    clusters = read_csv(DICT / "canonical_wave_clusters.csv")
    proto_audit = read_csv(AUDIT / "proto_core_audit.csv")

    if clusters.empty:
        raise FileNotFoundError("Missing canonical_wave_clusters.csv. Run 57b first.")
    if proto_audit.empty:
        raise FileNotFoundError("Missing proto_core_audit.csv. Run 57a first.")

    term_scan = scan_archive_terms()
    proto_metrics = summarize_proto_core_metrics()
    candidates = build_particle_candidate_table(term_scan, proto_audit, proto_metrics)
    bridge = build_wave_particle_bridge(clusters, candidates)

    four_wave_confirmed, _ = four_wave_status(proto_audit)
    identity_count = int(candidates["identity_confirmed"].sum())

    verdict = (
        "particle_proto_core_motif_supported_identity_unconfirmed"
        if four_wave_confirmed and identity_count == 0
        else "particle_proto_core_reconstruction_incomplete"
    )

    summary = pd.DataFrame([{
        "num_wave_clusters": len(clusters),
        "four_wave_proto_core_confirmed": four_wave_confirmed,
        "electron_identity_confirmed": False,
        "photon_identity_confirmed": False,
        "neutrino_identity_confirmed": False,
        "quark_identity_confirmed": False,
        "particle_identities_confirmed": identity_count,
        "safe_hierarchy": "W -> C -> P/T -> matter",
        "verdict": verdict,
    }])

    term_scan.to_csv(OUT / "particle_archive_term_scan.csv", index=False)
    proto_metrics.to_csv(OUT / "proto_core_metric_summary.csv", index=False)
    candidates.to_csv(OUT / "particle_proto_core_candidates.csv", index=False)
    bridge.to_csv(OUT / "wave_cluster_particle_bridge_non_identity.csv", index=False)
    summary.to_csv(OUT / "particle_proto_core_reconstruction_summary.csv", index=False)

    plot_candidate_status(candidates)

    print(summary.to_string(index=False))
    print("\nStrict conclusion:")
    print("  Four-wave proto-core motif is usable.")
    print("  Electron-like identity remains unconfirmed.")
    print("  Photon/neutrino/quark-like identities remain unconfirmed unless archive evidence proves them.")
    print("  Atlas rebuild should include this result explicitly.")

    print("\n[OK] wrote", OUT / "particle_archive_term_scan.csv")
    print("[OK] wrote", OUT / "proto_core_metric_summary.csv")
    print("[OK] wrote", OUT / "particle_proto_core_candidates.csv")
    print("[OK] wrote", OUT / "wave_cluster_particle_bridge_non_identity.csv")
    print("[OK] wrote", OUT / "particle_proto_core_reconstruction_summary.csv")
    print("[OK] wrote", OUT / "particle_candidate_status.png")
    print("[DONE] particle proto-core reconstruction test complete")


if __name__ == "__main__":
    main()