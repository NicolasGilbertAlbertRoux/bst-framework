#!/usr/bin/env python3
"""Generate frozen canonical prerequisite datasets for BST validation.

These files close the dependency graph for the curated experiments. They are not
claimed as new physics results; they are deterministic upstream fixtures matching
the historical prerequisite families documented in the v7 audit.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/research_final")
ROOT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
N = 640
t = np.arange(N)

# 1) R_lambda functional role timeseries
R = 0.62 + 0.14*np.sin(2*np.pi*t/96) + 0.06*np.sin(2*np.pi*t/27) + 0.015*rng.normal(size=N)
phase = 0.22 + 0.08*np.sin(2*np.pi*t/64 + .7) + 0.012*rng.normal(size=N)
breath = 0.18 + 0.07*np.cos(2*np.pi*t/80 + .3) + 0.012*rng.normal(size=N)
Rdot = np.gradient(R)
out = ROOT/"Rlambda_functional_role_test"; out.mkdir(exist_ok=True)
pd.DataFrame({"t":t,"R_lambda":R,"Rdot":Rdot,"phase_error":phase,"breath_error":breath}).to_csv(out/"Rlambda_functional_role_timeseries.csv", index=False)

# 2) Branch resolution / coexistence events
idxs = np.arange(48, N-48, 12)
branching = 0.45 + 0.25*np.sin(2*np.pi*idxs/144)**2 + 0.04*rng.normal(size=len(idxs))
silhouette = 0.55 + 0.20*np.cos(2*np.pi*idxs/120) + 0.03*rng.normal(size=len(idxs))
cluster_dist = 0.30 + 0.25*np.abs(np.sin(2*np.pi*idxs/96)) + 0.03*rng.normal(size=len(idxs))
taxa = np.where(branching>0.62, "bifurcating", np.where(silhouette>0.62, "coherent", "mixed"))
out = ROOT/"branch_resolution_vs_coexistence_test"; out.mkdir(exist_ok=True)
pd.DataFrame({"index":idxs,"branching_ratio":branching.clip(0,1),"silhouette":silhouette.clip(0,1),"cluster_distance":cluster_dist.clip(0,None),"branch_taxon":taxa}).to_csv(out/"branch_resolution_events.csv", index=False)

# 3) Mode hierarchy correspondence
out = ROOT/"mode_hierarchy_correspondence_test"; out.mkdir(exist_ok=True)
modes = np.where((idxs//24)%3==0, "closure", np.where((idxs//24)%3==1, "projection", "breathing"))
pd.DataFrame({"index":idxs,"predicted_mode":modes,"branch_taxon":taxa,"mode_confidence":(0.55+0.25*np.sin(idxs/37)**2)}).to_csv(out/"mode_hierarchy_events.csv", index=False)

# 4) Delayed synchronization attractor
out = ROOT/"delayed_synchronization_attractor_test"; out.mkdir(exist_ok=True)
phase_lag = 0.35*np.sin(2*np.pi*t/88+.2)+0.03*rng.normal(size=N)
breath_lag = 0.30*np.cos(2*np.pi*t/104-.1)+0.03*rng.normal(size=N)
pd.DataFrame({"t":t,"R_lambda":R,"Rdot":Rdot,"phase_lag":phase_lag,"breath_lag":breath_lag}).to_csv(out/"delayed_synchronization_timeseries.csv", index=False)

# 5) Hidden breathing modes
out = ROOT/"hidden_breathing_modes_test"; out.mkdir(exist_ok=True)
m1=np.sin(2*np.pi*t/74)+.05*rng.normal(size=N)
m2=np.cos(2*np.pi*t/113)+.05*rng.normal(size=N)
m3=np.sin(2*np.pi*t/41+.4)*np.cos(2*np.pi*t/170)+.05*rng.normal(size=N)
pd.DataFrame({"t":t,"mode1":m1,"mode2":m2,"mode3":m3,"mode4":0.5*m1-0.25*m2+0.05*rng.normal(size=N)}).to_csv(out/"hidden_modes_timeseries.csv", index=False)

# 6) Zeta global transverse closure sectors
out = ROOT/"zeta_global_transverse_closure_test"; out.mkdir(exist_ok=True)
clusters=np.arange(8)
closure=0.35+0.45*np.exp(-clusters/8)+0.03*rng.normal(size=len(clusters))
horizon=1.0+0.25*clusters+0.02*rng.normal(size=len(clusters))
expansion=1e-6*(1+0.2*np.sin(clusters))
breath=0.4+0.08*np.cos(clusters)
pd.DataFrame({"multisector_cluster":clusters,"sector_transverse_closure":closure.clip(0,1),"sector_horizon":horizon.clip(.1,None),"expansion_factor":expansion.clip(1e-9,None),"cosmic_breath":breath.clip(.01,None)}).to_csv(out/"zeta_global_transverse_closure_sectors.csv", index=False)

# 7) Zeta physical law recovery events
out = ROOT/"zeta_physical_law_recovery_test"; out.mkdir(exist_ok=True)
M=120
ia=np.arange(M); ib=ia+1
qstate=np.where(ia%3==0,"bound",np.where(ia%3==1,"transition","scattering"))
pd.DataFrame({
 "index_a":ia,"index_b":ib,
 "mode_transition":np.where(ia%2==0,"closure->projection","projection->breathing"),
 "quantum_state":qstate,
 "relativity_candidate":0.55+0.35*np.sin(ia/11)**2,
 "pseudo_interval":0.08*np.cos(ia/9),
 "quantization_distance":0.15+0.12*np.sin(ia/7)**2,
 "symmetry_invariant_2":1.0+0.08*np.cos(ia/13),
 "proto_constant":0.7+0.05*np.sin(ia/17)
}).to_csv(out/"zeta_physical_law_recovery_events.csv", index=False)

# 8) Zeta emergent causal cone events
out = ROOT/"zeta_emergent_causal_cone_test"; out.mkdir(exist_ok=True)
pd.DataFrame({
 "index_a":ia,"index_b":ib,
 "mode_transition":np.where(ia%2==0,"closure->projection","projection->breathing"),
 "cluster_transition":np.where(ia%4<2,"intra","inter"),
 "causal_region":np.where(np.abs(np.cos(ia/10))>.5,"inside","boundary"),
 "pseudo_interval":0.12*np.cos(ia/10),
 "cone_opening":0.5+0.3*np.sin(ia/15)**2,
 "local_causal_speed":0.8+0.1*np.cos(ia/16)
}).to_csv(out/"zeta_emergent_causal_cone_events.csv", index=False)

# 9) Source CSVs for string-bridge scanner
out = ROOT/"string_bridge_inputs"; out.mkdir(exist_ok=True)
P=200
z1=rng.normal(size=P); z2=rng.normal(size=P)
br=0.4+0.5*np.exp(-((np.sqrt(z1*z1+z2*z2)-1.1)**2)/0.12)+0.02*rng.normal(size=P)
pd.DataFrame({"z1":z1,"z2":z2,"branching_ratio":br.clip(0,1),"high_branching":br>0.65}).to_csv(out/"branching_zone_projection.csv", index=False)
pd.DataFrame({"center_label":[0,1],"z1_center":[0.0,0.8],"z2_center":[0.0,-0.5]}).to_csv(out/"branching_zone_centers.csv", index=False)
pd.DataFrame({"from_mode":["a","b","c","a"]*20,"to_mode":["b","c","a","a"]*20,"energy_delta":rng.normal(0,.1,80),"compression_delta":rng.normal(0,.1,80),"to_closure":rng.uniform(.4,.9,80),"from_closure":rng.uniform(.4,.9,80)}).to_csv(out/"local_36_1_microloop_events.csv", index=False)
pd.DataFrame({"transition_distance":rng.uniform(.05,.8,80),"closure_a":rng.uniform(.4,.95,80),"closure_b":rng.uniform(.4,.95,80)}).to_csv(out/"zeta_intersector_transition_candidates.csv", index=False)

print("[OK] generated canonical prerequisite datasets under results/research_final/")
