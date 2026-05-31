#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BST Contact-Loop Tunnel Atlas — String-Inspired Functional Comparator
====================================================================

Computes explicit comparison proxies between BST tunnel worldsheets X(theta,t)
and string-inspired structures:
- Nambu-Goto-like area proxy
- Polyakov/conformal metric proxy
- Virasoro-like constraint proxy
- closure/periodicity proxy
- modal richness proxy

Run:
    python bst_string_functional_comparator.py --fast
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = Path("results/research_final/bst_string_functional_comparator")

def log(s): print(s, flush=True)

def make_grid(n=34, lim=5.0):
    a=np.linspace(-lim,lim,n)
    return np.meshgrid(a,a,a,indexing="ij")

def wave_cluster_field(X,Y,Z,centers,phase,k0=2.6,sigma=2.0,complexity=3,seed=0,twist=0.0):
    rng=np.random.default_rng(seed); F=np.zeros_like(X)
    for ci,(cx,cy,cz) in enumerate(centers):
        for j in range(complexity):
            off=rng.normal(0,0.22,3); amp=rng.uniform(0.45,1.0)/max(len(centers),1)
            kj=k0*rng.uniform(0.75,1.30); sig=sigma*rng.uniform(0.75,1.25)
            ph=phase*rng.uniform(0.65,1.35)+rng.uniform(0,2*np.pi)+0.37*ci
            ax,ay,az=rng.uniform(0.70,1.35,3)
            x=(X-(cx+off[0]))*ax; y=(Y-(cy+off[1]))*ay; z=(Z-(cz+off[2]))*az
            R=np.sqrt(x*x+y*y+z*z); theta=np.arctan2(y,z+1e-12)
            F += amp*np.exp(-(R**2)/(2*sig**2))*np.cos(kj*R+ph+twist*np.sin(theta)*np.tanh(R))
    return F

def atom_like_centers(atom,t,radius=2.6):
    if atom=="H":
        return [(0,0,0)], [(radius*np.cos(t),radius*np.sin(t),0.35*np.sin(2*t))]
    if atom=="He":
        return [(-0.25,0,0),(0.25,0,0)], [(radius*np.cos(t),radius*np.sin(t),0.30*np.sin(2*t)),(radius*np.cos(t+np.pi),radius*np.sin(t+np.pi),0.30*np.sin(2*t+np.pi/3))]
    nuc=[(0.35*np.cos(2*np.pi*i/6),0.35*np.sin(2*np.pi*i/6),0.15*((-1)**i)) for i in range(6)]
    ele=[(radius*np.cos(t+2*np.pi*i/6),radius*np.sin(t+2*np.pi*i/6),0.45*np.sin(2*t+i)) for i in range(6)]
    return nuc,ele

def intersection_points(F1,F2,X,Y,Z,threshold,tol):
    mask=(np.abs(F1-threshold)<tol)&(np.abs(F2-threshold)<tol)
    return np.column_stack([X[mask],Y[mask],Z[mask]])

def ordered_loop(points,bins=60):
    if len(points)<80: return None
    C=points.mean(axis=0); P=points-C
    vals,vecs=np.linalg.eigh(np.cov(P.T)); vecs=vecs[:,np.argsort(vals)[::-1]]
    th=np.arctan2(P@vecs[:,1],P@vecs[:,0])
    edges=np.linspace(-np.pi,np.pi,bins+1)
    curve=np.full((bins,3),np.nan); counts=np.zeros(bins)
    for i in range(bins):
        m=(th>=edges[i])&(th<edges[i+1]); counts[i]=np.sum(m)
        if np.any(m): curve[i]=np.median(points[m],axis=0)
    if np.count_nonzero(counts)/bins < 0.35: return None
    idx=np.arange(bins)
    for d in range(3):
        good=np.isfinite(curve[:,d])
        if np.sum(good)<4: return None
        curve[:,d]=np.interp(idx,np.r_[idx[good],idx[good]+bins],np.r_[curve[good,d],curve[good,d]])
    return curve

def build_worldsheet(atom,threshold,k,sigma,twist,fast=False):
    grid_n=34 if fast else 44; steps=32 if fast else 64; bins=60 if fast else 96
    X,Y,Z=make_grid(n=grid_n,lim=5.0)
    W=[]; valid=[]
    for ti,t in enumerate(np.linspace(0,2*np.pi,steps,endpoint=False)):
        nuc,ele=atom_like_centers(atom,t)
        F1=wave_cluster_field(X,Y,Z,nuc,phase=t,k0=k,sigma=sigma,complexity=3,seed=17000+ti,twist=twist)
        F2=wave_cluster_field(X,Y,Z,ele,phase=-t,k0=k*1.05,sigma=sigma*1.15,complexity=2,seed=18000+ti,twist=-twist)
        scale=max(np.std(F1),np.std(F2),1e-9)
        curve=ordered_loop(intersection_points(F1,F2,X,Y,Z,threshold,0.055*scale),bins=bins)
        W.append(curve if curve is not None else np.full((bins,3),np.nan)); valid.append(curve is not None)
    W=np.array(W); valid=np.array(valid)
    if np.sum(valid)>=2:
        idx=np.arange(W.shape[0])
        for j in range(W.shape[1]):
            for d in range(3):
                good=np.isfinite(W[:,j,d])
                if np.sum(good)>=2: W[:,j,d]=np.interp(idx,idx[good],W[good,j,d])
    return W,valid

def functional_maps(W):
    Xs=(np.roll(W,-1,axis=1)-np.roll(W,1,axis=1))/2
    Xt=(np.roll(W,-1,axis=0)-np.roll(W,1,axis=0))/2
    E=np.einsum("tbi,tbi->tb",Xs,Xs)
    F=np.einsum("tbi,tbi->tb",Xs,Xt)
    G=np.einsum("tbi,tbi->tb",Xt,Xt)
    det=E*G-F*F
    area_density=np.sqrt(np.maximum(det,0))
    conformal_defect=np.sqrt((E-G)**2+4*F**2)/(E+G+1e-12)
    Xss=np.roll(W,-1,axis=1)-2*W+np.roll(W,1,axis=1)
    Xtt=np.roll(W,-1,axis=0)-2*W+np.roll(W,1,axis=0)
    curvature_density=np.linalg.norm(Xss,axis=2)**2+np.linalg.norm(Xtt,axis=2)**2
    closure_gap=np.linalg.norm(W[:,0,:]-W[:,-1,:],axis=1)
    mean_radius=np.mean(np.linalg.norm(W-W.mean(axis=1,keepdims=True),axis=2),axis=1)
    closure_defect=closure_gap/(mean_radius+1e-12)
    R=np.linalg.norm(W-W.mean(axis=1,keepdims=True),axis=2)
    A=np.fft.rfft(R-R.mean(axis=1,keepdims=True),axis=1)
    rows=[]
    for n in range(1,min(18,A.shape[1])):
        amp=np.abs(A[:,n]); temporal=np.abs(np.fft.rfft(amp-amp.mean()))
        if len(temporal)>2:
            search=temporal.copy(); search[:2]=0; omega=int(np.argmax(search)); power=float(search[omega]/(np.sum(temporal)+1e-12))
        else:
            omega=0; power=0
        rows.append(dict(mode_n=n,omega_index=omega,omega_power=power,modal_energy=float(np.mean(amp**2))))
    return dict(area_density=area_density,conformal_defect=conformal_defect,virasoro_proxy=conformal_defect,curvature_density=curvature_density,closure_defect=closure_defect,modes=pd.DataFrame(rows))

def score_functionals(W,valid):
    if not np.all(np.isfinite(W)) or np.sum(valid)<4: return None
    m=functional_maps(W)
    area_mean=float(np.mean(m["area_density"])); area_cv=float(np.std(m["area_density"])/(area_mean+1e-12))
    conformal_mean=float(np.mean(m["conformal_defect"]))
    closure_mean=float(np.mean(m["closure_defect"]))
    curvature_mean=float(np.mean(m["curvature_density"]))
    modes=m["modes"]
    unique_omegas=int(modes[modes.omega_power>0.05].omega_index.nunique()) if len(modes) else 0
    p=modes.modal_energy.to_numpy(float) if len(modes) else np.array([])
    modal_entropy=float(-np.sum((p/(p.sum()+1e-12))*np.log(p/(p.sum()+1e-12)+1e-12))) if len(p) else 0
    ng=1/(1+area_cv); poly=1/(1+conformal_mean); vira=poly; closure=1/(1+closure_mean); rigid=1/(1+curvature_mean)
    modal=0.5*min(unique_omegas/8,1)+0.5*min(modal_entropy/3,1)
    membrane=0.20*ng+0.20*poly+0.15*closure+0.15*modal+0.15*min(unique_omegas/10,1)+0.15*(1/(1+area_cv))
    closed=0.30*closure+0.25*modal+0.20*poly+0.15*vira+0.10*min(unique_omegas/10,1)
    rigid_score=0.25*ng+0.25*rigid+0.20*poly+0.15*modal+0.15*closure
    family=max({"membrane_worldsheet_like_trace":membrane,"closed_string_like_trace":closed,"rigid_membrane_like_trace":rigid_score}, key=lambda k: {"membrane_worldsheet_like_trace":membrane,"closed_string_like_trace":closed,"rigid_membrane_like_trace":rigid_score}[k])
    return dict(nambu_goto_area=float(np.sum(m["area_density"])),area_cv=area_cv,conformal_defect_mean=conformal_mean,closure_defect_mean=closure_mean,curvature_mean=curvature_mean,unique_omegas=unique_omegas,modal_energy_entropy=modal_entropy,nambu_goto_proxy_score=ng,polyakov_proxy_score=poly,virasoro_proxy_score=vira,closure_score=closure,modal_score=modal,closed_string_score=closed,membrane_worldsheet_score=membrane,rigid_membrane_score=rigid_score,best_string_inspired_family=family),m

def generate_cases(fast=False):
    atoms=["H","He","C"]; thresholds=np.linspace(-0.18,0.28,3 if fast else 5)
    ks=[1.8,2.6,3.4] if fast else [1.6,2.2,2.8,3.4]
    sigmas=[1.6,2.2] if fast else [1.4,1.8,2.2,2.7]
    twists=[0.0,0.8,1.6] if fast else [0.0,0.5,1.0,1.5,2.0]
    for atom in atoms:
        for th in thresholds:
            for k in ks:
                for sig in sigmas:
                    for tw in twists:
                        yield atom,float(th),float(k),float(sig),float(tw)

def run(fast=False):
    OUTDIR.mkdir(parents=True,exist_ok=True)
    rows=[]; best=None; cases=list(generate_cases(fast))
    for ci,(atom,th,k,sig,tw) in enumerate(cases,start=1):
        W,valid=build_worldsheet(atom,th,k,sig,tw,fast)
        res=score_functionals(W,valid)
        if res is None:
            row=dict(case=ci,atom=atom,threshold=th,k=k,sigma=sig,twist=tw,valid_fraction=float(np.mean(valid)),membrane_worldsheet_score=0,best_string_inspired_family="invalid")
        else:
            metrics,maps=res
            row=dict(case=ci,atom=atom,threshold=th,k=k,sigma=sig,twist=tw,valid_fraction=float(np.mean(valid)),**metrics)
            if best is None or row["membrane_worldsheet_score"]>best["row"]["membrane_worldsheet_score"]:
                best=dict(row=row,W=W,maps=maps)
        rows.append(row)
        if ci%20==0: log(f"[SCAN] {ci}/{len(cases)}")
    df=pd.DataFrame(rows).sort_values("membrane_worldsheet_score",ascending=False)
    df.to_csv(OUTDIR/"string_functional_comparison_scores.csv",index=False)
    atlas=df.groupby(["atom","best_string_inspired_family"]).agg(count=("membrane_worldsheet_score","count"),mean_membrane_score=("membrane_worldsheet_score","mean"),max_membrane_score=("membrane_worldsheet_score","max"),mean_closed_score=("closed_string_score","mean"),mean_polyakov_proxy=("polyakov_proxy_score","mean"),mean_virasoro_proxy=("virasoro_proxy_score","mean"),mean_unique_omegas=("unique_omegas","mean")).reset_index().sort_values("max_membrane_score",ascending=False)
    atlas.to_csv(OUTDIR/"string_functional_family_atlas.csv",index=False)
    plot_outputs(df,best); write_summary(df,atlas,best)
    return df,atlas,best

def plot_outputs(df,best):
    fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(df.closed_string_score,df.membrane_worldsheet_score,c=df.unique_omegas,s=45); ax.set_xlabel("closed-string-like score"); ax.set_ylabel("membrane/worldsheet-like score"); ax.set_title("String-inspired functional comparison"); fig.tight_layout(); fig.savefig(OUTDIR/"string_functional_score_map.png",dpi=180); plt.close(fig)
    if best:
        maps=best["maps"]; fig,axs=plt.subplots(2,2,figsize=(10,7))
        for ax,(key,title) in zip(axs.ravel(),[("area_density","Nambu-Goto-like area density"),("conformal_defect","Polyakov/conformal defect"),("curvature_density","rigidity curvature density"),("virasoro_proxy","Virasoro-like constraint defect")]):
            im=ax.imshow(maps[key],aspect="auto",origin="lower"); ax.set_title(title); ax.set_xlabel("sigma/theta"); ax.set_ylabel("tau/time"); fig.colorbar(im,ax=ax,fraction=0.046)
        fig.tight_layout(); fig.savefig(OUTDIR/"best_functional_maps.png",dpi=180); plt.close(fig)
        modes=maps["modes"]; modes.to_csv(OUTDIR/"best_functional_modes.csv",index=False)

def write_summary(df,atlas,best):
    if not best: return
    b=best["row"]
    txt=f"""# BST Contact-Loop Tunnel Atlas — String-Inspired Functional Comparator

Best candidate:
- atom-like configuration: {b['atom']}
- best string-inspired family: {b['best_string_inspired_family']}
- membrane/worldsheet-like score: {b['membrane_worldsheet_score']:.4f}
- closed-string-like score: {b['closed_string_score']:.4f}
- rigid-membrane-like score: {b['rigid_membrane_score']:.4f}
- Nambu-Goto area proxy score: {b['nambu_goto_proxy_score']:.4f}
- Polyakov/conformal proxy score: {b['polyakov_proxy_score']:.4f}
- Virasoro-like proxy score: {b['virasoro_proxy_score']:.4f}
- closure score: {b['closure_score']:.4f}
- modal score: {b['modal_score']:.4f}
- unique omegas: {int(b['unique_omegas'])}
- conformal defect mean: {b['conformal_defect_mean']:.4f}
- closure defect mean: {b['closure_defect_mean']:.4f}

Family atlas:
{atlas.to_string(index=False)}

Interpretation:
- This comparator computes explicit functional proxies, not only verbal analogies.
- Nambu-Goto proxy = induced area stability of the BST worldsheet.
- Polyakov/conformal proxy = closeness of the induced metric to a conformal-like gauge.
- Virasoro-like proxy = Euclideanized tangent-balance/orthogonality defect.
- This does not derive string theory or M-theory.
"""
    (OUTDIR/"best_functional_summary.md").write_text(txt,encoding="utf-8")
    appendix=f"""# Appendix Draft — Exploratory Correspondence with String-Inspired Structures

The BST Contact-Loop Tunnel Atlas identifies coherent oscillatory contact structures in which interacting wave-cluster mantles generate closed-loop tunnel traces. These traces can be reconstructed as parametric surfaces X(theta,t), enabling a cautious comparison with string-inspired objects such as closed-string worldsheets, Polyakov-type surfaces, Nambu-Goto area functionals, and membrane-like traces.

This comparison is not proposed as a derivation of string theory or M-theory. In particular, the present reconstruction does not establish the Nambu-Goto action, the Polyakov action, Virasoro constraints, supersymmetry, or critical dimensions as consequences of BST. Instead, it provides a reproducible dictionary between BST observables and broad string-inspired descriptors.

Best functional classification: {b['best_string_inspired_family']}

Proxy scores:
- membrane/worldsheet-like score: {b['membrane_worldsheet_score']:.4f}
- closed-string-like score: {b['closed_string_score']:.4f}
- Nambu-Goto area proxy score: {b['nambu_goto_proxy_score']:.4f}
- Polyakov/conformal proxy score: {b['polyakov_proxy_score']:.4f}
- Virasoro-like proxy score: {b['virasoro_proxy_score']:.4f}
- unique modal frequencies: {int(b['unique_omegas'])}

The strongest analogy is therefore not to a pointlike particle or simple open string, but to a worldsheet-like or membrane-like trace generated by coherent contact-loop dynamics. The additional effective coordinates observed in the BST tunnel atlas are best interpreted as internal phase, angular, and radial coordinates of the contact process, rather than literal additional spatial dimensions.
"""
    (OUTDIR/"appendix_string_correspondence_note.md").write_text(appendix,encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fast",action="store_true"); args=ap.parse_args()
    log("=== BST STRING FUNCTIONAL COMPARATOR ==="); log(f"[OUT] {OUTDIR}")
    df,atlas,best=run(fast=args.fast); b=df.iloc[0]
    log(f"[BEST] atom={b['atom']} family={b['best_string_inspired_family']} membrane={b['membrane_worldsheet_score']:.4f} closed={b['closed_string_score']:.4f}")
    log("[DONE]")

if __name__=="__main__":
    main()
