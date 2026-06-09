# Beating Substrate Theory (BST)

Beating Substrate Theory (BST) is an experimental computational framework exploring the emergence of observable structures from a discrete oscillatory substrate.

This repository contains the reference numerical experiments developed within the BST research program, together with the supporting infrastructure required for reproducibility and further investigation.

The repository focuses on computational reconstruction and experimental exploration. Interpretative, theoretical, and correspondence studies are maintained separately from the framework itself.

## Repository Structure

experiments/   Numerical experiments and reconstruction studies

results/       Generated outputs and validation artefacts

scripts/       Reproduction and utility scripts

historical/    Historical research stages and precursor documents

## Experimental Scope

The framework currently includes experiments investigating:

- Closure dynamics
- Latent projection and observable equivalence
- Reconstruction and generative memory
- Arbitration and branch selection
- Multi-resolution (Zeta) dynamics
- Emergent geometry
- Interaction and matter reconstruction
- Vacuum-wave backgrounds
- Structure formation
- Curvature and gravity-like emergence
- Robustness and falsifiability analyses

Experiments are organized progressively.

The repository includes foundational experiments (00–19), intermediate reconstruction studies (20–56), and higher-level reconstruction and synthesis experiments (57–71).

Not all experiments are executed by the default reproduction script.

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Individual experiments can be executed directly:

```bash
python experiments/01_closure/canonical_closure_law_test.py
```

Some higher-level experiments are included for completeness and reproducibility purposes and may require substantial computation time.

A repository-wide reproduction utility is also provided:

```bash
python scripts/reproduce.py
```

The default reproduction script currently executes the foundational experiment suite used for framework validation.

Additional experiments, including higher-level reconstruction studies and synthesis milestones, are available in the experiments/ directory and may be executed individually when required.

## Historical Research

The historical/ directory contains archived research documents and intermediate development stages that contributed to the evolution of the BST framework.

These materials are preserved for transparency and traceability.

## Reproducibility

The primary purpose of this repository is reproducibility.

All scientific claims associated with BST should be evaluated through the experiments, generated outputs, validation procedures, and formal verification resources associated with the framework.

Numerical experiments, reconstruction studies, and independent formal verification are provided to facilitate transparent evaluation and replication of results.

## Formal Verification (Lean 4)

In addition to the numerical experiments contained in this repository, a Lean 4 formalization of the BST theorem framework has been developed.

The purpose of the formalization is not to establish the physical validity of BST, but to verify that the formalized theorem chain follows rigorously from the adopted definitions and assumptions.

Current verification status:

- 27 verified Lean modules
- 0 compilation errors
- 0 warnings
- 0 sorry
- No additional axioms introduced

The Lean project provides an independent logical consistency check that complements the experimental and computational results contained in this repository.

The formalization currently covers major components of the BST framework, including reconstruction, closure dynamics, observable equivalence, arbitration, branch selection, emergent geometry, wave-cluster taxonomy, periodic attractors, matter taxonomy, topological selection, contact networks, and universality.

The Lean formalization is maintained separately from the experimental framework and may be consulted independently by reviewers interested in formal verification.

## Citation

If you use this repository in academic work, please cite the project using the information provided in CITATION.cff.

## License

This repository is distributed under the license provided in LICENSE.
