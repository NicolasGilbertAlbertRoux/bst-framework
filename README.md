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

All scientific claims associated with BST should be evaluated through the experiments, generated outputs, and validation procedures contained within the framework.

## Citation

If you use this repository in academic work, please cite the project using the information provided in CITATION.cff.

## License

This repository is distributed under the license provided in LICENSE.
