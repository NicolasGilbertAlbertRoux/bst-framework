# BST Lean Formalization

This repository contains a Lean 4 formalization of the BST theorem framework.

## Purpose

The goal of this project is not to prove the physical validity of BST.

Instead, the objective is to formally verify that the theorem chain can be derived rigorously from the mathematical definitions adopted in the framework.

The formalization currently contains:

- 27 verified Lean modules
- 0 compilation errors
- 0 warnings
- 0 sorry
- No additional axioms

## Requirements

- Lean 4
- Elan
- Lake

Recommended installation:

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

Verify installation:

```bash
elan --version
lake --version
```

## Building the Project

From the repository root:

```bash
lake build
```

A successful verification should return:

```bash
Build completed successfully
```

## Project Structure

The formalization covers the main conceptual chain of BST:

- Latent Equivalence
- Resolution Monotonicity
- Hierarchical Refinement
- Multi-Observable Reconstruction
- Observer-Dependent Reconstruction
- Generative Memory
- Arbitration Consistency
- Branch Selection
- Closure Persistence
- Closure Invariance
- Closure Equilibrium
- Observable Emergence
- Closure Reconstruction
- Closure Equivalence
- Gauge Structure
- Geodesic Principle
- Relativistic Reconstruction
- Wave Cluster Taxonomy
- Orbit Structure
- Periodic Attractors
- Matter Taxonomy
- Topological Selection
- Contact Networks
- Universality

## Scope

The Lean formalization verifies logical consistency.

It does not establish:

- physical correctness,
- empirical validity,
- numerical accuracy,
- experimental confirmation.

Those aspects remain the subject of simulation, experimentation, and peer review.

## License

Same license as the parent BST framework repository.