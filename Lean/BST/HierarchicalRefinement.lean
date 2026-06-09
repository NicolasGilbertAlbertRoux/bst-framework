/-
BST Formalization — Hierarchical Refinement

This file formalizes the logical core of the hierarchical refinement theorem.

A finer resolution refines a coarser one when every equivalence class induced
by the finer projection is included in an equivalence class induced by the
coarser projection.

We avoid a full finite partition library here and encode the theorem as the
class-inclusion property on projection-induced equivalence classes.
-/

import BST.ResolutionMonotonicity

namespace BST

universe u v w x

/--
Class inclusion at two resolutions.

This predicate says that the equivalence class of `a` induced by `Pfine`
is included in the equivalence class of `a` induced by `Pcoarse`.
-/
def ClassIncluded
    {L : Type u} {Ofine : Type v} {Ocoarse : Type w}
    (Pfine : L → Ofine)
    (Pcoarse : L → Ocoarse) : Prop :=
  ∀ a x : L, ResolutionEquiv Pfine x a → ResolutionEquiv Pcoarse x a

/--
Two-level hierarchical refinement.

If a coarser projection factors through a finer projection, then every
finer equivalence class is included in the corresponding coarser
equivalence class.
-/
theorem hierarchical_refinement_two_level
    {L : Type u} {Ofine : Type v} {Ocoarse : Type w}
    (Pfine : L → Ofine)
    (Pcoarse : L → Ocoarse)
    (C : Ofine → Ocoarse)
    (hFactor : ∀ x : L, Pcoarse x = C (Pfine x)) :
    ClassIncluded Pfine Pcoarse := by
  intro a x hFine
  exact finer_class_subset_coarser Pfine Pcoarse C hFactor a x hFine

/--
Three-level hierarchical refinement.

If P₂ factors through P₁ and P₃ factors through P₂, then class inclusion
holds from P₁ to P₂ and from P₂ to P₃. This is the finite-chain version
used as a first Lean stepping stone toward arbitrary resolution chains.
-/
theorem hierarchical_refinement_three_level
    {L : Type u} {O₁ : Type v} {O₂ : Type w} {O₃ : Type x}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (P₃ : L → O₃)
    (C₁₂ : O₁ → O₂)
    (C₂₃ : O₂ → O₃)
    (h12 : ∀ x : L, P₂ x = C₁₂ (P₁ x))
    (h23 : ∀ x : L, P₃ x = C₂₃ (P₂ x)) :
    ClassIncluded P₁ P₂ ∧ ClassIncluded P₂ P₃ := by
  constructor
  · exact hierarchical_refinement_two_level P₁ P₂ C₁₂ h12
  · exact hierarchical_refinement_two_level P₂ P₃ C₂₃ h23

/--
Transitive class inclusion.

If classes for P₁ are included in classes for P₂, and classes for P₂ are
included in classes for P₃, then classes for P₁ are included in classes for P₃.
-/
theorem class_inclusion_trans
    {L : Type u} {O₁ : Type v} {O₂ : Type w} {O₃ : Type x}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (P₃ : L → O₃)
    (h12 : ClassIncluded P₁ P₂)
    (h23 : ClassIncluded P₂ P₃) :
    ClassIncluded P₁ P₃ := by
  intro a x hFine
  exact h23 a x (h12 a x hFine)

end BST
