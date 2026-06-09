/-
BST Formalization — Multi-Observable Reconstruction

A single latent substrate may be equipped with several observable projections.
Each surjective projection induces its own quotient reconstruction.

This file proves the logical core only: every projection in a family has a
bijective canonical quotient-to-observable map, provided it is surjective.
-/

import BST.LatentEquivalence

namespace BST

universe u v

/--
A family of observable projections indexed by `I`.

For each observer / modality / resolution index `i`, `O i` is the associated
observable type and `P i` is the projection from latent states to that
observable type.
-/
structure ProjectionFamily (I : Type u) (L : Type v) where
  O : I → Type u
  P : (i : I) → L → O i

/--
Each surjective observable projection in a family induces a bijective
canonical reconstruction map from its quotient space to its observable space.
-/
theorem multi_observable_reconstruction
    {I : Type u} {L : Type v}
    (F : ProjectionFamily I L)
    (hSurj : ∀ i : I, Function.Surjective (F.P i)) :
    ∀ i : I, Bijective (quotientToObservable (F.P i)) := by
  intro i
  exact observable_reconstruction_bijective (F.P i) (hSurj i)

/--
Non-uniqueness at the level of equivalence relations.

If two projections disagree on a latent pair in equality status, then their
induced equivalence relations differ on that pair.
-/
theorem projections_can_induce_distinct_equivalence
    {L : Type u} {O₁ : Type v} {O₂ : Type v}
    (P₁ : L → O₁)
    (P₂ : L → O₂)
    (a b : L)
    (h₁ : ResolutionEquiv P₁ a b)
    (h₂ : ¬ ResolutionEquiv P₂ a b) :
    ResolutionEquiv P₁ a b ∧ ¬ ResolutionEquiv P₂ a b := by
  exact ⟨h₁, h₂⟩

end BST
