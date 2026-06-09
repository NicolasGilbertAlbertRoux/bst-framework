/-
BST Formalization — Observer-Dependent Reconstruction

An observer is represented by a reconstruction operator from latent states
to an observer-specific observable space.

The observer induces an equivalence relation by equality of reconstructions.
This file formalizes the quotient reconstruction and the possibility that
two observers induce different equivalence relations.
-/

import BST.LatentEquivalence

namespace BST

universe u v

/-- An observer-dependent reconstruction operator. -/
structure Observer (L : Type u) where
  O : Type v
  R : L → O

/-- Observer-induced equivalence relation on latent states. -/
def ObserverEquiv {L : Type u} (ω : Observer L) (a b : L) : Prop :=
  ω.R a = ω.R b

/-- Observer-induced setoid. -/
def observerSetoid {L : Type u} (ω : Observer L) : Setoid L where
  r := fun a b => ObserverEquiv ω a b
  iseqv := by
    constructor
    · intro a
      rfl
    · intro a b h
      exact h.symm
    · intro a b c hab hbc
      exact hab.trans hbc

/-- Observer reconstruction class. -/
def observerClass {L : Type u} (ω : Observer L) (a : L) :
    Quotient (observerSetoid ω) :=
  Quotient.mk (observerSetoid ω) a

/--
Observer-dependent reconstruction.

A surjective observer reconstruction operator induces a bijective canonical
quotient-to-observable map.
-/
theorem observer_dependent_reconstruction
    {L : Type u}
    (ω : Observer L)
    (hSurj : Function.Surjective ω.R) :
    Bijective (quotientToObservable ω.R) := by
  exact observable_reconstruction_bijective ω.R hSurj

/--
Observer relativity of equivalence.

If two observers disagree on whether a pair of latent states has the same
observable reconstruction, then their induced equivalence relations differ
on that pair.
-/
theorem observer_equivalence_can_differ
    {L : Type u}
    (ω₁ ω₂ : Observer L)
    (a b : L)
    (h₁ : ObserverEquiv ω₁ a b)
    (h₂ : ¬ ObserverEquiv ω₂ a b) :
    ObserverEquiv ω₁ a b ∧ ¬ ObserverEquiv ω₂ a b := by
  exact ⟨h₁, h₂⟩

end BST
