/-
BST Formalization — Latent Equivalence and Observable Reconstruction

This file formalizes the foundational quotient idea:

  non-injective observation induces latent equivalence classes;
  observable states correspond to quotient classes when the projection is surjective.

This proves the logical core only. It does not assert any physical correspondence.
-/

namespace BST

universe u v

/-- A map is non-injective when two distinct latent states have the same image. -/
def NonInjective {L : Type u} {O : Type v} (P : L → O) : Prop :=
  ∃ a b : L, a ≠ b ∧ P a = P b

/-- Projection-induced resolution equivalence. -/
def ResolutionEquiv {L : Type u} {O : Type v} (P : L → O) (a b : L) : Prop :=
  P a = P b

/-- A local definition of bijectivity, avoiding any dependency on Mathlib. -/
def Bijective {A : Type u} {B : Type v} (f : A → B) : Prop :=
  Function.Injective f ∧ Function.Surjective f

/--
Latent Equivalence Under Resolution.

If an observable projection is non-injective, then two distinct latent states
are observationally equivalent.
-/
theorem latent_equivalence_under_resolution
    {L : Type u} {O : Type v}
    (P : L → O)
    (h : NonInjective P) :
    ∃ a b : L, a ≠ b ∧ ResolutionEquiv P a b := by
  exact h

/-- The setoid induced on latent states by an observable projection. -/
def projectionSetoid {L : Type u} {O : Type v} (P : L → O) : Setoid L where
  r := fun a b => P a = P b
  iseqv := by
    constructor
    · intro a
      rfl
    · intro a b h
      exact h.symm
    · intro a b c hab hbc
      exact hab.trans hbc

/-- The observable reconstruction class of a latent state. -/
def reconstructionClass {L : Type u} {O : Type v} (P : L → O) (a : L) :
    Quotient (projectionSetoid P) :=
  Quotient.mk (projectionSetoid P) a

/-- The canonical map from quotient classes to observable states. -/
def quotientToObservable {L : Type u} {O : Type v} (P : L → O) :
    Quotient (projectionSetoid P) → O :=
  Quotient.lift P (by
    intro a b h
    exact h)

/--
Surjectivity of the quotient-to-observable map.

If P is surjective, then every observable state is represented by at least
one projection-induced reconstruction class.
-/
theorem quotientToObservable_surjective
    {L : Type u} {O : Type v}
    (P : L → O)
    (hSurj : Function.Surjective P) :
    Function.Surjective (quotientToObservable P) := by
  intro o
  rcases hSurj o with ⟨a, ha⟩
  exists Quotient.mk (projectionSetoid P) a

/--
Injectivity of the quotient-to-observable map.

Two quotient classes with the same observable image are equal.
-/
theorem quotientToObservable_injective
    {L : Type u} {O : Type v}
    (P : L → O) :
    Function.Injective (quotientToObservable P) := by
  intro q₁ q₂ hEq
  induction q₁ using Quotient.inductionOn with
  | h a =>
    induction q₂ using Quotient.inductionOn with
    | h b =>
      apply Quotient.sound
      exact hEq

/--
Observable Reconstruction Classes.

If the observable projection is surjective, then the canonical quotient map
is bijective. This is the Lean-safe formulation of:

  O ≃ L / ~
-/
theorem observable_reconstruction_bijective
    {L : Type u} {O : Type v}
    (P : L → O)
    (hSurj : Function.Surjective P) :
    Bijective (quotientToObservable P) := by
  constructor
  · exact quotientToObservable_injective P
  · exact quotientToObservable_surjective P hSurj

end BST
