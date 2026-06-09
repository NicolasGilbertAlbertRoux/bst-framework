import BST.GeodesicPrinciple

namespace BST

universe u v

/--
Two observers may reconstruct different closure states.
-/
def ObserverReconstruction
    (L : Type u)
    (O : Type v) :=
  O → ClosureState L

/--
Relativistic agreement means equality of observables.
-/
def RelativisticAgreement
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L) : Prop :=
  P c₁ = P c₂

/--
If two reconstructed states have the same observable image,
they are observationally equivalent.
-/
theorem reconstruction_agreement_implies_equivalence
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L)
    (h : RelativisticAgreement P c₁ c₂) :
    ClosureEquivalent P c₁ c₂ := by
  exact h

/--
Conversely, equivalent reconstructions agree observationally.
-/
theorem equivalence_implies_reconstruction_agreement
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L)
    (h : ClosureEquivalent P c₁ c₂) :
    RelativisticAgreement P c₁ c₂ := by
  exact h

/--
Relativistic Reconstruction Theorem.

Observational equivalence and reconstruction agreement
are logically equivalent.
-/
theorem relativistic_reconstruction
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L) :
    RelativisticAgreement P c₁ c₂
      ↔
    ClosureEquivalent P c₁ c₂ := by
  constructor
  · intro h
    exact reconstruction_agreement_implies_equivalence P c₁ c₂ h
  · intro h
    exact equivalence_implies_reconstruction_agreement P c₁ c₂ h

end BST