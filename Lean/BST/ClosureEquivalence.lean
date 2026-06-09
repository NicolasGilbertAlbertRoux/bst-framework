import BST.ClosureReconstruction

namespace BST

universe u v

/--
Closure equivalence is observable indistinguishability
under a closure projection.
-/
def ClosureEquivalent
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L) : Prop :=
  ClosureReconstruction P c₁ c₂

/--
Closure equivalence is exactly equality of observable projections.
-/
theorem closure_equivalence_iff_observable_equality
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L) :
    ClosureEquivalent P c₁ c₂ ↔ P c₁ = P c₂ := by
  constructor
  · intro h
    exact h
  · intro h
    exact h

/--
Closure equivalence is reflexive.
-/
theorem closure_equivalence_refl
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c : ClosureState L) :
    ClosureEquivalent P c c := by
  rfl

/--
Closure equivalence is symmetric.
-/
theorem closure_equivalence_symm
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L)
    (h : ClosureEquivalent P c₁ c₂) :
    ClosureEquivalent P c₂ c₁ := by
  exact h.symm

/--
Closure equivalence is transitive.
-/
theorem closure_equivalence_trans
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ c₃ : ClosureState L)
    (h₁ : ClosureEquivalent P c₁ c₂)
    (h₂ : ClosureEquivalent P c₂ c₃) :
    ClosureEquivalent P c₁ c₃ := by
  exact h₁.trans h₂

/--
Closure Equivalence Theorem.

Observable indistinguishability of closure states defines
an equivalence relation.
-/
theorem closure_equivalence_relation
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O) :
    (∀ c, ClosureEquivalent P c c) ∧
    (∀ c₁ c₂,
      ClosureEquivalent P c₁ c₂ →
      ClosureEquivalent P c₂ c₁) ∧
    (∀ c₁ c₂ c₃,
      ClosureEquivalent P c₁ c₂ →
      ClosureEquivalent P c₂ c₃ →
      ClosureEquivalent P c₁ c₃) := by
  constructor
  · intro c
    exact closure_equivalence_refl P c
  constructor
  · intro c₁ c₂ h
    exact closure_equivalence_symm P c₁ c₂ h
  · intro c₁ c₂ c₃ h₁ h₂
    exact closure_equivalence_trans P c₁ c₂ c₃ h₁ h₂

end BST