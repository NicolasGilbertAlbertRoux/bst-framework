import BST.ClosureEquivalence

namespace BST

universe u v

def GaugeTransform
    (L : Type u) :=
  ClosureState L → ClosureState L

def GaugePreserving
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (G : GaugeTransform L) : Prop :=
  ∀ c : ClosureState L, P (G c) = P c

theorem gauge_preserving_implies_closure_equivalence
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (G : GaugeTransform L)
    (hG : GaugePreserving P G)
    (c : ClosureState L) :
    ClosureEquivalent P (G c) c := by
  exact hG c

theorem gauge_preserving_is_observable_identity
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (G : GaugeTransform L)
    (hG : GaugePreserving P G)
    (c : ClosureState L) :
    P (G c) = P c := by
  exact hG c

theorem gauge_equivalent_states_are_observationally_indistinguishable
    {L : Type u}
    {O : Type v}
    (P : ObservableProjection L O)
    (c₁ c₂ : ClosureState L)
    (hEq : ClosureEquivalent P c₁ c₂) :
    P c₁ = P c₂ := by
  exact hEq

end BST