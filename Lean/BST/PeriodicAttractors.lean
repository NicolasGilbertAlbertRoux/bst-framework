import BST.OrbitStructure

namespace BST

universe u

def StableOrbit
    {L : Type u}
    (Stability : Orbit L → Prop)
    (γ : Orbit L) : Prop :=
  Stability γ

def PeriodicAttractor
    {L : Type u}
    (Stability : Orbit L → Prop)
    (γ : Orbit L) : Prop :=
  Periodic γ ∧ StableOrbit Stability γ

theorem periodic_attractor_is_periodic
    {L : Type u}
    (Stability : Orbit L → Prop)
    (γ : Orbit L)
    (hAttr : PeriodicAttractor Stability γ) :
    Periodic γ := by
  exact hAttr.left

theorem periodic_attractor_is_stable
    {L : Type u}
    (Stability : Orbit L → Prop)
    (γ : Orbit L)
    (hAttr : PeriodicAttractor Stability γ) :
    StableOrbit Stability γ := by
  exact hAttr.right

theorem periodic_attractor_is_recurrent
    {L : Type u}
    (Stability : Orbit L → Prop)
    (γ : Orbit L)
    (hAttr : PeriodicAttractor Stability γ) :
    Recurrent γ := by
  exact periodic_implies_recurrent γ hAttr.left

end BST