import BST.ContactNetworks

namespace BST

universe u

/--
An admissible perturbation of objects of type X.
-/
def AdmissiblePerturbation (X : Type u) :=
  X → X

/--
A property stable under all admissible perturbations.
-/
def StableUnderPerturbations
    {X : Type u}
    (P : X → Prop) : Prop :=
  ∀ (f : AdmissiblePerturbation X) (x : X),
    P x → P (f x)

/--
A universal property in the BST sense.
-/
def UniversalProperty
    {X : Type u}
    (P : X → Prop) : Prop :=
  StableUnderPerturbations P

theorem stable_implies_universal
    {X : Type u}
    (P : X → Prop)
    (h : StableUnderPerturbations P) :
    UniversalProperty P := by
  exact h

theorem universal_is_stable
    {X : Type u}
    (P : X → Prop)
    (h : UniversalProperty P) :
    StableUnderPerturbations P := by
  exact h

theorem universality_equivalence
    {X : Type u}
    (P : X → Prop) :
    UniversalProperty P ↔ StableUnderPerturbations P := by
  constructor
  · intro h
    exact h
  · intro h
    exact h

end BST