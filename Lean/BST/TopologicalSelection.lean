import BST.MatterTaxonomy

namespace BST

universe u

/--
A topological signature attached to a matter species.
-/
def TopologyClassifier
    (L : Type u)
    (T : Type u) :=
  MatterSpecies L → T

/--
Two matter species belong to the same topological family
if they share the same topological signature.
-/
def SameTopologicalFamily
    {L : Type u}
    {T : Type u}
    (F : TopologyClassifier L T)
    (m₁ m₂ : MatterSpecies L) : Prop :=
  F m₁ = F m₂

theorem same_topological_family_refl
    {L : Type u}
    {T : Type u}
    (F : TopologyClassifier L T)
    (m : MatterSpecies L) :
    SameTopologicalFamily F m m := by
  rfl

theorem same_topological_family_symm
    {L : Type u}
    {T : Type u}
    (F : TopologyClassifier L T)
    (m₁ m₂ : MatterSpecies L)
    (h : SameTopologicalFamily F m₁ m₂) :
    SameTopologicalFamily F m₂ m₁ := by
  exact h.symm

theorem same_topological_family_trans
    {L : Type u}
    {T : Type u}
    (F : TopologyClassifier L T)
    (m₁ m₂ m₃ : MatterSpecies L)
    (h₁ : SameTopologicalFamily F m₁ m₂)
    (h₂ : SameTopologicalFamily F m₂ m₃) :
    SameTopologicalFamily F m₁ m₃ := by
  exact h₁.trans h₂

/--
Topological selection induces an equivalence relation.
-/
theorem topological_selection_equivalence
    {L : Type u}
    {T : Type u}
    (F : TopologyClassifier L T) :
    (∀ m : MatterSpecies L,
      SameTopologicalFamily F m m) ∧
    (∀ m₁ m₂ : MatterSpecies L,
      SameTopologicalFamily F m₁ m₂ →
      SameTopologicalFamily F m₂ m₁) ∧
    (∀ m₁ m₂ m₃ : MatterSpecies L,
      SameTopologicalFamily F m₁ m₂ →
      SameTopologicalFamily F m₂ m₃ →
      SameTopologicalFamily F m₁ m₃) := by
  constructor
  · intro m
    exact same_topological_family_refl F m
  constructor
  · intro m₁ m₂ h
    exact same_topological_family_symm F m₁ m₂ h
  · intro m₁ m₂ m₃ h₁ h₂
    exact same_topological_family_trans F m₁ m₂ m₃ h₁ h₂

end BST