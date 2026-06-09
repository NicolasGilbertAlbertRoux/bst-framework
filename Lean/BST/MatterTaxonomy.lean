import BST.PeriodicAttractors

namespace BST

universe u

/--
A matter species is represented by a periodic attractor.
-/
structure MatterSpecies (L : Type u) where
  orbit : Orbit L

/--
A classifier assigns a label to a matter species.
-/
def MatterClassifier
    (L : Type u)
    (C : Type u) :=
  MatterSpecies L → C

/--
Two matter species belong to the same class if they
receive the same classification.
-/
def SameMatterClass
    {L : Type u}
    {C : Type u}
    (F : MatterClassifier L C)
    (m₁ m₂ : MatterSpecies L) : Prop :=
  F m₁ = F m₂

theorem same_matter_class_refl
    {L : Type u}
    {C : Type u}
    (F : MatterClassifier L C)
    (m : MatterSpecies L) :
    SameMatterClass F m m := by
  rfl

theorem same_matter_class_symm
    {L : Type u}
    {C : Type u}
    (F : MatterClassifier L C)
    (m₁ m₂ : MatterSpecies L)
    (h : SameMatterClass F m₁ m₂) :
    SameMatterClass F m₂ m₁ := by
  exact h.symm

theorem same_matter_class_trans
    {L : Type u}
    {C : Type u}
    (F : MatterClassifier L C)
    (m₁ m₂ m₃ : MatterSpecies L)
    (h₁ : SameMatterClass F m₁ m₂)
    (h₂ : SameMatterClass F m₂ m₃) :
    SameMatterClass F m₁ m₃ := by
  exact h₁.trans h₂

/--
Matter taxonomy induces an equivalence relation.
-/
theorem matter_taxonomy_equivalence
    {L : Type u}
    {C : Type u}
    (F : MatterClassifier L C) :
    (∀ m : MatterSpecies L, SameMatterClass F m m) ∧
    (∀ m₁ m₂ : MatterSpecies L,
      SameMatterClass F m₁ m₂ →
      SameMatterClass F m₂ m₁) ∧
    (∀ m₁ m₂ m₃ : MatterSpecies L,
      SameMatterClass F m₁ m₂ →
      SameMatterClass F m₂ m₃ →
      SameMatterClass F m₁ m₃) := by
  constructor
  · intro m
    exact same_matter_class_refl F m
  constructor
  · intro m₁ m₂ h
    exact same_matter_class_symm F m₁ m₂ h
  · intro m₁ m₂ m₃ h₁ h₂
    exact same_matter_class_trans F m₁ m₂ m₃ h₁ h₂

end BST