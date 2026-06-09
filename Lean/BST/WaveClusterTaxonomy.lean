import BST.RelativisticReconstruction

namespace BST

universe u

/--
A wave cluster.
-/
structure WaveCluster (L : Type u) where
  state : L

/--
A stable wave cluster.
-/
structure StableWaveCluster (L : Type u) where
  cluster : WaveCluster L
  stable : Prop

/--
Admissible deformation between stable clusters.
-/
class WaveDeformation
    (L : Type u) where
  deformable :
    StableWaveCluster L →
    StableWaveCluster L →
    Prop

  refl :
    ∀ c, deformable c c

  symm :
    ∀ c₁ c₂,
      deformable c₁ c₂ →
      deformable c₂ c₁

  trans :
    ∀ c₁ c₂ c₃,
      deformable c₁ c₂ →
      deformable c₂ c₃ →
      deformable c₁ c₃

/--
Two stable clusters belong to the same family
iff they are admissibly deformable.
-/
def SameWaveFamily
    {L : Type u}
    [WaveDeformation L]
    (c₁ c₂ : StableWaveCluster L) : Prop :=
  WaveDeformation.deformable c₁ c₂

theorem wave_family_refl
    {L : Type u}
    [WaveDeformation L]
    (c : StableWaveCluster L) :
    SameWaveFamily c c := by
  exact WaveDeformation.refl c

theorem wave_family_symm
    {L : Type u}
    [WaveDeformation L]
    (c₁ c₂ : StableWaveCluster L)
    (h : SameWaveFamily c₁ c₂) :
    SameWaveFamily c₂ c₁ := by
  exact WaveDeformation.symm c₁ c₂ h

theorem wave_family_trans
    {L : Type u}
    [WaveDeformation L]
    (c₁ c₂ c₃ : StableWaveCluster L)
    (h₁ : SameWaveFamily c₁ c₂)
    (h₂ : SameWaveFamily c₂ c₃) :
    SameWaveFamily c₁ c₃ := by
  exact WaveDeformation.trans c₁ c₂ c₃ h₁ h₂

theorem wave_family_equivalence
    {L : Type u}
    [WaveDeformation L] :
    (∀ c : StableWaveCluster L, SameWaveFamily c c) ∧
    (∀ c₁ c₂ : StableWaveCluster L,
      SameWaveFamily c₁ c₂ →
      SameWaveFamily c₂ c₁) ∧
    (∀ c₁ c₂ c₃ : StableWaveCluster L,
      SameWaveFamily c₁ c₂ →
      SameWaveFamily c₂ c₃ →
      SameWaveFamily c₁ c₃) := by
  constructor
  · intro c
    exact wave_family_refl c
  constructor
  · intro c₁ c₂ h
    exact wave_family_symm c₁ c₂ h
  · intro c₁ c₂ c₃ h₁ h₂
    exact wave_family_trans c₁ c₂ c₃ h₁ h₂

end BST