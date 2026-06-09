namespace BST

universe u

def StructurallyConsistentSet (L : Type u) := L → Prop

theorem arbitration_consistency
{L : Type u}
(S : StructurallyConsistentSet L)
(γstar : L)
(hγ : S γstar) :
S γstar := by
exact hγ

theorem consistency_preservation
{L : Type u}
(S : StructurallyConsistentSet L)
(Candidate : L → Prop)
(γstar : L)
(hCandidatesConsistent : ∀ γ : L, Candidate γ → S γ)
(hSelected : Candidate γstar) :
S γstar := by
exact hCandidatesConsistent γstar hSelected

end BST