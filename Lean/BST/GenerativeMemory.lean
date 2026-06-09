namespace BST

universe u

def History (L : Type u) := List L

def ContinuationSet (L : Type u) := L → Prop

def GenerativeMemory (L : Type u) :=
History L → ContinuationSet L → ContinuationSet L

theorem generative_memory_constraint
{L : Type u}
(M : GenerativeMemory L)
(H : History L)
(C : ContinuationSet L)
(hSubset : ∀ γ : L, M H C γ → C γ) :
∀ γ : L, M H C γ → C γ := by
intro γ hγ
exact hSubset γ hγ

theorem generative_memory_reduces_possibilities
{L : Type u}
(M : GenerativeMemory L)
(H : History L)
(C : ContinuationSet L)
(hSubset : ∀ γ : L, M H C γ → C γ) :
∀ γ : L, M H C γ → C γ := by
exact generative_memory_constraint M H C hSubset

end BST