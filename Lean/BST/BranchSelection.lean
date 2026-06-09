namespace BST

universe u

structure BranchSystem (L : Type u) where

  admissible : L → Prop

  score : L → Nat

def IsSelected

    {L : Type u}

    (B : BranchSystem L)

    (γstar : L) : Prop :=

  B.admissible γstar ∧

  ∀ γ : L,

    B.admissible γ →

    B.score γ ≤ B.score γstar

theorem selected_branch_is_admissible

    {L : Type u}

    (B : BranchSystem L)

    (γstar : L)

    (hSel : IsSelected B γstar) :

    B.admissible γstar := by

  exact hSel.left

theorem selected_branch_is_optimal

    {L : Type u}

    (B : BranchSystem L)

    (γstar : L)

    (hSel : IsSelected B γstar) :

    ∀ γ : L,

      B.admissible γ →

      B.score γ ≤ B.score γstar := by

  exact hSel.right

theorem branch_selection_consistency

    {L : Type u}

    (B : BranchSystem L)

    (γstar : L)

    (hSel : IsSelected B γstar) :

    B.admissible γstar := by

  exact selected_branch_is_admissible B γstar hSel

end BST