import BST.WaveClusterTaxonomy

namespace BST

universe u

/--
An orbit is a discrete trajectory through states.
-/
def Orbit (L : Type u) :=
  Nat → L

/--
A recurrent orbit revisits a previous state.
-/
def Recurrent
    {L : Type u}
    (γ : Orbit L) : Prop :=
  ∃ n m : Nat,
    n < m ∧ γ n = γ m

/--
A periodic orbit possesses a positive period.
-/
def Periodic
    {L : Type u}
    (γ : Orbit L) : Prop :=
  ∃ p : Nat,
    p > 0 ∧
    ∀ n : Nat,
      γ (n + p) = γ n

/--
Every periodic orbit is recurrent.
-/
theorem periodic_implies_recurrent

    {L : Type u}

    (γ : Orbit L)

    (hPer : Periodic γ) :

    Recurrent γ := by

  rcases hPer with ⟨p, hp, hperiod⟩

  refine ⟨0, p, hp, ?_⟩

  simpa [Nat.zero_add] using (hperiod 0).symm
  
/--
A periodic orbit returns to its initial state
after one period.
-/
theorem periodic_returns
    {L : Type u}
    (γ : Orbit L)
    (hPer : Periodic γ) :
    ∃ p : Nat,
      p > 0 ∧ γ p = γ 0 := by
  rcases hPer with ⟨p, hp, hperiod⟩
  refine ⟨p, hp, ?_⟩
  simpa using (hperiod 0)

end BST