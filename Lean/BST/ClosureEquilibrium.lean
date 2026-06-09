import BST.ClosureInvariance

namespace BST

universe u

/--
A closure equilibrium is a fixed point of the evolution.
-/
def Equilibrium
    {L : Type u}
    (Φ : ClosureEvolution L)
    (c : ClosureState L) : Prop :=
  Φ c = c

/--
An equilibrium remains invariant under evolution.
-/
theorem equilibrium_invariant
    {L : Type u}
    (Φ : ClosureEvolution L)
    (c : ClosureState L)
    (hEq : Equilibrium Φ c) :
    Φ c = c := by
  exact hEq

/--
Repeated evolution preserves equilibrium.
-/
theorem equilibrium_persistent
    {L : Type u}
    (Φ : ClosureEvolution L)
    (c : ClosureState L)
    (hEq : Equilibrium Φ c) :
    Φ (Φ c) = c := by
  unfold Equilibrium at hEq
  rw [hEq]
  exact hEq

/--
Closure Equilibrium Theorem.

If a closure state is an equilibrium,
then the first and second iterates coincide.
-/
theorem closure_equilibrium
    {L : Type u}
    (Φ : ClosureEvolution L)
    (c : ClosureState L)
    (hEq : Equilibrium Φ c) :
    Φ (Φ c) = Φ c := by
  unfold Equilibrium at hEq
  rw [hEq]
  exact hEq

end BST