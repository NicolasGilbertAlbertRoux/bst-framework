import BST.ClosurePersistence

namespace BST

universe u

/--

Closure invariance.

A property is invariant under an evolution operator if

every closed state remains closed after evolution.

-/

def Invariant

    {L : Type u}

    (P : ClosureState L → Prop)

    (Φ : ClosureEvolution L) : Prop :=

  ∀ c, P c ↔ P (Φ c)

/--

Invariance implies persistence.

-/

theorem invariance_implies_persistence

    {L : Type u}

    (P : ClosureState L → Prop)

    (Φ : ClosureEvolution L)

    (hInv : Invariant P Φ) :

    Persistent P Φ := by

  intro c hClosed

  exact (hInv c).mp hClosed

/--

Closure Invariance Theorem.

If closure is invariant under the evolution operator,

then closure is preserved by evolution.

-/

theorem closure_invariance

    {L : Type u}

    (P : ClosureState L → Prop)

    (Φ : ClosureEvolution L)

    (hInv : Invariant P Φ)

    (c : ClosureState L)

    (hClosed : P c) :

    P (Φ c) := by

  exact (hInv c).mp hClosed

/--

Backward closure recovery.

Under invariance, closure after evolution implies

closure before evolution.

-/

theorem closure_recovery

    {L : Type u}

    (P : ClosureState L → Prop)

    (Φ : ClosureEvolution L)

    (hInv : Invariant P Φ)

    (c : ClosureState L)

    (hClosed : P (Φ c)) :

    P c := by

  exact (hInv c).mpr hClosed

end BST