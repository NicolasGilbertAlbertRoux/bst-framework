namespace BST

universe u

/--

A closure state is represented abstractly.

-/

structure ClosureState (L : Type u) where

  carrier : L

/--

A closure evolution operator.

-/

def ClosureEvolution (L : Type u) :=

  ClosureState L → ClosureState L

/--

Persistence of closure.

A closure state is persistent if one application of the

evolution preserves the closure property.

-/

def Persistent

    {L : Type u}

    (P : ClosureState L → Prop)

    (Φ : ClosureEvolution L) : Prop :=

  ∀ c, P c → P (Φ c)

/--

Closure Persistence Theorem.

If closure is preserved by the evolution operator,

then every closed state remains closed after evolution.

-/

theorem closure_persistence

    {L : Type u}

    (P : ClosureState L → Prop)

    (Φ : ClosureEvolution L)

    (hPersist : Persistent P Φ)

    (c : ClosureState L)

    (hClosed : P c) :

    P (Φ c) := by

  exact hPersist c hClosed

/--

Iterated persistence.

Closure remains preserved under repeated application

of the evolution operator.

-/

theorem closure_persistence_twice

    {L : Type u}

    (P : ClosureState L → Prop)

    (Φ : ClosureEvolution L)

    (hPersist : Persistent P Φ)

    (c : ClosureState L)

    (hClosed : P c) :

    P (Φ (Φ c)) := by

  apply hPersist

  exact hPersist c hClosed

end BST