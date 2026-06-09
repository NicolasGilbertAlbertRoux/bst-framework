import BST.ClosureEquilibrium

namespace BST

universe u v

/--

An observable projection from closure states.

-/

def ObservableProjection

    (L : Type u)

    (O : Type v) :=

  ClosureState L → O

/--

An observable structure emerges if a closure state

possesses a well-defined observable image.

-/

def ObservableEmergent

    {L : Type u}

    {O : Type v}

    (P : ObservableProjection L O)

    (c : ClosureState L) : Prop :=

  ∃ o : O, P c = o

/--

Observable Emergence Theorem.

Every observable projection produces an observable

image for any closure state.

-/

theorem observable_emergence

    {L : Type u}

    {O : Type v}

    (P : ObservableProjection L O)

    (c : ClosureState L) :

    ObservableEmergent P c := by

  refine ⟨P c, rfl⟩

/--

Equilibrium observables remain unchanged under

an evolution-compatible projection.

-/

theorem equilibrium_observable_stability

    {L : Type u}

    {O : Type v}

    (Φ : ClosureEvolution L)

    (P : ObservableProjection L O)

    (c : ClosureState L)

    (hCompat : ∀ x, P (Φ x) = P x) :

    P (Φ c) = P c := by

  exact hCompat c

end BST