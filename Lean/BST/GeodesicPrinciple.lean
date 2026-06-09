import BST.GaugeStructure

namespace BST

universe u

/--

A path through closure states.

-/

def Path (L : Type u) :=

  Nat → ClosureState L

/--

Compatibility of a path with a local constraint.

-/

def Compatible

    {L : Type u}

    (Constraint : ClosureState L → Prop)

    (γ : Path L) : Prop :=

  ∀ n, Constraint (γ n)

/--

A geodesic is a path satisfying the active constraint.

-/

def Geodesic

    {L : Type u}

    (Constraint : ClosureState L → Prop)

    (γ : Path L) : Prop :=

  Compatible Constraint γ

theorem geodesic_is_compatible

    {L : Type u}

    (Constraint : ClosureState L → Prop)

    (γ : Path L)

    (hGeo : Geodesic Constraint γ) :

    Compatible Constraint γ := by

  exact hGeo

theorem compatible_path_is_geodesic

    {L : Type u}

    (Constraint : ClosureState L → Prop)

    (γ : Path L)

    (hComp : Compatible Constraint γ) :

    Geodesic Constraint γ := by

  exact hComp

theorem geodesic_principle

    {L : Type u}

    (Constraint : ClosureState L → Prop)

    (γ : Path L) :

    Geodesic Constraint γ ↔ Compatible Constraint γ := by

  constructor

  · intro h

    exact h

  · intro h

    exact h

end BST