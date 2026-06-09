import BST.ObservableEmergence

namespace BST

universe u v

/--

A reconstruction relation induced by an observable projection.

-/

def ClosureReconstruction

    {L : Type u}

    {O : Type v}

    (P : ObservableProjection L O)

    (c₁ c₂ : ClosureState L) : Prop :=

  P c₁ = P c₂

/--

Reflexivity of closure reconstruction.

-/

theorem closure_reconstruction_refl

    {L : Type u}

    {O : Type v}

    (P : ObservableProjection L O)

    (c : ClosureState L) :

    ClosureReconstruction P c c := by

  rfl

/--

Symmetry of closure reconstruction.

-/

theorem closure_reconstruction_symm

    {L : Type u}

    {O : Type v}

    (P : ObservableProjection L O)

    (c₁ c₂ : ClosureState L)

    (h : ClosureReconstruction P c₁ c₂) :

    ClosureReconstruction P c₂ c₁ := by

  exact h.symm

/--

Transitivity of closure reconstruction.

-/

theorem closure_reconstruction_trans

    {L : Type u}

    {O : Type v}

    (P : ObservableProjection L O)

    (c₁ c₂ c₃ : ClosureState L)

    (h₁ : ClosureReconstruction P c₁ c₂)

    (h₂ : ClosureReconstruction P c₂ c₃) :

    ClosureReconstruction P c₁ c₃ := by

  exact h₁.trans h₂

/--

Closure Reconstruction Theorem.

Observable indistinguishability defines an equivalence relation.

-/

theorem closure_reconstruction_equivalence

    {L : Type u}

    {O : Type v}

    (P : ObservableProjection L O) :

    (∀ c, ClosureReconstruction P c c) ∧

    (∀ c₁ c₂,

      ClosureReconstruction P c₁ c₂ →

      ClosureReconstruction P c₂ c₁) ∧

    (∀ c₁ c₂ c₃,

      ClosureReconstruction P c₁ c₂ →

      ClosureReconstruction P c₂ c₃ →

      ClosureReconstruction P c₁ c₃) := by

  constructor

  · intro c

    exact closure_reconstruction_refl P c

  constructor

  · intro c₁ c₂ h

    exact closure_reconstruction_symm P c₁ c₂ h

  · intro c₁ c₂ c₃ h₁ h₂

    exact closure_reconstruction_trans P c₁ c₂ c₃ h₁ h₂

end BST