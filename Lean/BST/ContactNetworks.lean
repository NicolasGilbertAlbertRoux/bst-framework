import BST.TopologicalSelection

namespace BST

universe u

/--
A contact relation between matter species.
-/
def ContactRelation (L : Type u) :=
  MatterSpecies L → MatterSpecies L → Prop

/--
A contact network is a contact relation.
-/
def ContactNetwork (L : Type u) :=
  ContactRelation L

/--
Symmetric contact.
-/
def SymmetricContact
    {L : Type u}
    (R : ContactNetwork L) : Prop :=
  ∀ a b, R a b → R b a

/--
Reflexive contact.
-/
def ReflexiveContact
    {L : Type u}
    (R : ContactNetwork L) : Prop :=
  ∀ a, R a a

/--
A well-formed contact network.
-/
def WellFormedContactNetwork
    {L : Type u}
    (R : ContactNetwork L) : Prop :=
  SymmetricContact R ∧ ReflexiveContact R

theorem contact_network_is_symmetric
    {L : Type u}
    (R : ContactNetwork L)
    (h : WellFormedContactNetwork R) :
    SymmetricContact R := by
  exact h.left

theorem contact_network_is_reflexive
    {L : Type u}
    (R : ContactNetwork L)
    (h : WellFormedContactNetwork R) :
    ReflexiveContact R := by
  exact h.right

end BST