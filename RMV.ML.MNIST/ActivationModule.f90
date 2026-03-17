module ActivationModule
   use NodeModule
   implicit none
   type :: Activation
      type(Node) :: node
      real :: value
   end type Activation
end module ActivationModule