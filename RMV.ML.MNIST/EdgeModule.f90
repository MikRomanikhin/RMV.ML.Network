!--------------------------------------------------------------------------------------------
! Defines the Edge type, which represents a connection between two nodes in a neural network.
! Each edge has a weight, and methods to calculate the weighted target error, weighted source 
! value, and to learn and update the weight based on the error and learning rate.
!--------------------------------------------------------------------------------------------
module EdgeModule
use, intrinsic :: iso_fortran_env, only: int32, real64
use NodeModule
implicit none

real, parameter :: LEARNING_RATE = 0.01, MOMENTUM = 0.1
real :: delta = 0.0, sum = 0.0

 type :: Edge
  	 integer :: ID
    real :: Weight    
    type(Node) :: From
    type(Node) :: To    
    
 contains
      procedure, pass(self) :: WeightedTargetError, WeightedSourceValue, Learn, Update    
 end type Edge
    
    contains   
   
    !------------------------------------------------------------------------------------
    ! Accumulates the product of the source node value and target node error for learning
    !------------------------------------------------------------------------------------
    function Learn(self) result(res)
       class(Edge) :: self
       real :: res
       sum = sum + self%From%Value * self%To%Error   
       res = sum
    end function Learn
    
    !-----------------------------------------------------------------------
    ! Updates weight based on the accumulated gradient and the learning rate
    !-----------------------------------------------------------------------
    subroutine Update(self, batchSize)
     class(Edge) :: self
     integer, intent(in) :: batchSize
     real :: gradient       
       gradient = sum / batchSize
       delta = gradient * LEARNING_RATE + delta * MOMENTUM
       self%Weight = self%Weight + delta
       sum = 0.0
    end subroutine Update
    
    
     function WeightedTargetError(self) result(res)
       class(Edge) :: self
       real :: res
       res = self%To%Error * self%Weight
    end function WeightedTargetError
    
    function WeightedSourceValue(self) result(res)
       class(Edge) :: self
       real :: res
       res = self%From%Value * self%Weight
    end function WeightedSourceValue
   
end module EdgeModule