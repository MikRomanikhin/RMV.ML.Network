! ----------------------------------------------------------------------------------------------
! This module defines the Node and Edge types for a simple neural network implementation.
! Each Node represents a neuron with its value, error, and connections to other nodes via Edges.
! Each Edge represents a weighted connection between two nodes and includes methods for learning
! and updating the weights based on the error and learning rate.
! ---------------------------------------------------------------------------------------------
module NodeModule
use, intrinsic :: iso_fortran_env, only: int32, real64
use ToolsModule
implicit none
private
public :: Node, Edge
real, parameter :: LEARNING_RATE = 0.01, MOMENTUM = 0.1

!-----------------------------------------------------------------------
! Represents a neuron in a neural network. Each node has an ID, a value, 
! a raw value, an error, and lists of incoming and outgoing edges.
!-----------------------------------------------------------------------
   type :: Node
      integer :: ID = 0
      real :: Value = 0.0
      real:: RawValue = 0.0
      real :: Error = 0.0
      integer, allocatable :: InEdges(:)   ! indices into an Edges array
      integer, allocatable :: OutEdges(:)  ! indices into an Edges array
      real :: bias = 0.0, derivative = 0.0, delta = 0.0, sum = 0.0
   contains
      procedure, pass(self) :: Initialize
   end type Node

   
!----------------------------------------------------------------------------------------
! Represents a connection between two nodes in a neural network. Each edge has a weight, 
! and methods to calculate the weighted target error, weighted source value, and to learn
! and update the weight based on the error and learning rate.
!----------------------------------------------------------------------------------------
   type :: Edge
  	 integer :: ID
    real :: Weight    
    type(Node) :: From
    type(Node) :: To    
    real :: delta = 0.0, sum = 0.0
 contains
      procedure, pass(self) :: WeightedTargetError, WeightedSourceValue, Learn, Update    
 end type Edge      
    
   contains   
 
   ! ========================= Edge Methods =========================
    !------------------------------------------------------------------------------------
    ! Accumulates the product of the source node value and target node error for learning
    !------------------------------------------------------------------------------------
    function Learn(self) result(res)
       class(Edge) :: self
       real :: res
       self%sum = self%sum + self%From%Value * self%To%Error   
       res = self%sum
    end function Learn
    
    !-----------------------------------------------------------------------
    ! Updates weight based on the accumulated gradient and the learning rate
    !-----------------------------------------------------------------------
    subroutine Update(self, batchSize)
     class(Edge) :: self
     integer, intent(in) :: batchSize
     real :: gradient       
       gradient = self%sum / batchSize
       self%delta = gradient * LEARNING_RATE + self%delta * MOMENTUM
       self%Weight = self%Weight + self%delta
       self%sum = 0.0
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
    
    
! ================================ Node Methods ================================
    
    !-------------------------------------------------------------------------
    ! Initializes the node's incoming edge weights using Xavier initialization
    !-------------------------------------------------------------------------
    subroutine Initialize(self)
      class(Node) :: self
      real :: std
      integer :: i
      if( size(self%InEdges) == 0 ) return
		std = sqrt( 2.0 / size(self%InEdges) )
      do i = 1, size(self%InEdges)
         self%InEdges(i)%Weight = GetGaussian() * std      	
      end do
		self%bias = 0.0
	}
    
    !------------------------------------------------------------------------------------
    ! Calculates the error for this node based on the weighted errors from outgoing edges
    !------------------------------------------------------------------------------------
    subroutine CalculateError(self, outgoingEdges)
      class(Node) :: self
      type(Edge), intent(in) :: outgoingEdges(:)
      integer :: i
      self%Error = 0.0
      do i = 1, size(outgoingEdges)
         self%Error = self%Error + outgoingEdges(i)%WeightedTargetError()
      end do
    end subroutine CalculateError

   
end module NodeModule