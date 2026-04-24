module ReluLayerModule
   use, intrinsic :: iso_fortran_env, only: real64, error_unit
   implicit none
   private

   public :: ReluLayer

   type :: ReluLayer
      real(real64), allocatable :: mask(:, :)
   contains
      procedure :: forward => relu_forward
      procedure :: backward => relu_backward
      procedure :: reset => relu_reset
   end type ReluLayer

contains

   subroutine relu_reset(self)
      class(ReluLayer), intent(inout) :: self

      if (allocated(self%mask)) deallocate(self%mask)
   end subroutine relu_reset
   

   function relu_forward(self, x) result(y)
      class(ReluLayer), intent(inout) :: self
      real(real64), intent(in) :: x(:, :)
      real(real64), allocatable :: y(:, :)

      if (allocated(self%mask)) deallocate(self%mask)
      allocate(self%mask(size(x, 1), size(x, 2)))
      allocate(y(size(x, 1), size(x, 2)))

      self%mask = 0.0_real64
      where (x > 0.0_real64)
         self%mask = 1.0_real64
      end where

      y = x * self%mask
   end function relu_forward
   

   function relu_backward(self, dout) result(dx)
      class(ReluLayer), intent(in) :: self
      real(real64), intent(in) :: dout(:, :)
      real(real64), allocatable :: dx(:, :)

      if (.not. allocated(self%mask)) then
         write(error_unit, '(a)') 'Relu.backward: Forward must be called before Backward.'
         error stop 1
      end if

      if (size(dout, 1) /= size(self%mask, 1) .or. size(dout, 2) /= size(self%mask, 2)) then
         write(error_unit, '(a)') 'Relu.backward: dout shape mismatch.'
         error stop 1
      end if

      allocate(dx(size(dout, 1), size(dout, 2)))
      dx = dout * self%mask
   end function relu_backward

end module ReluLayerModule
 