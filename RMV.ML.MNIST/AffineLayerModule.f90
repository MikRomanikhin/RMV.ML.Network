module AffineLayerModule
   use, intrinsic :: iso_fortran_env, only: real64, error_unit
   implicit none
   private

   public :: AffineLayer

   type :: AffineLayer
      
      real(real64), allocatable :: w(:, :), b(:)   ! y = x * W + b      
      real(real64), allocatable :: dw(:, :), db(:) ! grads

      ! cache
      real(real64), allocatable :: x(:, :)
   contains
      procedure :: init => affine_init
      procedure :: forward => affine_forward
      procedure :: backward => affine_backward
      procedure :: reset => affine_reset
   end type AffineLayer

contains

   subroutine affine_init(self, w, b)
      class(AffineLayer), intent(inout) :: self
      real(real64), intent(in) :: w(:, :)
      real(real64), intent(in) :: b(:)

      if (size(w, 2) /= size(b)) then
         write(error_unit, '(a)') 'Affine.init: W columns must match b length.'
         error stop 1
      end if

      if (allocated(self%w)) deallocate(self%w)
      if (allocated(self%b)) deallocate(self%b)

      allocate(self%w(size(w, 1), size(w, 2)))
      allocate(self%b(size(b)))

      self%w = w
      self%b = b

      call self%reset()
   end subroutine affine_init
   

   subroutine affine_reset(self)
      class(AffineLayer), intent(inout) :: self

      if (allocated(self%x)) deallocate(self%x)

      if (allocated(self%dw)) deallocate(self%dw)
      if (allocated(self%db)) deallocate(self%db)
   end subroutine affine_reset
   

   function affine_forward(self, input) result(out)
      class(AffineLayer), intent(inout) :: self
      real(real64), intent(in) :: input(:, :)
      real(real64), allocatable :: out(:, :)

      integer :: batch, out_dim

      if (.not. allocated(self%w) .or. .not. allocated(self%b)) then
         write(error_unit, '(a)') 'Affine.forward: Layer not initialized. Call init first.'
         error stop 1
      end if

      if (size(input, 2) /= size(self%w, 1)) then
         write(error_unit, '(a)') 'Affine.forward: input shape mismatch (input cols must match W rows).'
         error stop 1
      end if

      batch = size(input, 1)
      out_dim = size(self%w, 2)

      if (allocated(self%x)) deallocate(self%x)
      allocate(self%x(batch, size(input, 2)))
      self%x = input

      allocate(out(batch, out_dim))

      out = matmul(input, self%w)
      out = out + spread(self%b, dim=1, ncopies=batch)
   end function affine_forward
   

   function affine_backward(self, dout) result(dx)
      class(AffineLayer), intent(inout) :: self
      real(real64), intent(in) :: dout(:, :)
      real(real64), allocatable :: dx(:, :)

      integer :: batch

      if (.not. allocated(self%x)) then
         write(error_unit, '(a)') 'Affine.backward: Forward must be called before Backward.'
         error stop 1
      end if

      if (size(dout, 1) /= size(self%x, 1) .or. size(dout, 2) /= size(self%w, 2)) then
         write(error_unit, '(a)') 'Affine.backward: dout shape mismatch.'
         error stop 1
      end if

      batch = size(dout, 1)

      allocate(dx(size(dout, 1), size(self%w, 1)))
      dx = matmul(dout, transpose(self%w))                ! dx = dout * W^T

      if (allocated(self%dw)) deallocate(self%dw)
      allocate(self%dw(size(self%w, 1), size(self%w, 2)))
      self%dw = matmul(transpose(self%x), dout)           ! dW = x^T * dout

      if (allocated(self%db)) deallocate(self%db)
      allocate(self%db(size(self%b)))
      self%db = sum(dout, dim=1)                          ! dB = column-wise sum over batch
   end function affine_backward

end module AffineLayerModule