module SoftmaxWithLossModule
   use, intrinsic :: iso_fortran_env, only: real64, error_unit
   implicit none
   private

   public :: SoftmaxWithLossLayer

   type :: SoftmaxWithLossLayer
      real(real64), allocatable :: y(:, :)  ! softmax output [batch, classes]
      real(real64), allocatable :: t(:, :)  ! targets: one-hot [batch, classes] OR label-index [batch, 1]
   contains
      procedure :: forward => softmax_with_loss_forward
      procedure :: backward => softmax_with_loss_backward
      procedure :: reset => softmax_with_loss_reset
   end type SoftmaxWithLossLayer

contains

   subroutine softmax_with_loss_reset(self)
      class(SoftmaxWithLossLayer), intent(inout) :: self

      if (allocated(self%y)) deallocate(self%y)
      if (allocated(self%t)) deallocate(self%t)
   end subroutine softmax_with_loss_reset
   

   function softmax_with_loss_forward(self, x, t) result(loss)
      class(SoftmaxWithLossLayer), intent(inout) :: self
      real(real64), intent(in) :: x(:, :)
      real(real64), intent(in) :: t(:, :)
      real(real64) :: loss

      if (size(t, 1) /= size(x, 1)) then
         write(error_unit, '(a)') 'SoftmaxWithLoss.forward: batch size mismatch.'
         error stop 1
      end if

      if (allocated(self%t)) deallocate(self%t)
      allocate(self%t(size(t, 1), size(t, 2)))
      self%t = t

      call softmax_2d(x, self%y)

      loss = cross_entropy_error(self%y, self%t)
   end function softmax_with_loss_forward
   

   function softmax_with_loss_backward(self, dout) result(dx)
      class(SoftmaxWithLossLayer), intent(inout) :: self
      real(real64), intent(in), optional :: dout
      real(real64), allocatable :: dx(:, :)

      real(real64) :: dout_local
      integer :: batch, classes, i, label

      if (.not. allocated(self%y) .or. .not. allocated(self%t)) then
         write(error_unit, '(a)') 'SoftmaxWithLoss.backward: Forward must be called before Backward.'
         error stop 1
      end if

      dout_local = 1.0_real64
      if (present(dout)) dout_local = dout

      batch = size(self%y, 1)
      classes = size(self%y, 2)

      allocate(dx(batch, classes))

      if (size(self%t, 1) == batch .and. size(self%t, 2) == classes) then
         ! one-hot targets
         dx = (self%y - self%t) / real(batch, real64)
      else if (size(self%t, 1) == batch .and. size(self%t, 2) == 1) then
         ! label-index targets (0-based, matching the C# implementation)
         dx = self%y
         do i = 1, batch
            label = int(self%t(i, 1))
            if (label < 0 .or. label >= classes) then
               write(error_unit, '(a)') 'SoftmaxWithLoss.backward: label index out of range.'
               error stop 1
            end if
            dx(i, label + 1) = dx(i, label + 1) - 1.0_real64
         end do
         dx = dx / real(batch, real64)
      else
         write(error_unit, '(a)') 'SoftmaxWithLoss.backward: t shape must be [batch, classes] or [batch, 1].'
         error stop 1
      end if

      if (dout_local /= 1.0_real64) dx = dx * dout_local
   end function softmax_with_loss_backward
   

   subroutine softmax_2d(x, y)
      real(real64), intent(in) :: x(:, :)
      real(real64), allocatable, intent(out) :: y(:, :)

      integer :: batch, classes, i
      real(real64) :: row_max, row_sum

      batch = size(x, 1)
      classes = size(x, 2)

      if (allocated(y)) deallocate(y)
      allocate(y(batch, classes))

      do i = 1, batch
         row_max = maxval(x(i, :))
         y(i, :) = exp(x(i, :) - row_max)
         row_sum = sum(y(i, :))
         y(i, :) = y(i, :) / row_sum
      end do
   end subroutine softmax_2d
   

   real(real64) function cross_entropy_error(y, t) result(loss)
      real(real64), intent(in) :: y(:, :)
      real(real64), intent(in) :: t(:, :)

      integer :: batch, classes, i, label
      real(real64), parameter :: delta = 1.0e-7_real64

      batch = size(y, 1)
      classes = size(y, 2)

      if (size(t, 1) /= batch) then
         write(error_unit, '(a)') 'SoftmaxWithLoss.cross_entropy_error: batch size mismatch.'
         error stop 1
      end if

      if (size(t, 2) == classes) then
         ! one-hot: loss = -sum(t * log(y + delta)) / batch
         loss = -sum(t * log(y + delta)) / real(batch, real64)
      else if (size(t, 2) == 1) then
         ! label-index (0-based): loss = -avg(log(y(i, label)))
         loss = 0.0_real64
         do i = 1, batch
            label = int(t(i, 1))
            if (label < 0 .or. label >= classes) then
               write(error_unit, '(a)') 'SoftmaxWithLoss.cross_entropy_error: label index out of range.'
               error stop 1
            end if
            loss = loss - log(y(i, label + 1) + delta)
         end do
         loss = loss / real(batch, real64)
      else
         write(error_unit, '(a)') 'SoftmaxWithLoss.cross_entropy_error: t shape must be [batch, classes] or [batch, 1].'
         error stop 1
      end if
   end function cross_entropy_error

end module SoftmaxWithLossModule