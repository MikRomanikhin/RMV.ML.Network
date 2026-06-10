module LayersModule
	 implicit none
	 private
	 public :: ilayer, relu, sigmoid, affine, softmax_with_loss

	 ! Base abstract type for layer
	 type, abstract :: ilayer
	 contains
		  procedure(forward_if), deferred, pass :: forward
		  procedure(backward_if), deferred, pass :: backward
	 end type ilayer

	 abstract interface
		  function forward_if(this, x) result(out)
				import :: ilayer
				class(ilayer), intent(inout) :: this
				real, dimension(:,:), intent(in) :: x
				real, allocatable, dimension(:,:) :: out
		  end function forward_if

		  function backward_if(this, dout) result(dx)
				import :: ilayer
				class(ilayer), intent(inout) :: this
				real, dimension(:,:), intent(in) :: dout
				real, allocatable, dimension(:,:) :: dx
		  end function backward_if
	 end interface
	 

	 ! ReLU
	 type, extends(ilayer) :: relu
		  real, allocatable, dimension(:,:) :: matrix_mask
	 contains
		  procedure, pass :: forward => relu_forward
		  procedure, pass :: backward => relu_backward
	 end type relu
	 

	 ! Sigmoid
	 type, extends(ilayer) :: sigmoid
		  real, allocatable, dimension(:,:) :: out_matrix
	 contains
		  procedure, pass :: forward => sigmoid_forward
		  procedure, pass :: backward => sigmoid_backward
	 end type sigmoid
	 

	 ! Affine
	 type, extends(ilayer) :: affine
		  real, allocatable, dimension(:,:) :: w
		  real, allocatable, dimension(:) :: b
		  real, allocatable, dimension(:,:) :: dw
		  real, allocatable, dimension(:) :: db
		  real, allocatable, dimension(:,:) :: x
	 contains
		  procedure, pass :: init => affine_init
		  procedure, pass :: forward => affine_forward
		  procedure, pass :: backward => affine_backward
	 end type affine

	 ! SoftmaxWithLoss
	 type :: softmax_with_loss
		  real, allocatable, dimension(:,:) :: y
		  real, allocatable, dimension(:,:) :: t
	 contains
		  procedure, pass :: forward => swl_forward
		  procedure, pass :: backward => swl_backward
	 end type softmax_with_loss

contains

	 ! ReLU implementation
	 function relu_forward(this, x) result(out)
		  class(relu), intent(inout) :: this
		  real, contiguous, dimension(:,:), intent(in) :: x
		  real, allocatable, dimension(:,:) :: out

		  if (allocated(this%matrix_mask)) deallocate(this%matrix_mask)
		  allocate(this%matrix_mask(size(x, 1), size(x, 2)))
		  this%matrix_mask = merge(1.0, 0.0, x > 0.0)  ! single SIMD expression, no branch
		  out = x * this%matrix_mask
	 end function relu_forward

	 function relu_backward(this, dout) result(dx)
		  class(relu), intent(inout) :: this
		  real, contiguous, dimension(:,:), intent(in) :: dout
		  real, allocatable, dimension(:,:) :: dx

		  dx = dout * this%matrix_mask
	 end function relu_backward

	 ! Sigmoid implementation
	 function sigmoid_forward(this, x) result(out)
		  class(sigmoid), intent(inout) :: this
		  real, contiguous, dimension(:,:), intent(in) :: x
		  real, allocatable, dimension(:,:) :: out

		  if (allocated(this%out_matrix)) deallocate(this%out_matrix)
		  allocate(this%out_matrix(size(x, 1), size(x, 2)))

		  this%out_matrix = 1.0 / (1.0 + exp(-x))
		  out = this%out_matrix
	 end function sigmoid_forward

	 function sigmoid_backward(this, dout) result(dx)
		  class(sigmoid), intent(inout) :: this
		  real, contiguous, dimension(:,:), intent(in) :: dout
		  real, allocatable, dimension(:,:) :: dx

		  dx = dout * this%out_matrix * (1.0 - this%out_matrix)
	 end function sigmoid_backward

	 ! Affine implementation
	 subroutine affine_init(this, w, b)
		  class(affine), intent(inout) :: this
		  real, dimension(:,:), intent(in) :: w
		  real, dimension(:), intent(in) :: b

		  if (allocated(this%w)) deallocate(this%w)
		  if (allocated(this%b)) deallocate(this%b)
		  allocate(this%w(size(w, 1), size(w, 2)))
		  allocate(this%b(size(b, 1)))
		  this%w = w
		  this%b = b
	 end subroutine affine_init

	 function affine_forward(this, x) result(out)
		  class(affine), intent(inout) :: this
		  real, contiguous, dimension(:,:), intent(in) :: x
		  real, allocatable, dimension(:,:) :: out

		  if (allocated(this%x)) deallocate(this%x)
		  allocate(this%x(size(x, 1), size(x, 2)))
		  this%x = x

		  allocate(out(size(x, 1), size(this%w, 2)))
		  out = matmul(x, this%w) + spread(this%b, 1, size(x, 1))
	 end function affine_forward

	 function affine_backward(this, dout) result(dx)
		  class(affine), intent(inout) :: this
		  real, contiguous, dimension(:,:), intent(in) :: dout
		  real, allocatable, dimension(:,:) :: dx
		  real, allocatable :: ones(:)

		  allocate(dx(size(this%x, 1), size(this%w, 1)))
		  dx = matmul(dout, transpose(this%w))

		  if (allocated(this%dw)) deallocate(this%dw)
		  allocate(this%dw(size(this%x,2), size(dout,2)))
		  this%dw = matmul(transpose(this%x), dout)

		  if (allocated(this%db)) deallocate(this%db)
		  allocate(this%db(size(dout, 2)))
		  allocate(ones(size(dout, 1)))
		  ones = 1.0
		  this%db = matmul(ones, dout)  ! BLAS SGEMV: ones^T * dout = column sums, replaces scalar loop
	 end function affine_backward

	 ! SoftmaxWithLoss implementation
	 function swl_forward(this, x, t) result(loss)
		  class(softmax_with_loss), intent(inout) :: this
		  real, contiguous, dimension(:,:), intent(in) :: x, t
		  real :: loss
		  integer :: batch_size, classes
		  real, allocatable :: row_max(:), row_sum(:)

		  batch_size = size(x, 1)
		  classes = size(x, 2)

		  if (allocated(this%y)) deallocate(this%y)
		  allocate(this%y(batch_size, classes))
		  if (allocated(this%t)) deallocate(this%t)
		  allocate(this%t(size(t,1), size(t,2)))
		  this%t = t

		  allocate(row_max(batch_size))
		  allocate(row_sum(batch_size))
		  row_max = maxval(x, dim=2)                          ! per-row max, vectorized over all rows
		  this%y  = exp(x - spread(row_max, 2, classes))     ! broadcast subtract + full-array exp
		  row_sum = sum(this%y, dim=2)                        ! per-row sum, vectorized
		  this%y  = this%y / spread(row_sum, 2, classes)     ! broadcast divide, fully vectorized

		  loss = cross_entropy_error(this%y, this%t)
	 end function swl_forward

	 function swl_backward(this, dout_opt) result(dx)
		  class(softmax_with_loss), intent(inout) :: this
		  real, optional, intent(in) :: dout_opt
		  real, dimension(size(this%y,1), size(this%y,2)) :: dx
		  real :: dout
		  integer :: batch_size, i, label

		  dout = 1.0
		  if (present(dout_opt)) dout = dout_opt

		  batch_size = size(this%t, 1)

		  if (size(this%t, 1) == size(this%y, 1) .and. size(this%t, 2) == size(this%y, 2)) then
				dx = (this%y - this%t) / real(batch_size)
		  else
				dx = this%y
				do i = 1, batch_size
					 label = int(this%t(i, 1))
					 dx(i, label + 1) = dx(i, label + 1) - 1.0
				end do
				dx = dx / real(batch_size)
		  end if

		  if (dout /= 1.0) then
				dx = dx * dout
		  end if
	 end function swl_backward

	 function cross_entropy_error(y, t) result(loss)
		  real, dimension(:,:), intent(in) :: y, t
		  real :: loss
		  real, parameter :: delta = 1.0e-7
		  integer :: batch_size, i, label
		  real, allocatable, dimension(:,:) :: log_y

		  batch_size = size(y, 1)

		  if (size(t, 1) == size(y, 1) .and. size(t, 2) == size(y, 2)) then
				allocate(log_y(size(y,1), size(y,2)))
				log_y = log(y + delta)
				loss = -sum(t * log_y) / real(batch_size)
				deallocate(log_y)
		  else
				loss = 0.0
				do i = 1, batch_size
					 label = int(t(i, 1))
					 loss = loss - log(y(i, label + 1) + delta)
				end do
				loss = loss / real(batch_size)
		  end if
	 end function cross_entropy_error

end module LayersModule
