module Functions
    implicit none
    private

    public :: identity_function
    public :: step_function
    !public :: sigmoid
    !public :: sigmoid_grad
    !public :: relu
    !public :: relu_grad
    !public :: softmax
    public :: sum_squared_error
    !public :: cross_entropy_error
    !public :: softmax_loss

contains

    !----------------------------------------------------------------------
    ! Element-wise Functions
    !----------------------------------------------------------------------
    elemental real function identity_function(x)
        real, intent(in) :: x
        identity_function = x
    end function identity_function

    elemental integer function step_function(x)
        real, intent(in) :: x
        if (x > 0.0) then
            step_function = 1
        else
            step_function = 0
        end if
    end function step_function

    !elemental real function sigmoid(x)
    !    real, intent(in) :: x
    !    sigmoid = 1.0 / (1.0 + exp(-x))
    !end function sigmoid
    !
    !elemental real function sigmoid_grad(x)
    !    real, intent(in) :: x
    !    real :: sig
    !    sig = sigmoid(x)
    !    sigmoid_grad = (1.0 - sig) * sig
    !end function sigmoid_grad

    !elemental real function relu(x)
    !    real, intent(in) :: x
    !    relu = max(0.0, x)
    !end function relu
    !
    !elemental real function relu_grad(x)
    !    real, intent(in) :: x
    !    if (x >= 0.0) then
    !        relu_grad = 1.0
    !    else
    !        relu_grad = 0.0
    !    end if
    !end function relu_grad

    !----------------------------------------------------------------------
    ! Matrix/Array-based Functions 
    ! (Assuming 2D input where dim 1 is batches and dim 2 is features/classes)
    !----------------------------------------------------------------------
    !pure function softmax(x) result(y)
    !    real, intent(in) :: x(:,:)
    !    real, allocatable :: y(:,:)
    !    real, allocatable :: max_x(:), sum_exp(:)
    !    integer :: i, N, D
    !
    !    N = size(x, 1)
    !    D = size(x, 2)
    !    allocate(y(N, D), max_x(N), sum_exp(N))
    !
    !    ! Find max along the features axis (-1 in Python) for numerical stability
    !    do i = 1, N
    !        max_x(i) = maxval(x(i, :))
    !        y(i, :) = exp(x(i, :) - max_x(i))
    !        sum_exp(i) = sum(y(i, :))
    !        y(i, :) = y(i, :) / sum_exp(i)
    !    end do
    !end function softmax

    pure real function sum_squared_error(y, t)
        real, intent(in) :: y(:,:), t(:,:)
        sum_squared_error = 0.5 * sum((y - t)**2)
    end function sum_squared_error

    !! Assuming t contains 1-based class indices for each batch sample
    !pure real function cross_entropy_error(y, t)
    !    real, intent(in) :: y(:,:)
    !    integer, intent(in) :: t(:) ! 1D array of actual label indices
    !    integer :: i, batch_size
    !    real, parameter :: delta = 1e-7
    !    real :: loss
    !
    !    batch_size = size(y, 1)
    !    loss = 0.0
    !
    !    do i = 1, batch_size
    !        loss = loss + log(y(i, t(i)) + delta)
    !    end do
    !
    !    cross_entropy_error = -loss / real(batch_size)
    !end function cross_entropy_error
    !
    !pure real function softmax_loss(x, t)
    !    real, intent(in) :: x(:,:)
    !    integer, intent(in) :: t(:)
    !    real, allocatable :: y(:,:)
    !
    !    y = softmax(x)
    !    softmax_loss = cross_entropy_error(y, t)
    !end function softmax_loss

end module Functions