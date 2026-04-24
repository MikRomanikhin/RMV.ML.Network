module Layers
    implicit none
    private
    
    public :: Relu, Affine, SoftmaxWithLoss

    !----------------------------------------------------------------------
    ! Relu Layer
    !----------------------------------------------------------------------
    type :: Relu
        logical, allocatable :: mask(:,:)
    contains
        procedure :: forward => relu_forward
        procedure :: backward => relu_backward
    end type Relu
    
    !----------------------------------------------------------------------
    ! Sigmoid Layer
    !----------------------------------------------------------------------
    !type :: Sigmoid
    !    real, allocatable :: out(:,:)
    !contains
    !    procedure :: forward => sigmoid_forward
    !    procedure :: backward => sigmoid_backward
    !end type Sigmoid

    !----------------------------------------------------------------------
    ! Affine (Fully Connected) Layer
    !----------------------------------------------------------------------
    type :: Affine
        real, allocatable :: W(:,:), B(:), x(:,:), dW(:,:), dB(:)
    contains
        procedure :: init => affine_init
        procedure :: forward => affine_forward
        procedure :: backward => affine_backward
    end type Affine

    !----------------------------------------------------------------------
    ! SoftmaxWithLoss Layer
    !----------------------------------------------------------------------
    type :: SoftmaxWithLoss
        real :: loss
        real, allocatable :: y(:,:)
        integer, allocatable :: t(:) 
    contains
        procedure :: forward => softmax_forward
        procedure :: backward => softmax_backward
    end type SoftmaxWithLoss

contains

    ! ==========================================
    ! Relu Methods
    ! ==========================================
    function relu_forward(this, x) result(out)
        class(Relu), intent(inout) :: this
        real, intent(in) :: x(:,:)
        real, allocatable :: out(:,:)
        
        if (allocated(this%mask)) deallocate(this%mask)
        allocate(this%mask(size(x, 1), size(x, 2)))
        
        this%mask = (x <= 0.0)
        
        allocate(out(size(x, 1), size(x, 2)))
        out = x
        where (this%mask) out = 0.0
    end function relu_forward

    function relu_backward(this, dout) result(dx)
        class(Relu), intent(inout) :: this
        real, intent(in) :: dout(:,:)
        real, allocatable :: dx(:,:)
        
        allocate(dx(size(dout, 1), size(dout, 2)))
        dx = dout
        where (this%mask) dx = 0.0
    end function relu_backward   
   

    ! ==========================================
    ! Affine Methods
    ! ==========================================
    subroutine affine_init(this, W, B)
      class(Affine), intent(inout) :: this
      real, intent(in) :: W(:, :), B(:)        
        this%W = W
        this%B = B 
    end subroutine affine_init

    function affine_forward(this, x) result(out)
      class(Affine), intent(inout) :: this
      real, intent(in) :: x(:,:)
      real, allocatable :: out(:,:)
      integer :: i, N, D_out
        
        this%x = x
        N = size(x, 1)
        D_out = size(this%W, 2)
        
        allocate(out(N, D_out))        
        out = matmul(this%x, this%W) ! out = np.dot(x, W) + b
        do i = 1, N
            out(i, :) = out(i, :) + this%B
        end do
    end function affine_forward

    function affine_backward(this, dout) result(dx)
      class(Affine), intent(inout) :: this
      real, intent(in) :: dout(:,:)
      real, allocatable :: dx(:,:)
      integer :: i, N
                
        dx = matmul(dout, transpose(this%W)) ! dx = np.dot(dout, W.T)
                
        this%dW = matmul(transpose(this%x), dout) ! dW = np.dot(x.T, dout)
        
        ! db = np.sum(dout, axis=0)
        N = size(dout, 1)
        if (allocated(this%dB)) deallocate(this%dB)
        allocate(this%dB(size(dout, 2)))
        this%dB = 0.0
        do i = 1, size(dout, 2)
            this%dB(i) = sum(dout(:, i))
        end do
    end function affine_backward
    

    ! ==========================================
    ! SoftmaxWithLoss Methods
    ! ==========================================
    function softmax_forward(this, x, t) result(loss)
      class(SoftmaxWithLoss), intent(inout) :: this
      real, intent(in) :: x(:,:)
      integer, intent(in) :: t(:) ! Switched to match 1D index
      real :: loss              
        
        this%t = t
        this%y = softmax(x)         
        loss = cross_entropy_error(this%y, this%t) 
        this%loss = loss
    end function softmax_forward

    function softmax_backward(this, dout) result(dx)
        class(SoftmaxWithLoss), intent(inout) :: this
        real, intent(in), optional :: dout
        real, allocatable :: dx(:,:)
        real :: batch_size
        integer :: i
        
        batch_size = real(size(this%t))
        allocate(dx(size(this%y, 1), size(this%y, 2)))        
         
        dx = this%y
        ! Subtract 1 at the actual index label representing the correct class probabilities y - t logic
        do i = 1, size(this%t)
            dx(i, this%t(i)) = dx(i, this%t(i)) - 1.0 
        end do
        
        dx = dx / batch_size
        
    end function softmax_backward
    
     pure function softmax(x) result(y)
        real, intent(in) :: x(:,:)
        real, allocatable :: y(:,:)
        real, allocatable :: max_x(:), sum_exp(:)
        integer :: i, N, D

        N = size(x, 1)
        D = size(x, 2)
        allocate(y(N, D), max_x(N), sum_exp(N))

        ! Find max along the features axis (-1 in Python) for numerical stability
        do i = 1, N
            max_x(i) = maxval(x(i, :))
            y(i, :) = exp(x(i, :) - max_x(i))
            sum_exp(i) = sum(y(i, :))
            y(i, :) = y(i, :) / sum_exp(i)
        end do
    end function softmax
    
     ! Assuming t contains 1-based class indices for each batch sample
    pure real function cross_entropy_error(y, t)
      real, intent(in) :: y(:,:)
      integer, intent(in) :: t(:) ! 1D array of actual label indices
      integer :: i, batch_size
      real, parameter :: delta = 1e-7
      real :: loss

        batch_size = size(y, 1)
        loss = 0.0
        do i = 1, batch_size
            loss = loss + log(y(i, t(i)) + delta)
        end do
        cross_entropy_error = -loss / real(batch_size)        
    end function cross_entropy_error

    pure real function softmax_loss(x, t)
      real, intent(in) :: x(:,:)
      integer, intent(in) :: t(:)
      real, allocatable :: y(:,:)
        y = softmax(x)
        softmax_loss = cross_entropy_error(y, t)
    end function softmax_loss

    ! ==========================================
    ! Sigmoid Methods
    ! ==========================================
    !function sigmoid_forward(this, x) result(out_val)
    !    class(Sigmoid), intent(inout) :: this
    !    real, intent(in) :: x(:,:)
    !    real, allocatable :: out_val(:,:)
    !    
    !    if (allocated(this%out)) deallocate(this%out)
    !    allocate(this%out(size(x, 1), size(x, 2)))        
    !   
    !    this%out = sigmoid(x) ! from the Functions module
    !    
    !    allocate(out_val(size(x, 1), size(x, 2)))
    !    out_val = this%out
    !end function sigmoid_forward
    !
    !function sigmoid_backward(this, dout) result(dx)
    !  class(Sigmoid), intent(inout) :: this
    !  real, intent(in) :: dout(:,:)
    !  real, allocatable :: dx(:,:)
    !    
    !    allocate(dx(size(dout, 1), size(dout, 2)))               
    !    dx = dout * sigmoid_grad(this%out)
    !end function sigmoid_backward
    !
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

end module Layers