module optimizers
    implicit none
    private

    public :: SGD, Momentum, Nesterov, AdaGrad, RMSprop, Adam

    !----------------------------------------------------------------------
    ! SGD
    !----------------------------------------------------------------------
    type :: SGD
        real :: lr = 0.01
    contains
        procedure :: update_1d => sgd_update_1d
        procedure :: update_2d => sgd_update_2d        
        generic :: update => update_1d, update_2d
    end type SGD

    !----------------------------------------------------------------------
    ! Momentum
    !----------------------------------------------------------------------
    type :: Momentum
        real :: lr = 0.01
        real :: momentum_val = 0.9
        real, allocatable :: v_1d(:)
        real, allocatable :: v_2d(:,:)
    contains
        procedure :: update_1d => momentum_update_1d
        procedure :: update_2d => momentum_update_2d
        generic :: update => update_1d, update_2d
    end type Momentum

    !----------------------------------------------------------------------
    ! Nesterov
    !----------------------------------------------------------------------
    type :: Nesterov
        real :: lr = 0.01
        real :: momentum_val = 0.9
        real, allocatable :: v_1d(:)
        real, allocatable :: v_2d(:,:)
    contains
        procedure :: update_1d => nesterov_update_1d
        procedure :: update_2d => nesterov_update_2d
        generic :: update => update_1d, update_2d
    end type Nesterov

    !----------------------------------------------------------------------
    ! AdaGrad
    !----------------------------------------------------------------------
    type :: AdaGrad
        real :: lr = 0.01
        real, allocatable :: h_1d(:)
        real, allocatable :: h_2d(:,:)
    contains
        procedure :: update_1d => adagrad_update_1d
        procedure :: update_2d => adagrad_update_2d
        generic :: update => update_1d, update_2d
    end type AdaGrad

    !----------------------------------------------------------------------
    ! RMSprop
    !----------------------------------------------------------------------
    type :: RMSprop
        real :: lr = 0.01
        real :: decay_rate = 0.99
        real, allocatable :: h_1d(:)
        real, allocatable :: h_2d(:,:)
    contains
        procedure :: update_1d => rmsprop_update_1d
        procedure :: update_2d => rmsprop_update_2d
        generic :: update => update_1d, update_2d
    end type RMSprop

    !----------------------------------------------------------------------
    ! Adam
    !----------------------------------------------------------------------
    type :: Adam
        real :: lr = 0.001
        real :: beta1 = 0.9
        real :: beta2 = 0.999
        integer :: iter = 0
        real, allocatable :: m_1d(:), v_1d(:)
        real, allocatable :: m_2d(:,:), v_2d(:,:)
    contains
        procedure :: update_1d => adam_update_1d
        procedure :: update_2d => adam_update_2d
        generic :: update => update_1d, update_2d
    end type Adam

contains

    ! ==========================================
    ! SGD
    ! ==========================================
    subroutine sgd_update_1d(this, params, grads)
      class(SGD), intent(inout) :: this
      real, intent(inout) :: params(:)
      real, intent(in) :: grads(:)
        params = params - this%lr * grads
    end subroutine sgd_update_1d
    
    subroutine sgd_update_2d(this, params, grads)
      class(SGD), intent(inout) :: this
      real, intent(inout) :: params(:,:)
      real, intent(in) :: grads(:,:)
        params = params - this%lr * grads
    end subroutine sgd_update_2d   
    

    ! ==========================================
    ! Momentum
    ! ==========================================
    subroutine momentum_update_1d(this, params, grads)
      class(Momentum), intent(inout) :: this
      real, intent(inout) :: params(:)
      real, intent(in) :: grads(:)
        if (.not. allocated(this%v_1d)) then
            allocate(this%v_1d(size(params)))
            this%v_1d = 0.0
        end if
        this%v_1d = this%momentum_val * this%v_1d - this%lr * grads
        params = params + this%v_1d
    end subroutine momentum_update_1d

    subroutine momentum_update_2d(this, params, grads)
      class(Momentum), intent(inout) :: this
      real, intent(inout) :: params(:,:)
      real, intent(in) :: grads(:,:)
        if (.not. allocated(this%v_2d)) then
            allocate(this%v_2d(size(params, 1), size(params, 2)))
            this%v_2d = 0.0
        end if
        this%v_2d = this%momentum_val * this%v_2d - this%lr * grads
        params = params + this%v_2d
    end subroutine momentum_update_2d

    ! ==========================================
    ! Nesterov
    ! ==========================================
    subroutine nesterov_update_1d(this, params, grads)
      class(Nesterov), intent(inout) :: this
      real, intent(inout) :: params(:)
      real, intent(in) :: grads(:)
        if (.not. allocated(this%v_1d)) then
            allocate(this%v_1d(size(params)))
            this%v_1d = 0.0
        end if
        this%v_1d = this%v_1d * this%momentum_val
        this%v_1d = this%v_1d - this%lr * grads
        params = params + this%momentum_val * this%momentum_val * this%v_1d
        params = params - (1.0 + this%momentum_val) * this%lr * grads
    end subroutine nesterov_update_1d

    subroutine nesterov_update_2d(this, params, grads)
      class(Nesterov), intent(inout) :: this
      real, intent(inout) :: params(:,:)
      real, intent(in) :: grads(:,:)
        if (.not. allocated(this%v_2d)) then
            allocate(this%v_2d(size(params, 1), size(params, 2)))
            this%v_2d = 0.0
        end if
        this%v_2d = this%v_2d * this%momentum_val
        this%v_2d = this%v_2d - this%lr * grads
        params = params + this%momentum_val * this%momentum_val * this%v_2d
        params = params - (1.0 + this%momentum_val) * this%lr * grads
    end subroutine nesterov_update_2d

    ! ==========================================
    ! AdaGrad
    ! ==========================================
    subroutine adagrad_update_1d(this, params, grads)
      class(AdaGrad), intent(inout) :: this
      real, intent(inout) :: params(:)
      real, intent(in) :: grads(:)
        if (.not. allocated(this%h_1d)) then
            allocate(this%h_1d(size(params)))
            this%h_1d = 0.0
        end if
        this%h_1d = this%h_1d + grads * grads
        params = params - this%lr * grads / (sqrt(this%h_1d) + 1e-7)
    end subroutine adagrad_update_1d

    subroutine adagrad_update_2d(this, params, grads)
      class(AdaGrad), intent(inout) :: this
      real, intent(inout) :: params(:,:)
      real, intent(in) :: grads(:,:)
        if (.not. allocated(this%h_2d)) then
            allocate(this%h_2d(size(params, 1), size(params, 2)))
            this%h_2d = 0.0
        end if
        this%h_2d = this%h_2d + grads * grads
        params = params - this%lr * grads / (sqrt(this%h_2d) + 1e-7)
    end subroutine adagrad_update_2d

    ! ==========================================
    ! RMSprop
    ! ==========================================
    subroutine rmsprop_update_1d(this, params, grads)
      class(RMSprop), intent(inout) :: this
      real, intent(inout) :: params(:)
      real, intent(in) :: grads(:)
        if (.not. allocated(this%h_1d)) then
            allocate(this%h_1d(size(params)))
            this%h_1d = 0.0
        end if
        this%h_1d = this%h_1d * this%decay_rate
        this%h_1d = this%h_1d + (1.0 - this%decay_rate) * grads * grads
        params = params - this%lr * grads / (sqrt(this%h_1d) + 1e-7)
    end subroutine rmsprop_update_1d

    subroutine rmsprop_update_2d(this, params, grads)
      class(RMSprop), intent(inout) :: this
      real, intent(inout) :: params(:,:)
      real, intent(in) :: grads(:,:)
        if (.not. allocated(this%h_2d)) then
            allocate(this%h_2d(size(params, 1), size(params, 2)))
            this%h_2d = 0.0
        end if
        this%h_2d = this%h_2d * this%decay_rate
        this%h_2d = this%h_2d + (1.0 - this%decay_rate) * grads * grads
        params = params - this%lr * grads / (sqrt(this%h_2d) + 1e-7)
    end subroutine rmsprop_update_2d

    ! ==========================================
    ! Adam
    ! ==========================================
    subroutine adam_update_1d(this, params, grads)
      class(Adam), intent(inout) :: this
      real, intent(inout) :: params(:)
      real, intent(in) :: grads(:)
      real :: lr_t

        if (.not. allocated(this%m_1d)) then
            allocate(this%m_1d(size(params)))
            allocate(this%v_1d(size(params)))
            this%m_1d = 0.0
            this%v_1d = 0.0
        end if

        this%iter = this%iter + 1
        lr_t = this%lr * sqrt(1.0 - this%beta2**this%iter) / (1.0 - this%beta1**this%iter)

        this%m_1d = this%m_1d + (1.0 - this%beta1) * (grads - this%m_1d)
        this%v_1d = this%v_1d + (1.0 - this%beta2) * (grads**2 - this%v_1d)

        params = params - lr_t * this%m_1d / (sqrt(this%v_1d) + 1e-7)
    end subroutine adam_update_1d

    subroutine adam_update_2d(this, params, grads)
      class(Adam), intent(inout) :: this
      real, intent(inout) :: params(:,:)
      real, intent(in) :: grads(:,:)
      real :: lr_t

        if (.not. allocated(this%m_2d)) then
            allocate(this%m_2d(size(params, 1), size(params, 2)))
            allocate(this%v_2d(size(params, 1), size(params, 2)))
            this%m_2d = 0.0
            this%v_2d = 0.0
        end if

        this%iter = this%iter + 1
        lr_t = this%lr * sqrt(1.0 - this%beta2**this%iter) / (1.0 - this%beta1**this%iter)

        this%m_2d = this%m_2d + (1.0 - this%beta1) * (grads - this%m_2d)
        this%v_2d = this%v_2d + (1.0 - this%beta2) * (grads**2 - this%v_2d)

        params = params - lr_t * this%m_2d / (sqrt(this%v_2d) + 1e-7)
    end subroutine adam_update_2d

end module optimizers