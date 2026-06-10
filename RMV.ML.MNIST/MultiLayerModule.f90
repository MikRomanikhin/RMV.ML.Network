module MultiLayerNetModule
    use LayersModule
    implicit none
    private

    public :: MultiLayer, AppSettings

    !----------------------------------------------------------------------
    ! AppSettings Type
    !----------------------------------------------------------------------
    type :: AppSettings
       integer :: input_size = 784
       integer :: output_size = 10
       integer :: hidden_size_list(2) = [400, 100]
       integer :: iterations = 5000
       integer :: batch = 2000
       integer :: print_interval = 10
       integer :: epoch = 50
       integer :: stagnation = 20
       real :: learning_rate = 0.001
       real :: momentum = 0.5
       real :: weight_decay_lambda = 0.0001
       character(len=10) :: optimizer = "adam"
       character(len=256) :: error_path = "f-errors.txt"
       character(len=256) :: index_path = "f-indexes.txt"
    end type AppSettings

    type :: AdamState
        real, allocatable :: mW(:,:), vW(:,:)
        real, allocatable :: mB(:), vB(:)
    end type AdamState

    !----------------------------------------------------------------------
    ! MultiLayerNet Type
    !----------------------------------------------------------------------
    type :: MultiLayer
        integer :: input_size
        integer :: output_size
        integer :: hidden_layer_num
        real :: weight_decay_lambda
        real :: learning_rate
        character(len=10) :: activation_type
        character(len=10) :: optimizer_type
        integer :: iter
        integer, allocatable :: hidden_size_list(:)

        ! explicit arrays of layers.
        type(Affine), allocatable :: AffineLayers(:)
        type(Relu), allocatable :: ReluLayers(:)
        type(Softmax_With_Loss) :: LastLayer

        type(AdamState), allocatable :: adam_states(:)

    contains
        procedure :: Init => net_init
        procedure :: InitWeights => net_init_weights
        procedure :: Predict => net_predict
        procedure :: Loss => net_loss
        procedure :: Accuracy => net_accuracy
        procedure :: Gradient => net_gradient
        procedure :: Update => net_update
    end type MultiLayer


contains

    ! ==========================================
    ! Initialization
    ! ==========================================
    subroutine net_init( this, settings )
        class(MultiLayer), intent(inout) :: this
        type(AppSettings), intent(in) :: settings       

        this%input_size = settings%input_size
        this%output_size = settings%output_size
        this%hidden_layer_num = size(settings%hidden_size_list)
        
        allocate(this%hidden_size_list(this%hidden_layer_num))
        this%hidden_size_list = settings%hidden_size_list

        this%weight_decay_lambda = settings%weight_decay_lambda
        this%learning_rate = settings%learning_rate
        this%activation_type = 'relu'
        this%optimizer_type = settings%optimizer
        this%iter = 0

        allocate(this%AffineLayers(this%hidden_layer_num + 1))
        allocate(this%adam_states(this%hidden_layer_num + 1))
        allocate(this%ReluLayers(this%hidden_layer_num))

        call random_seed()  ! seed once before all weight initialization
        call this%InitWeights()
    end subroutine net_init

    !----------------------------------------------------------------------
    ! Initializes weights and biases for all layers using He initialization
    !----------------------------------------------------------------------
    subroutine net_init_weights( this )
        class(MultiLayer), intent(inout) :: this                
        integer, allocatable :: all_sizes(:)
        integer :: i, N_in, N_out
        real :: scale

        allocate(all_sizes(this%hidden_layer_num + 2))
        all_sizes(1) = this%input_size
        all_sizes(2:this%hidden_layer_num+1) = this%hidden_size_list
        all_sizes(this%hidden_layer_num+2) = this%output_size

        do i = 1, this%hidden_layer_num + 1
            N_in = all_sizes(i)
            N_out = all_sizes(i+1)

            scale = sqrt(2.0 / real(N_in))

            allocate(this%AffineLayers(i)%W(N_in, N_out))
            allocate(this%AffineLayers(i)%B(N_out))
            allocate(this%adam_states(i)%mW(N_in, N_out))
            allocate(this%adam_states(i)%vW(N_in, N_out))
            allocate(this%adam_states(i)%mB(N_out))
            allocate(this%adam_states(i)%vB(N_out))

            call random_normal_matrix(this%AffineLayers(i)%W, scale)
            this%AffineLayers(i)%B = 0.0
            this%adam_states(i)%mW = 0.0
            this%adam_states(i)%vW = 0.0
            this%adam_states(i)%mB = 0.0
            this%adam_states(i)%vB = 0.0
        end do
    end subroutine net_init_weights

    ! =====================
    ! Forward and Loss
    ! =====================
    subroutine net_predict( this, x, y )
        class(MultiLayer), intent(inout) :: this
        real, intent(in) :: x(:,:)
        real, allocatable, intent(out) :: y(:,:)
        real, allocatable :: temp_in(:,:), temp_out(:,:)
        integer :: i

        temp_in = x
        do i = 1, this%hidden_layer_num
            temp_out = this%AffineLayers(i)%forward(temp_in)
            call move_alloc(temp_out, temp_in)           ! avoids copy: moves allocation, frees old temp_in
            temp_out = this%ReluLayers(i)%forward(temp_in)
            call move_alloc(temp_out, temp_in)           ! avoids copy: moves allocation, frees old temp_in
        end do

        y = this%AffineLayers(this%hidden_layer_num + 1)%forward(temp_in) ! Last affine layer
    end subroutine net_predict


    real function net_loss(this, x, t)
        class(MultiLayer), intent(inout) :: this
        real, intent(in) :: x(:,:)
        real, intent(in) :: t(:,:)
        real, allocatable :: y(:,:)
        real, allocatable :: t_2d(:,:)
        real :: weight_decay
        integer :: i

        allocate(t_2d(size(t, 1), 1))
        t_2d(:, 1) = real(maxloc(t, dim=2) - 1)  ! vectorized: convert one-hot to 0-based label indices

        call this%predict(x, y)

        weight_decay = 0.0
        do i = 1, this%hidden_layer_num + 1
            weight_decay = weight_decay + 0.5 * this%weight_decay_lambda * sum(this%AffineLayers(i)%W**2)
        end do

        net_loss = this%LastLayer%forward(y, t_2d) + weight_decay
    end function net_loss

    ! ==========================================
    ! Accuracy and Backpropagation
    ! ==========================================
    subroutine net_accuracy( this, x, t, acc, errors, indexes )
        class(MultiLayer), intent(inout) :: this
        real, intent(in) :: x(:,:), t(:,:)
        real, intent(out) :: acc
        integer, allocatable, intent(out), optional :: errors(:), indexes(:)
        real, allocatable :: y(:,:)
        integer, allocatable :: y_argmax(:), t_argmax(:)
        integer :: i, n_errors, k

        call this%predict(x, y)
        allocate(y_argmax(size(x, 1)))
        allocate(t_argmax(size(x, 1)))

        y_argmax = maxloc(y, dim=2)   ! vectorized: index of max per row
        t_argmax = maxloc(t, dim=2)   ! vectorized: index of max per row

        acc = real(count(y_argmax == t_argmax)) / real(size(x, 1))

        if( present(errors) .and. present(indexes) ) then
            n_errors = count(y_argmax /= t_argmax)
            allocate(errors(n_errors))
            allocate(indexes(n_errors))
            k = 0
            do i = 1, size(x, 1)
                if (y_argmax(i) /= t_argmax(i)) then
                    k = k + 1
                    errors(k)  = y_argmax(i) - 1  ! store as 0-based predicted label
                    indexes(k) = i
                end if
            end do
        end if
    end subroutine net_accuracy
    

    subroutine net_gradient(this, x, t)
        class(MultiLayer), intent(inout) :: this
        real, intent(in) :: x(:,:), t(:,:)
        real :: dummy_loss
        real, allocatable :: dout(:,:), temp_dout(:,:)
        integer :: i

        dummy_loss = this%loss(x, t) ! Forward pass mapping

        dout = this%LastLayer%backward() ! Backward pass mapping  Initialize dout as 1.0 

        ! Backward through the last affine mapping
        temp_dout = this%AffineLayers(this%hidden_layer_num + 1)%backward(dout)
        call move_alloc(temp_dout, dout)   ! avoids copy: moves allocation, frees old dout
        this%AffineLayers(this%hidden_layer_num + 1)%dW = this%AffineLayers(this%hidden_layer_num + 1)%dW &
            + this%weight_decay_lambda * this%AffineLayers(this%hidden_layer_num + 1)%W

        ! Reversible propagation through hidden layers
        do i = this%hidden_layer_num, 1, -1
            temp_dout = this%ReluLayers(i)%backward(dout)
            call move_alloc(temp_dout, dout)   ! avoids copy: moves allocation, frees old dout

            temp_dout = this%AffineLayers(i)%backward(dout)
            call move_alloc(temp_dout, dout)   ! avoids copy: moves allocation, frees old dout

            this%AffineLayers(i)%dW = this%AffineLayers(i)%dW + this%weight_decay_lambda * this%AffineLayers(i)%W
        end do

    end subroutine net_gradient

    subroutine net_update(this, x, t)
        class(MultiLayer), intent(inout) :: this
        real, intent(in) :: x(:,:), t(:,:)
        integer :: i
        real :: lr_t
        real, parameter :: beta1 = 0.9d0, beta2 = 0.999d0
        real, parameter :: epsilon = 1e-8

        call this%gradient(x, t)

        this%iter = this%iter + 1

        if (this%optimizer_type == "adam") then
            lr_t = this%learning_rate * real(sqrt(1.0d0 - beta2**this%iter) / (1.0d0 - beta1**this%iter))
            do i = 1, this%hidden_layer_num + 1
                this%adam_states(i)%mW = beta1 * this%adam_states(i)%mW + (1.0 - beta1) * this%AffineLayers(i)%dW
                this%adam_states(i)%mB = beta1 * this%adam_states(i)%mB + (1.0 - beta1) * this%AffineLayers(i)%dB

                this%adam_states(i)%vW = beta2 * this%adam_states(i)%vW + (1.0 - beta2) * (this%AffineLayers(i)%dW**2)
                this%adam_states(i)%vB = beta2 * this%adam_states(i)%vB + (1.0 - beta2) * (this%AffineLayers(i)%dB**2)

                this%AffineLayers(i)%W = this%AffineLayers(i)%W - lr_t * this%adam_states(i)%mW / (sqrt(this%adam_states(i)%vW) + epsilon)
                this%AffineLayers(i)%B = this%AffineLayers(i)%B - lr_t * this%adam_states(i)%mB / (sqrt(this%adam_states(i)%vB) + epsilon)
            end do
        else
            do i = 1, this%hidden_layer_num + 1
                this%AffineLayers(i)%W = this%AffineLayers(i)%W - this%learning_rate * this%AffineLayers(i)%dW
                this%AffineLayers(i)%B = this%AffineLayers(i)%B - this%learning_rate * this%AffineLayers(i)%dB
            end do
        end if
    end subroutine net_update

    ! ==========================================
    ! Helper Function for Box-Muller Distribution
    ! ==========================================
    subroutine random_normal_matrix( mat, scale )
        real, intent(inout) :: mat(:,:)
        real, intent(in) :: scale
        integer :: total, k
        real :: u1, u2, r, angle, cached
        logical :: have_cached
        real, allocatable :: flat(:)
        real, parameter :: TWO_PI = 2.0 * 3.141592653589793

        total = size(mat, 1) * size(mat, 2)
        allocate(flat(total))
        have_cached = .false.
        cached = 0.0
        k = 1
        do while (k <= total)
            if (have_cached) then
                flat(k) = cached
                have_cached = .false.
                k = k + 1
            else
                call random_number(u1); call random_number(u2)
                if (u1 < 1e-15) u1 = 1e-15
                r     = scale * sqrt(-2.0 * log(u1))
                angle = TWO_PI * u2
                flat(k) = r * cos(angle)   ! first sample
                cached  = r * sin(angle)   ! second sample, reuses r and angle
                have_cached = .true.
                k = k + 1
            end if
        end do

        mat = reshape(flat, [size(mat, 1), size(mat, 2)])
    end subroutine random_normal_matrix

end module MultiLayerNetModule