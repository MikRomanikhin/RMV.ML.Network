module MultiLayerNetModule
    use Layers
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
       integer :: stagnation = 10
       real :: learning_rate = 0.01
       real :: momentum = 0.5
       real :: weight_decay_lambda = 0.0001
       character(len=256) :: error_path = "f-errors.txt"
       character(len=256) :: index_path = "f-indexes.txt"
    end type AppSettings

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
        integer, allocatable :: hidden_size_list(:)
         
        ! explicit arrays of layers.
        type(Affine), allocatable :: AffineLayers(:)
        type(Relu), allocatable :: ReluLayers(:)
        type(SoftmaxWithLoss) :: LastLayer

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

        allocate(this%AffineLayers(this%hidden_layer_num + 1))
        
        allocate(this%ReluLayers(this%hidden_layer_num))

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
            
            call random_normal_matrix(this%AffineLayers(i)%W, scale)
            this%AffineLayers(i)%B = 0.0
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
            deallocate(temp_in)            
            temp_in = this%ReluLayers(i)%forward(temp_out)
            deallocate(temp_out)
        end do
        
        y = this%AffineLayers(this%hidden_layer_num + 1)%forward(temp_in) ! Last affine layer
    end subroutine net_predict
    

    real function net_loss(this, x, t)
        class(MultiLayer), intent(inout) :: this
        real, intent(in) :: x(:,:)
        real, intent(in) :: t(:,:)
        real, allocatable :: y(:,:)
        integer, allocatable :: t_1d(:)
        real :: weight_decay
        integer :: i

        allocate(t_1d(size(t, 1)))
        do i = 1, size(t, 1)
            t_1d(i) = maxloc(t(i, :), 1)
        end do

        call this%predict(x, y)

        weight_decay = 0.0
        do i = 1, this%hidden_layer_num + 1
            weight_decay = weight_decay + 0.5 * this%weight_decay_lambda * sum(this%AffineLayers(i)%W**2)
        end do

        net_loss = this%LastLayer%forward(y, t_1d) + weight_decay
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
        integer :: i

        call this%predict(x, y)
        allocate(y_argmax(size(x, 1)))
        allocate(t_argmax(size(x, 1)))

        do i = 1, size(x, 1)
            y_argmax(i) = maxloc(y(i, :), 1)
            t_argmax(i) = maxloc(t(i, :), 1)
        end do

        acc = real(count(y_argmax == t_argmax)) / real(size(x, 1))
        
        if( present(errors) .and. present(indexes) ) then
           allocate(errors(size(x,1)))
           allocate(indexes(size(x,1)))
           errors = 0;   indexes = 0
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
        deallocate(dout)
        dout = temp_dout
        this%AffineLayers(this%hidden_layer_num + 1)%dW = this%AffineLayers(this%hidden_layer_num + 1)%dW &
            + this%weight_decay_lambda * this%AffineLayers(this%hidden_layer_num + 1)%W

        ! Reversible propagation through hidden layers
        do i = this%hidden_layer_num, 1, -1
            temp_dout = this%ReluLayers(i)%backward(dout)
            deallocate(dout)

            dout = this%AffineLayers(i)%backward(temp_dout)
            deallocate(temp_dout)

            ! Gather weights explicitly (store in layer fields)
            this%AffineLayers(i)%dW = this%AffineLayers(i)%dW + this%weight_decay_lambda * this%AffineLayers(i)%W
        end do

    end subroutine net_gradient

    subroutine net_update(this, x, t)
        class(MultiLayer), intent(inout) :: this
        real, intent(in) :: x(:,:), t(:,:)
        integer :: i

        call this%gradient(x, t)

        do i = 1, this%hidden_layer_num + 1
            this%AffineLayers(i)%W = this%AffineLayers(i)%W - this%learning_rate * this%AffineLayers(i)%dW
            this%AffineLayers(i)%B = this%AffineLayers(i)%B - this%learning_rate * this%AffineLayers(i)%dB
        end do
    end subroutine net_update

    ! ==========================================
    ! Helper Function for Box-Muller Distribution
    ! ==========================================
    subroutine random_normal_matrix( mat, scale )
        real, intent(inout) :: mat(:,:)
        real, intent(in) :: scale
        integer :: i, j
        real :: u1, u2, z0

        call random_seed()
        do j = 1, size(mat, 2)
            do i = 1, size(mat, 1)
                call random_number( u1 );   call random_number( u2 )               
                if( u1 < 1e-15 ) u1 = 1e-15   ! Prevent log(0)
                z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * u2)
                mat(i, j) = scale * z0
            end do
        end do
    end subroutine random_normal_matrix

end module MultiLayerNetModule