module MultiLayerNetModule
   use, intrinsic :: iso_fortran_env, only: real64, error_unit
   !use AffineLayerModule, only: AffineLayer
   !use ReluLayerModule, only: ReluLayer
   !use SoftmaxWithLossModule, only: SoftmaxWithLossLayer
   use Layers
   use OptimizersModule, only: OptimizerBase
   implicit none
   private

   public :: ActivationType
   public :: MultiLayerNet

   enum, bind(c)
      enumerator :: Activation_Relu = 1
      enumerator :: Activation_Sigmoid = 2
   end enum

   type :: ActivationType
      integer :: value = Activation_Relu
   end type ActivationType

   type :: MultiLayerNet
      integer :: input_size = 0
      integer :: output_size = 0
      integer :: hidden_layer_num = 0
      real(real64) :: weight_decay_lambda = 0.0_real64

      integer, allocatable :: hidden_sizes(:)

      ! Parameters packed for optimizer: weights [layer, in, out], biases [layer, out]
      real(real64), allocatable :: w(:, :, :)
      real(real64), allocatable :: b(:, :)

      ! Layers: Affine layers + activations
      type(AffineLayer), allocatable :: affine_layers(:)
      type(ReluLayer), allocatable :: relu_layers(:)

      type(SoftmaxWithLossLayer) :: last_layer

      class(OptimizerBase), allocatable :: optimizer

      type(ActivationType) :: activation
      
   contains
      procedure :: init => mln_init
      procedure :: predict => mln_predict
      procedure :: loss => mln_loss
      procedure :: accuracy => mln_accuracy
      procedure :: update => mln_update
      procedure :: gradient => mln_gradient
      procedure :: reset => mln_reset
   end type MultiLayerNet

contains

   subroutine mln_reset(self)
      class(MultiLayerNet), intent(inout) :: self

      if (allocated(self%hidden_sizes)) deallocate(self%hidden_sizes)
      if (allocated(self%w)) deallocate(self%w)
      if (allocated(self%b)) deallocate(self%b)

      if (allocated(self%affine_layers)) deallocate(self%affine_layers)
      if (allocated(self%relu_layers)) deallocate(self%relu_layers)

      call self%last_layer%reset()

      if (allocated(self%optimizer)) then
         call self%optimizer%reset()
         deallocate(self%optimizer)
      end if

      self%input_size = 0
      self%output_size = 0
      self%hidden_layer_num = 0
      self%weight_decay_lambda = 0.0_real64
      self%activation%value = Activation_Relu
   end subroutine mln_reset

   subroutine mln_init(self, input_size, hidden_sizes, output_size, weight_decay_lambda, activation, optimizer)
      class(MultiLayerNet), intent(inout) :: self
      integer, intent(in) :: input_size
      integer, intent(in) :: hidden_sizes(:)
      integer, intent(in) :: output_size
      real(real64), intent(in) :: weight_decay_lambda
      type(ActivationType), intent(in) :: activation
      class(OptimizerBase), allocatable, intent(inout) :: optimizer

      integer :: total_layers, max_in, max_out, i
      integer :: layer_in, layer_out

      call self%reset()

      self%input_size = input_size
      self%output_size = output_size
      self%hidden_layer_num = size(hidden_sizes)
      self%weight_decay_lambda = weight_decay_lambda
      self%activation = activation

      allocate(self%hidden_sizes(self%hidden_layer_num))
      self%hidden_sizes = hidden_sizes

      total_layers = self%hidden_layer_num + 1

      allocate(self%affine_layers(total_layers))
      allocate(self%relu_layers(self%hidden_layer_num))

      ! Pack parameters into rectangular 3D/2D arrays (works with previous OptimizersModule).
      max_in = input_size
      do i = 1, self%hidden_layer_num
         max_in = max(max_in, self%hidden_sizes(i))
      end do

      max_out = output_size
      do i = 1, self%hidden_layer_num
         max_out = max(max_out, self%hidden_sizes(i))
      end do

      allocate(self%w(total_layers, max_in, max_out))
      allocate(self%b(total_layers, max_out))
      self%w = 0.0_real64
      self%b = 0.0_real64

      call init_weights(self%w, self%b, input_size, self%hidden_sizes, output_size, activation)

      ! Initialize affine layer views (each gets a tight slice copied into its own allocatables).
      do i = 1, total_layers
         call get_layer_dims(i, input_size, self%hidden_sizes, output_size, layer_in, layer_out)

         call self%affine_layers(i)%init(self%w(i, 1:layer_in, 1:layer_out), self%b(i, 1:layer_out))
      end do

      ! take ownership of optimizer polymorphic instance
      call move_alloc(optimizer, self%optimizer)
   end subroutine mln_init
   

   function mln_predict(self, x) result(y)
      class(MultiLayerNet), intent(inout) :: self
      real(real64), intent(in) :: x(:, :)
      real(real64), allocatable :: y(:, :)

      real(real64), allocatable :: tmp(:, :)
      integer :: i

      tmp = x

      do i = 1, self%hidden_layer_num
         tmp = self%affine_layers(i)%forward(tmp)

         !select case (self%activation%value)
         !case (Activation_Relu)
            tmp = self%relu_layers(i)%forward(tmp)
         !case default
         !   write(error_unit, '(a)') 'MultiLayerNet.predict: Sigmoid activation not implemented in Fortran layers yet.'
         !   error stop 1
         !end select
      end do

      tmp = self%affine_layers(self%hidden_layer_num + 1)%forward(tmp)

      y = tmp
   end function mln_predict
   

   function mln_loss(self, x, t) result(loss)
      class(MultiLayerNet), intent(inout) :: self
      real(real64), intent(in) :: x(:, :), t(:, :)
      real(real64) :: loss

      real(real64), allocatable :: y(:, :)
      real(real64) :: weight_decay
      integer :: i, layer_in, layer_out

      y = self%predict(x)

      weight_decay = 0.0_real64
      do i = 1, self%hidden_layer_num + 1
         call get_layer_dims(i, self%input_size, self%hidden_sizes, self%output_size, layer_in, layer_out)
         weight_decay = weight_decay + 0.5_real64 * self%weight_decay_lambda * sum(self%w(i, 1:layer_in, 1:layer_out)**2)
      end do

      loss = self%last_layer%forward(y, t) + weight_decay
   end function mln_loss
   

   subroutine mln_accuracy(self, x, t, acc, errors, indexes)
      class(MultiLayerNet), intent(inout) :: self
      real(real64), intent(in) :: x(:, :), t(:, :)
      real(real64), intent(out) :: acc
      integer, allocatable, intent(out) :: errors(:), indexes(:)

      real(real64), allocatable :: y(:, :)
      integer :: batch, classes, i, y_pred, t_index, correct, err_count

      y = self%predict(x)

      batch = size(y, 1)
      classes = size(y, 2)

      correct = 0
      err_count = 0
      allocate(errors(batch))
      allocate(indexes(batch))

      do i = 1, batch
         y_pred = argmax_1based(y(i, :)) - 1   ! return 0-based like C# MaximumIndex()

         if (size(t, 2) == 1) then
            t_index = int(t(i, 1))
         else
            t_index = argmax_1based(t(i, :)) - 1
         end if

         if (y_pred == t_index) then
            correct = correct + 1
         else
            err_count = err_count + 1
            errors(err_count) = y_pred
            indexes(err_count) = i - 1  ! store 0-based index (matches C# usage)
         end if
      end do

      if (err_count == 0) then
         deallocate(errors);    deallocate(indexes)
         allocate(errors(0));   allocate(indexes(0))
      else
         errors = errors(1:err_count)
         indexes = indexes(1:err_count)
      end if

      acc = real(correct, real64) / real(batch, real64)
   end subroutine mln_accuracy
   

   subroutine mln_update(self, x, t)
      class(MultiLayerNet), intent(inout) :: self
      real(real64), intent(in) :: x(:, :), t(:, :)
      real(real64), allocatable :: dw(:, :, :), db(:, :)
      
      call self%gradient(x, t, dw, db)

      call self%optimizer%update(self%w, self%b, dw, db)

      call sync_params_into_layers(self)
   end subroutine mln_update
   

   subroutine mln_gradient(self, x, t, dw, db)
      class(MultiLayerNet), intent(inout) :: self
      real(real64), intent(in) :: x(:, :), t(:, :)
      real(real64), allocatable, intent(out) :: dw(:, :, :), db(:, :)

      real(real64), allocatable :: dout(:, :), a(:, :)
      integer :: total_layers, i, layer_in, layer_out      
      real(real64) :: loss_value

      total_layers = self%hidden_layer_num + 1

      loss_value = self%loss(x, t) ! forward pass; caches are stored in layers; last_layer stores y/t

      dout = self%last_layer%backward(1.0_real64)

      ! Backprop: last affine, then repeating activation+affine
      dout = self%affine_layers(total_layers)%backward(dout)

      do i = self%hidden_layer_num, 1, -1
         !select case (self%activation%value)
         !case (Activation_Relu)
            dout = self%relu_layers(i)%backward(dout)
         !case default
         !   write(error_unit, '(a)') 'MultiLayerNet.gradient: Sigmoid activation not implemented in Fortran layers yet.'
         !   error stop 1
         !end select

         dout = self%affine_layers(i)%backward(dout)
      end do

      allocate(dw(size(self%w, 1), size(self%w, 2), size(self%w, 3)))
      allocate(db(size(self%b, 1), size(self%b, 2)))
      dw = 0.0_real64
      db = 0.0_real64

      ! Collect gradients + weight decay term: dW = layer.dW + lambda * W
      do i = 1, total_layers
         call get_layer_dims(i, self%input_size, self%hidden_sizes, self%output_size, layer_in, layer_out)

         if (.not. allocated(self%affine_layers(i)%dw) .or. .not. allocated(self%affine_layers(i)%db)) then
            write(error_unit, '(a)') 'MultiLayerNet.gradient: missing affine gradients.'
            error stop 1
         end if

         dw(i, 1:layer_in, 1:layer_out) = self%affine_layers(i)%dw + self%weight_decay_lambda * self%w(i, 1:layer_in, 1:layer_out)
         db(i, 1:layer_out) = self%affine_layers(i)%db
      end do
   end subroutine mln_gradient
   

   subroutine sync_params_into_layers(self)
      class(MultiLayerNet), intent(inout) :: self
      integer :: i, layer_in, layer_out, total_layers

      total_layers = self%hidden_layer_num + 1

      do i = 1, total_layers
         call get_layer_dims(i, self%input_size, self%hidden_sizes, self%output_size, layer_in, layer_out)
         call self%affine_layers(i)%init(self%w(i, 1:layer_in, 1:layer_out), self%b(i, 1:layer_out))
      end do
   end subroutine sync_params_into_layers
   

   subroutine get_layer_dims(layer_idx, input_size, hidden_sizes, output_size, layer_in, layer_out)
      integer, intent(in) :: layer_idx, input_size, hidden_sizes(:), output_size
      integer, intent(out) :: layer_in, layer_out

      if (layer_idx == 1) then
         layer_in = input_size
         if (size(hidden_sizes) >= 1) then
            layer_out = hidden_sizes(1)
         else
            layer_out = output_size
         end if
      else if (layer_idx <= size(hidden_sizes)) then
         layer_in = hidden_sizes(layer_idx - 1)
         layer_out = hidden_sizes(layer_idx)
      else
         layer_in = hidden_sizes(size(hidden_sizes))
         layer_out = output_size
      end if
   end subroutine get_layer_dims
   

   integer function argmax_1based(v) result(idx)
      real(real64), intent(in) :: v(:)
      integer :: i
      real(real64) :: best

      idx = 1
      best = v(1)
      do i = 2, size(v)
         if (v(i) > best) then
            best = v(i)
            idx = i
         end if
      end do
   end function argmax_1based
   

   subroutine init_weights(w, b, input_size, hidden_sizes, output_size, activation)
      real(real64), intent(inout) :: w(:, :, :), b(:, :)
      integer, intent(in) :: input_size, output_size, hidden_sizes(:)
      type(ActivationType), intent(in) :: activation

      integer :: total_layers, i, layer_in, layer_out
      real(real64) :: scale

      total_layers = size(hidden_sizes) + 1

      do i = 1, total_layers
         call get_layer_dims(i, input_size, hidden_sizes, output_size, layer_in, layer_out)
         
         scale = sqrt(2.0_real64 / real(layer_in, real64))

         !scale = 0.01_real64
         !if (activation%value == Activation_Relu) then
         !   scale = sqrt(2.0_real64 / real(layer_in, real64))
         !else if (activation%value == Activation_Sigmoid) then
         !   scale = sqrt(1.0_real64 / real(layer_in, real64))
         !end if

         call fill_randn(w(i, 1:layer_in, 1:layer_out), scale)
         b(i, 1:layer_out) = 0.0_real64
      end do
   end subroutine init_weights

     subroutine fill_randn(a, stddev)
      real(real64), parameter :: eps = 1.0e-8_real64
      real(real64), intent(inout) :: a(:, :)
      real(real64), intent(in) :: stddev

      integer :: i, j
      real(real64) :: u1, u2, r, theta, z
      real(real64), parameter :: pi = acos(-1.0_real64)

      do i = 1, size(a, 1)
         do j = 1, size(a, 2)
            call random_number(u1)
            call random_number(u2)
            if (u1 < eps) u1 = eps
            r = sqrt(-2.0_real64 * log(u1))
            theta = 2.0_real64 * pi * u2
            z = r * cos(theta)
            a(i, j) = z * stddev
         end do
      end do
   end subroutine fill_randn
  

end module MultiLayerNetModule