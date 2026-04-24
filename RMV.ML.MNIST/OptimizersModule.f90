module OptimizersModule
   use, intrinsic :: iso_fortran_env, only: real64, error_unit
   implicit none
   private

   public :: OptimizerBase
   public :: SgdOptimizer, MomentumOptimizer, NesterovOptimizer, AdaGradOptimizer, RmsPropOptimizer, AdamOptimizer

   real(real64), parameter :: eps = 1.0e-7_real64

   type, abstract :: OptimizerBase
   contains
      procedure(update_iface), deferred :: update
      procedure :: reset => optimizer_reset
   end type OptimizerBase
   

   abstract interface
      subroutine update_iface(self, w, b, dw, db)
         import :: OptimizerBase, real64
         class(OptimizerBase), intent(inout) :: self
         real(real64), intent(inout) :: w(:, :, :)   ! [layers, in, out]
         real(real64), intent(inout) :: b(:, :)      ! [layers, out]
         real(real64), intent(in) :: dw(:, :, :)
         real(real64), intent(in) :: db(:, :)
      end subroutine update_iface
   end interface
   

   type, extends(OptimizerBase) :: SgdOptimizer
      real(real64) :: rate = 0.01_real64
   contains
      procedure :: update => sgd_update
   end type SgdOptimizer
   

   type, extends(OptimizerBase) :: MomentumOptimizer
      real(real64) :: rate = 0.01_real64
      real(real64) :: momentum = 0.9_real64
      real(real64), allocatable :: vw(:, :, :)
      real(real64), allocatable :: vb(:, :)
   contains
      procedure :: update => momentum_update
      procedure :: reset => momentum_reset
   end type MomentumOptimizer
   

   type, extends(OptimizerBase) :: NesterovOptimizer
      real(real64) :: rate = 0.01_real64
      real(real64) :: momentum = 0.9_real64
      real(real64), allocatable :: vw(:, :, :)
      real(real64), allocatable :: vb(:, :)
   contains
      procedure :: update => nesterov_update
      procedure :: reset => nesterov_reset
   end type NesterovOptimizer
   

   type, extends(OptimizerBase) :: AdaGradOptimizer
      real(real64) :: rate = 0.01_real64
      real(real64), allocatable :: hw(:, :, :)
      real(real64), allocatable :: hb(:, :)
   contains
      procedure :: update => adagrad_update
      procedure :: reset => adagrad_reset
   end type AdaGradOptimizer
   

   type, extends(OptimizerBase) :: RmsPropOptimizer
      real(real64) :: rate = 0.01_real64
      real(real64) :: decay = 0.99_real64
      real(real64), allocatable :: hw(:, :, :)
      real(real64), allocatable :: hb(:, :)
   contains
      procedure :: update => rmsprop_update
      procedure :: reset => rmsprop_reset
   end type RmsPropOptimizer
   

   type, extends(OptimizerBase) :: AdamOptimizer
      real(real64) :: rate = 0.001_real64
      real(real64) :: beta1 = 0.9_real64
      real(real64) :: beta2 = 0.999_real64
      integer :: iter = 0
      real(real64), allocatable :: mw(:, :, :)
      real(real64), allocatable :: vw(:, :, :)
      real(real64), allocatable :: mb(:, :)
      real(real64), allocatable :: vb(:, :)
   contains
      procedure :: update => adam_update
      procedure :: reset => adam_reset
   end type AdamOptimizer

contains

   subroutine optimizer_reset(self)
      class(OptimizerBase), intent(inout) :: self
      ! default: no state
   end subroutine optimizer_reset

   subroutine ensure_state_3d(v, shape_like, name)
      real(real64), allocatable, intent(inout) :: v(:, :, :)
      real(real64), intent(in) :: shape_like(:, :, :)
      character(len=*), intent(in) :: name

      if (.not. allocated(v)) then
         allocate(v(size(shape_like, 1), size(shape_like, 2), size(shape_like, 3)))
         v = 0.0_real64
      else if (any(shape(v) /= shape(shape_like))) then
         write(error_unit, '(a)') trim(name)//': state shape mismatch.'
         error stop 1
      end if
   end subroutine ensure_state_3d

   subroutine ensure_state_2d(v, shape_like, name)
      real(real64), allocatable, intent(inout) :: v(:, :)
      real(real64), intent(in) :: shape_like(:, :)
      character(len=*), intent(in) :: name

      if (.not. allocated(v)) then
         allocate(v(size(shape_like, 1), size(shape_like, 2)))
         v = 0.0_real64
      else if (any(shape(v) /= shape(shape_like))) then
         write(error_unit, '(a)') trim(name)//': state shape mismatch.'
         error stop 1
      end if
   end subroutine ensure_state_2d
   

   ! ---------------- SGD --------------------------------------------
   ! No state to reset for SGD, so it uses the default optimizer_reset
   !------------------------------------------------------------------
   subroutine sgd_update(self, w, b, dw, db)
      class(SgdOptimizer), intent(inout) :: self
      real(real64), intent(inout) :: w(:, :, :)
      real(real64), intent(inout) :: b(:, :)
      real(real64), intent(in) :: dw(:, :, :)
      real(real64), intent(in) :: db(:, :)

      w = w - self%rate * dw
      b = b - self%rate * db
   end subroutine sgd_update
   

   !-------------------------------------------
   !               Momentum 
   !-------------------------------------------
   subroutine momentum_reset(self)
      class(MomentumOptimizer), intent(inout) :: self
      if (allocated(self%vw)) deallocate(self%vw)
      if (allocated(self%vb)) deallocate(self%vb)
   end subroutine momentum_reset

   subroutine momentum_update(self, w, b, dw, db)
      class(MomentumOptimizer), intent(inout) :: self
      real(real64), intent(inout) :: w(:, :, :)
      real(real64), intent(inout) :: b(:, :)
      real(real64), intent(in) :: dw(:, :, :)
      real(real64), intent(in) :: db(:, :)

      call ensure_state_3d(self%vw, w, 'Momentum.update(vW)')
      call ensure_state_2d(self%vb, b, 'Momentum.update(vB)')

      self%vw = self%momentum * self%vw - self%rate * dw
      w = w + self%vw

      self%vb = self%momentum * self%vb - self%rate * db
      b = b + self%vb
   end subroutine momentum_update
   

   !-------------------------------------------
   !               Nesterov 
   !-------------------------------------------
   subroutine nesterov_reset(self)
      class(NesterovOptimizer), intent(inout) :: self
      if (allocated(self%vw)) deallocate(self%vw)
      if (allocated(self%vb)) deallocate(self%vb)
   end subroutine nesterov_reset

   subroutine nesterov_update(self, w, b, dw, db)
      class(NesterovOptimizer), intent(inout) :: self
      real(real64), intent(inout) :: w(:, :, :)
      real(real64), intent(inout) :: b(:, :)
      real(real64), intent(in) :: dw(:, :, :)
      real(real64), intent(in) :: db(:, :)

      real(real64), allocatable :: vw_prev(:, :, :)
      real(real64), allocatable :: vb_prev(:, :)

      call ensure_state_3d(self%vw, w, 'Nesterov.update(vW)')
      call ensure_state_2d(self%vb, b, 'Nesterov.update(vB)')

      allocate(vw_prev(size(w, 1), size(w, 2), size(w, 3)))
      vw_prev = self%vw
      self%vw = self%momentum * self%vw - self%rate * dw
      w = w - self%momentum * vw_prev + (1.0_real64 + self%momentum) * self%vw

      allocate(vb_prev(size(b, 1), size(b, 2)))
      vb_prev = self%vb
      self%vb = self%momentum * self%vb - self%rate * db
      b = b - self%momentum * vb_prev + (1.0_real64 + self%momentum) * self%vb
   end subroutine nesterov_update
   

   !-------------------------------------------
   !               AdaGrad 
   !-------------------------------------------
   subroutine adagrad_reset(self)
      class(AdaGradOptimizer), intent(inout) :: self
      if (allocated(self%hw)) deallocate(self%hw)
      if (allocated(self%hb)) deallocate(self%hb)
   end subroutine adagrad_reset

   subroutine adagrad_update(self, w, b, dw, db)
      class(AdaGradOptimizer), intent(inout) :: self
      real(real64), intent(inout) :: w(:, :, :)
      real(real64), intent(inout) :: b(:, :)
      real(real64), intent(in) :: dw(:, :, :)
      real(real64), intent(in) :: db(:, :)

      call ensure_state_3d(self%hw, w, 'AdaGrad.update(hW)')
      call ensure_state_2d(self%hb, b, 'AdaGrad.update(hB)')

      self%hw = self%hw + dw * dw
      w = w - self%rate * dw / (sqrt(self%hw) + eps)

      self%hb = self%hb + db * db
      b = b - self%rate * db / (sqrt(self%hb) + eps)
   end subroutine adagrad_update

   !-------------------------------------------
   !               RMSprop 
   !-------------------------------------------
   subroutine rmsprop_reset(self)
      class(RmsPropOptimizer), intent(inout) :: self
      if (allocated(self%hw)) deallocate(self%hw)
      if (allocated(self%hb)) deallocate(self%hb)
   end subroutine rmsprop_reset

   subroutine rmsprop_update(self, w, b, dw, db)
      class(RmsPropOptimizer), intent(inout) :: self
      real(real64), intent(inout) :: w(:, :, :)
      real(real64), intent(inout) :: b(:, :)
      real(real64), intent(in) :: dw(:, :, :)
      real(real64), intent(in) :: db(:, :)

      call ensure_state_3d(self%hw, w, 'RmsProp.update(hW)')
      call ensure_state_2d(self%hb, b, 'RmsProp.update(hB)')

      self%hw = self%decay * self%hw + (1.0_real64 - self%decay) * (dw * dw)
      w = w - self%rate * dw / (sqrt(self%hw) + eps)

      self%hb = self%decay * self%hb + (1.0_real64 - self%decay) * (db * db)
      b = b - self%rate * db / (sqrt(self%hb) + eps)
   end subroutine rmsprop_update

   
   !-------------------------------------------
   !               Adam 
   !-------------------------------------------
   subroutine adam_reset(self)
      class(AdamOptimizer), intent(inout) :: self

      self%iter = 0
      if (allocated(self%mw)) deallocate(self%mw)
      if (allocated(self%vw)) deallocate(self%vw)
      if (allocated(self%mb)) deallocate(self%mb)
      if (allocated(self%vb)) deallocate(self%vb)
   end subroutine adam_reset

   subroutine adam_update(self, w, b, dw, db)
      class(AdamOptimizer), intent(inout) :: self
      real(real64), intent(inout) :: w(:, :, :)
      real(real64), intent(inout) :: b(:, :)
      real(real64), intent(in) :: dw(:, :, :)
      real(real64), intent(in) :: db(:, :)

      real(real64) :: lr_t
      real(real64) :: b1t, b2t

      call ensure_state_3d(self%mw, w, 'Adam.update(mW)')
      call ensure_state_3d(self%vw, w, 'Adam.update(vW)')
      call ensure_state_2d(self%mb, b, 'Adam.update(mB)')
      call ensure_state_2d(self%vb, b, 'Adam.update(vB)')

      self%iter = self%iter + 1

      b1t = 1.0_real64 - self%beta1**self%iter
      b2t = 1.0_real64 - self%beta2**self%iter
      lr_t = self%rate * sqrt(b2t) / b1t

      self%mw = self%beta1 * self%mw + (1.0_real64 - self%beta1) * dw
      self%vw = self%beta2 * self%vw + (1.0_real64 - self%beta2) * (dw * dw)
      w = w - lr_t * self%mw / (sqrt(self%vw) + eps)

      self%mb = self%beta1 * self%mb + (1.0_real64 - self%beta1) * db
      self%vb = self%beta2 * self%vb + (1.0_real64 - self%beta2) * (db * db)
      b = b - lr_t * self%mb / (sqrt(self%vb) + eps)
   end subroutine adam_update

end module OptimizersModule