module ControllerModule
    use DataSetModule
    use MultiLayerNetModule
    implicit none
    private

    public :: Controller, AppSettings

    !----------------------------------------------------------------------
    ! Controller Type
    !----------------------------------------------------------------------
    type :: Controller
        type(DataSet), pointer :: train_set => null()
        type(DataSet), pointer :: test_set => null()
        type(AppSettings) :: settings
    contains
        procedure :: run => controller_run
    end type Controller

contains

    !> Trains and evaluates a neural network using the configured datasets
    subroutine controller_run(this)
        class(Controller), intent(inout) :: this        
        type(MultiLayer) :: network
        integer :: clock_start, clock_curr, clock_rate
        real :: elapsed_sec, max_accuracy, loss, train_acc, test_acc
        integer :: i, stagnation, best_iter, error_unit, index_unit
        
        real, allocatable :: x_batch(:,:), t_batch(:,:)
        real, allocatable :: val_x(:,:), val_t(:,:)
        integer, allocatable :: errors(:), indexes(:)
        character(len=80) :: line_separator = repeat("-", 40)
       
        call system_clock(count_rate=clock_rate)
        call system_clock(count=clock_start)
        
        max_accuracy = -huge(1.0d0)
        stagnation = 0
        best_iter = 0
                
        val_x = this%test_set%source
        val_t = this%test_set%target
        
        call system_clock(count=clock_curr)
        elapsed_sec = real(clock_curr - clock_start, 8) / real(clock_rate, 8)
        print '("Loaded data sets. Time:", F6.2, " sec")', elapsed_sec
               
        call network%Init(this%settings)  ! Initialize network 
                
        do i = 0, this%settings%iterations - 1  ! Training loop
            
            call this%train_set%GetRandomTrain( this%settings%batch, x_batch, t_batch )
            
            call network%update(x_batch, t_batch)
            loss = network%loss(x_batch, t_batch)
            call network%Accuracy( x_batch, t_batch, train_acc ) ! Ignoring detailed errors for train
            
            call system_clock(count=clock_curr)
            elapsed_sec = real(clock_curr - clock_start, 8) / real(clock_rate, 8)
            
            if (mod(i, this%settings%print_interval) == 0) &
                print '("iter:", I0, " loss:", F8.4, "  accuracy:", F8.4, "  Time=", A)', i, loss, train_acc, format_time(int(elapsed_sec))           
            
            ! Validation testing at configured intervals
            if( mod(i, this%settings%epoch) == 0 ) then  
                call network%Accuracy( val_x, val_t, test_acc, errors, indexes )
                
                print '("iter:", I0, "  validation:", F8.4, "  Time=", A)', i, test_acc, format_time(int(elapsed_sec))
                print *, trim(line_separator)
                
                !if( test_acc > max_accuracy ) then
                !    max_accuracy = test_acc;   stagnation = 0;   best_iter = i                                       
                !    open(newunit=error_unit, file=trim(this%settings%error_path), status='replace', action='write')
                !    write(error_unit, *) errors
                !    close(error_unit)                                       
                !    open(newunit=index_unit, file=trim(this%settings%index_path), status='replace', action='write')
                !    write(index_unit, *) indexes
                !    close(index_unit)                    
                !    cycle
                !end if
                
                stagnation = stagnation + 1
                if( stagnation > this%settings%stagnation ) then
                    print '("Stopping at ", I0, " due to stagnation. Accuracy: ", F8.4, &
                            &" iteration ", I0, ". Time=", A)', i, max_accuracy, best_iter, format_time(int(elapsed_sec))
                    exit
                end if
                
            end if
        end do
        
        contains 
   
   function format_time( total_seconds ) result( time_str )
    integer, intent(in) :: total_seconds
    integer :: hours, minutes, seconds
    character(len=10) :: time_str
      hours   = total_seconds / 3600
      minutes = (total_seconds - hours * 3600) / 60
      seconds = total_seconds - hours * 3600 - minutes * 60
      write(time_str, '(I2.2, ":", I2.2, ":", I2.2)') hours, minutes, seconds
   end function format_time
        
    end subroutine controller_run   
   
   
   end module ControllerModule