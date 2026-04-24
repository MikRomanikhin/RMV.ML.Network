program main
    use MultiLayerNetModule, only: AppSettings
    use ControllerModule, only: Controller
    use ParserModule, only: Parser
    use DataSetModule, only: DataSet
    implicit none

    type(AppSettings) :: settings
    type(Parser) :: p
    type(DataSet), target :: train_set, test_set
    type(Controller) :: c
    
    character(len=8192), allocatable :: lines(:)
    integer :: file_unit, iostat, i, num_lines
    character(len=256) :: train_path = "d://MNIST/mnist_train.csv"
    character(len=256) :: test_path = "d://MNIST/mnist_test.csv"
    
    settings = AppSettings()  
    
    p%outputs = settings%output_size         
    
    lines = GetData( train_path )

    call p%Run( lines, train_set%source, train_set%target )
    deallocate( lines )        
   
    lines = GetData( test_path )

    call p%Run( lines, test_set%source, test_set%target )
    deallocate(lines)
    
    c%train_set => train_set
    c%test_set => test_set
    c%settings = settings

    call c%run()
    
   contains
   
   function GetData( file_path ) result( lines )
        character(len=*), intent(in) :: file_path
        integer :: num_lines, file_unit, iostat
        character(len=8192), allocatable :: lines(:)
        
        num_lines = 0
        open(newunit=file_unit, file=trim(file_path), status='old', iostat=iostat)
        if (iostat /= 0) then
            print *, "Failed to open dataset: ", trim(file_path)
            stop
        end if
        do
            read(file_unit, *, iostat=iostat) 
            if (iostat /= 0) exit
            num_lines = num_lines + 1
        end do
        rewind(file_unit)
    
      allocate(lines(num_lines))
      
      do i = 1, num_lines
         read(file_unit, '(A)') lines(i)
      end do
      close(file_unit)
   end function GetData

end program main