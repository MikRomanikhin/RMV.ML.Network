module ParserModule
    implicit none
    private

    public :: Parser   

    !----------------------------------------------------------------------
    ! Parser Type
    !----------------------------------------------------------------------
    type :: Parser
        integer :: outputs = 10 ! Default to 10 for MNIST
    contains
        procedure :: Run => parser_run
        procedure, private :: binary_vector
    end type Parser

   contains

    ! ==========================================
    ! Parser Methods
    ! ==========================================
    
    !----------------------------
    !> Builds Train and Test data
    !----------------------------
    subroutine parser_run(this, lines, input_data, output_data)
        class(Parser), intent(in) :: this
        character(len=*), intent(in) :: lines(:)
        real, allocatable, intent(out) :: input_data(:,:)
        real, allocatable, intent(out) :: output_data(:,:)        
        integer :: n_lines, n_cols, n_features, i
        real, allocatable :: row_data(:)
        
        n_lines = size(lines)
        if (n_lines == 0) return
                
        n_cols = count_columns(lines(1)) ! Determine columns from the first line
        n_features = n_cols - 1
        
        allocate(input_data(n_lines, n_features))
        allocate(output_data(n_lines, this%outputs))
        allocate(row_data(n_cols))
        
        do i = 1, n_lines            
            read(lines(i), *) row_data   ! list-directed read to handle comma-separated values
            
            input_data(i, :) = row_data(2:n_cols) / 255.0d0 ! Normalize input to [0, 1] 
                        
            output_data(i, :) = this%binary_vector(int(row_data(1))) ! Parse target output to binary one-hot vector
        end do
    end subroutine parser_run
    
    !------------------------------------------------------
    !> Helper: Counts the number of comma-separated columns
    !------------------------------------------------------
    pure function count_columns(line) result(count)
        character(len=*), intent(in) :: line
        integer :: i, count
        count = 1
        do i = 1, len_trim(line)
            if (line(i:i) == ',') count = count + 1
        end do
    end function count_columns
    
    !---------------------------
    !> Binary mapping for output
    !---------------------------
    function binary_vector(this, val, min_val, max_val) result(res)
      class(Parser), intent(in) :: this
      integer, intent(in) :: val
      real, intent(in), optional :: min_val, max_val
      real, allocatable :: res(:)
        
        real :: actual_min, actual_max
        integer :: fort_index
        
        actual_min = 0.0
        if (present(min_val)) actual_min = min_val
        
        actual_max = 1.0
        if (present(max_val)) actual_max = max_val
        
        allocate(res(this%outputs))
        res = actual_min        
        
        fort_index = val + 1 
        
        if (fort_index > 0 .and. fort_index <= this%outputs) res(fort_index) = actual_max        
    end function binary_vector   

end module ParserModule