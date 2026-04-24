module DataSetModule
    implicit none
    private

    public :: DataSet

    !----------------------------------------------------------------------
    ! DataSet Type
    !----------------------------------------------------------------------
    type :: DataSet
        real, allocatable :: source(:,:)
        real, allocatable :: target(:,:)
    contains
        procedure :: GetRandomBatch => dataset_get_random_batch
        procedure :: GetRandomTrain => dataset_get_random_train
        procedure :: Match => dataset_match
        procedure :: GetMaxItemIndex => dataset_get_max_item_index
    end type DataSet

contains

    !> Builds a random batch of source data arrays and returns a new DataSet object
    function dataset_get_random_batch(this, batch_size) result(batch_dataset)
        class(DataSet), intent(in) :: this
        integer, intent(in) :: batch_size
        type(DataSet) :: batch_dataset
        
        integer :: n_samples, i
        integer, allocatable :: indices(:)
        
        n_samples = size(this%source, 1)
        allocate(indices(n_samples))
        call generate_shuffled_indices(n_samples, indices)
        
        allocate(batch_dataset%source(batch_size, size(this%source, 2)))
        allocate(batch_dataset%target(batch_size, size(this%target, 2)))
        
        do i = 1, batch_size
            batch_dataset%source(i, :) = this%source(indices(i), :)
            batch_dataset%target(i, :) = this%target(indices(i), :)
        end do
    end function dataset_get_random_batch


    !> Selects a random batch and returns them as two matrices (via intent out)
    subroutine dataset_get_random_train(this, batch_size, x_batch, t_batch)
        class(DataSet), intent(in) :: this
        integer, intent(in) :: batch_size
        real, allocatable, intent(out) :: x_batch(:,:)
        real, allocatable, intent(out) :: t_batch(:,:)
        
        integer :: n_samples, i
        integer, allocatable :: indices(:)
        
        n_samples = size(this%source, 1)
        allocate(indices(n_samples))
        
        call generate_shuffled_indices(n_samples, indices)
        
        allocate(x_batch(batch_size, size(this%source, 2)))
        allocate(t_batch(batch_size, size(this%target, 2)))
        
        do i = 1, batch_size
            x_batch(i, :) = this%source(indices(i), :)
            t_batch(i, :) = this%target(indices(i), :)
        end do
    end subroutine dataset_get_random_train


    !> Determines whether the maximum item index equals the provided index
    function dataset_match(this, row_index, target_idx) result(is_match)
        class(DataSet), intent(in) :: this
        integer, intent(in) :: row_index
        integer, intent(in) :: target_idx
        logical :: is_match
        
        is_match = (this%GetMaxItemIndex(row_index) == target_idx)
    end function dataset_match


    !> Calculates index of the maximum value within the item array at the specified position   
    function dataset_get_max_item_index(this, row_index) result(max_idx)
        class(DataSet), intent(in) :: this
        integer, intent(in) :: row_index
        integer :: max_idx        
        integer :: loc(1)
        
        ! maxloc returns an array, we take the first element for 1D slice
        loc = maxloc(this%target(row_index, :))
        max_idx = loc(1)
    end function dataset_get_max_item_index


    ! ==========================================
    ! Helper Methods
    ! ==========================================

    !> Generates an array of indices [1..n] and shuffles them randomly
    subroutine generate_shuffled_indices(n, indices)
        integer, intent(in) :: n
        integer, intent(out) :: indices(n)        
        integer :: i, j, temp
        real :: rand_val
                       
        indices = [(i, i=1,n)] ! Initialize indices        
        ! Fisher-Yates shuffle
        do i = n, 2, -1
            call random_number(rand_val)
            j = 1 + int(rand_val * i) ! random index between 1 and i                        
            temp = indices(i)
            indices(i) = indices(j)
            indices(j) = temp
        end do
    end subroutine generate_shuffled_indices

end module DataSetModule