! ----------------------------------------------------------------------------
! This module provides utility functions for the neural network implementation
! ----------------------------------------------------------------------------
module ToolsModule
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none

   contains
! ----------------------------------------------------------------
! Generates Gaussian random numbers using the Box-Muller transform
! ----------------------------------------------------------------
function GetGaussian() result(res)
   real :: res
   real :: u1, u2
   call random_number(u1)
   call random_number(u2)
   res = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2)
end function GetGaussian
   
end module ToolsModule