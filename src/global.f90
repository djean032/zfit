module global
  use minpack_module
  use iso_fortran_env, only: dp => real64
  implicit none

  real(dp), parameter :: c = 299792458_dp
  real(dp), parameter :: h = 6.62607015e-34_dp
  real(dp), parameter :: pi = 4.0_dp * atan(1.0_dp)

contains
  function linspace(first, last, n)
    real(dp), intent(in) :: first
    real(dp), intent(in) :: last
    integer, intent(in) :: n
    integer :: i
    real(dp) :: span
    real(dp) :: linspace(n)

    span = last - first

    if (n .eq. 0) return

    if (n .eq. 1) then
      linspace(1) = first
      return
    end if

    do i = 1, n
      linspace(i) = first + span * (i - 1) / (n - 1)
    end do
  end function linspace
end module global

