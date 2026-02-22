module global
  use minpack_module
  use iso_c_binding
  implicit none

  real(c_double), parameter :: c = 299792458.0_c_double
  real(c_double), parameter :: h = 6.62607015e-34_c_double
  real(c_double), parameter :: pi = 4.0_c_double * atan(1.0_c_double)

contains
  function linspace(first, last, n)
    integer(c_int), intent(in) :: n
    real(c_double), intent(in) :: first, last
    integer(c_int) :: i
    real(c_double) :: linspace(n), span

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

