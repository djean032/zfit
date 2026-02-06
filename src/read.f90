module file_ops
  use global
  use stdlib_io, only: loadtxt
  implicit none

  private

  public :: read_exp

contains

  function read_exp(name) result(exp_data)
    character(len=*), intent(in) :: name
    real(8), dimension(:, :), allocatable :: exp_data
    integer :: i, io, nlines, skipnum, size, exp_data_size
    character(len=100) :: tmp_line
    real(8), dimension(4) :: data

    inquire (file=name, size=size)
    open (unit=10, file=name, iostat=io, status="old")

    if (io /= 0) then
      print *, "Error opening file"
      stop
    end if

    nlines = 0
    do
      read (10, *, iostat=io)
      if (io /= 0) exit
      nlines = nlines + 1
    end do
    close (10)

    open (unit=10, file=name, iostat=io, status="old")
    do i = 1, nlines
      read (10, *, iostat=io) tmp_line
      if (io /= 0) exit
      if (tmp_line(1:6) == "z(mm)") then
        skipnum = i
        exit
      end if
    end do

    do i = 1, nlines
      read (10, *, iostat=io) tmp_line
      select case (io)
      case (0)
        exp_data_size = i
      case (iostat_end)
        exit
      case default
        print *, "Error reading file"
        stop
      end select
    end do

    close (10)
    open (unit=10, file=name, iostat=io, status="old")

    do i = 1, nlines
      read (10, *, iostat=io) tmp_line
      if (io /= 0) exit
      if (tmp_line(1:6) == "z(mm)") then
        exit
      end if
    end do
    allocate (exp_data(exp_data_size, 4))
    do i = 1, nlines
      read (10, *, iostat=io) data
      select case (io)
      case (0)
        exp_data(i, :) = data
      case (iostat_end)
        exit
      case default
        print *, "Error reading file"
        stop
      end select
    end do
  end function read_exp

end module file_ops

