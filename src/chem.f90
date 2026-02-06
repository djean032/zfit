module chem
  use global
  implicit none

  private

  abstract interface
        pure function expr_f(z_positions, z_samples, times, pars, initial_population, &
                         laser_intensities) result(y)
      import :: dp
      implicit none
      real(dp), intent(in) :: pars(:)
      real(dp), intent(in) :: z_positions(:)
      real(dp), intent(in) :: z_samples(:)
      real(dp), intent(in) :: times(:)
      real(dp), intent(in) :: initial_population(:)
      real(dp), intent(in) :: laser_intensities(:, :)
      real(dp) :: y(size(z_positions))
    end function expr_f
  end interface

  abstract interface
    pure subroutine odepack_f(neq, time, y, ydot)
      import :: dp
      implicit none
      integer, intent(in) :: neq
      real(dp), intent(in) :: time
      real(dp), intent(in) :: y(15)
      real(dp), intent(out) :: ydot(neq)
    end subroutine

    pure subroutine odepack_jac()
    end subroutine
  end interface
  interface
    pure subroutine DLSODA(f, neq, y, t_in, t_out, itol, rtol, atol, itask, &
                           istate, iopt, rwork, lrw, iwork, liw, jac, jt)
      import :: odepack_f, odepack_jac
      import :: dp
      implicit none
      integer, intent(in) :: neq, itol, itask, iopt, lrw, liw, jt
      integer, intent(inout) :: istate
      real(dp), intent(in) :: rtol, atol
      real(dp), intent(inout) :: t_in
      real(dp), intent(in) :: t_out
      real(dp), intent(inout) :: y(neq)
      real(dp), intent(inout) :: rwork(lrw)
      integer, intent(inout) :: iwork(liw)
      procedure(odepack_f) :: f
      procedure(odepack_jac) :: jac
    end subroutine

  end interface

  public :: rhs_rates, rhs_intensity, jdum, solve_rates, solve_intensity, &
            solve_system, fit_scan

contains
  pure subroutine rhs_rates(neq, time, y, ydot)
    integer, intent(in) :: neq
    real(dp), intent(in) :: time
    real(dp), dimension(15), intent(in) :: y
    real(dp), dimension(neq), intent(out) :: ydot
    ydot(1) = -(y(8) * y(1) * y(6)) / (h * y(7)) &
              + (y(2) / y(11)) &
              + (y(4) / y(14))
    ydot(2) = (y(8) * y(1) * y(6)) / (h * y(7)) &
              - (y(2) / y(11)) &
              - (y(9) * y(2) * y(6)) / (h * y(7)) &
              + (y(3) / y(13)) &
              - (y(2) / y(12))
    ydot(3) = (y(9) * y(2) * y(6)) / (h * y(7)) &
              - (y(3) / y(13))
    ydot(4) = -(y(10) * y(4) * y(6)) / (h * y(7)) &
              + (y(5) / y(15)) &
              + (y(2) / y(12)) &
              - (y(4) / y(14))
    ydot(5) = (y(10) * y(4) * y(6)) / (h * y(7)) &
              - (y(5) / y(15))
  end subroutine rhs_rates

  pure subroutine rhs_intensity(neq, time, y, ydot)
    integer, intent(in) :: neq
    real(dp), intent(in) :: time
    real(dp), dimension(15), intent(in) :: y
    real(dp), dimension(neq), intent(out) :: ydot
    ydot(1) = -y(8) * y(2) * y(1) &
              - y(9) * y(3) * y(1) &
              - y(10) * y(5) * y(1)
  end subroutine rhs_intensity

  pure subroutine jdum()
  end subroutine jdum

  pure function solve_rates(y, t_in, t_out) result(y_ret)
    integer :: neq, itol, itask, iopt, lrw, &
               liw, jt, istate
    real(dp), intent(in) :: t_in, t_out
    real(dp) :: t_ret, tout_ret, rtol, atol, rwork(102)
    real(dp), intent(in) :: y(15)
    real(dp) :: y_ret(15)
    integer :: iwork(25)
    t_ret = t_in
    tout_ret = t_out
    neq = 5
    lrw = 102
    liw = 25
    rtol = 0.5e-6_dp
    atol = 0.5e-6_dp
    itol = 1
    itask = 1
    istate = 1
    iopt = 0
    lrw = 102
    liw = 25
    jt = 2
    y_ret = y
    call DLSODA(rhs_rates, neq, y_ret, t_ret, tout_ret, itol, &
                rtol, atol, itask, istate, iopt, rwork, lrw, iwork, &
                liw, jdum, jt)
  end function solve_rates

  pure function solve_intensity(y, t_in, t_out) result(y_ret)
    integer :: neq, itol, itask, iopt, lrw, &
               liw, jt, istate
    real(dp), intent(in) :: t_in, t_out
    real(dp), intent(in) :: y(15)
    real(dp) :: t_ret, tout_ret, rtol, atol, rwork(102)
    real(dp) :: y_ret(15)
    integer :: iwork(25)
    t_ret = t_in
    tout_ret = t_out
    neq = 1
    lrw = 102
    liw = 25
    rtol = 0.5e-6_dp
    atol = 0.5e-6_dp
    itol = 1
    itask = 1
    istate = 1
    iopt = 0
    lrw = 102
    liw = 25
    jt = 2
    y_ret = y
    call DLSODA(rhs_intensity, neq, y_ret, t_ret, tout_ret, itol, &
                rtol, atol, itask, istate, iopt, rwork, lrw, iwork, &
                liw, jdum, jt)
  end function solve_intensity

! Remove globals.
    pure function solve_system(z_positions, z_samples, times, pars, initial_population, &
                             laser_intensities) result(y)
    real(dp), intent(in) :: z_positions(:)
    real(dp), intent(in) :: z_samples(:)
    real(dp), intent(in) :: pars(:)
    real(dp), intent(in) :: initial_population(:)
    real(dp), intent(in) :: laser_intensities(:, :)
    real(dp), intent(in) :: times(:)
    real(dp) :: pop(5, size(z_samples))
real(dp) :: y(size(z_positions)), normalized_intensities(size(z_positions)), &
                final_intensities(size(z_positions), size(times)), &
                    intensity_0(size(z_positions)), current_population(15), current_intensity(15), &
                t0, tout, z0, zout
    integer :: pos_idx, t_idx, sample_idx
    current_population(1:5) = initial_population
    current_population(7) = frq
    current_population(8:15) = spec_pars(:)
    current_intensity(1) = 0.0_dp
    current_intensity(2:6) = initial_population
    current_intensity(7) = frq
    current_intensity(8:15) = spec_pars(:)
    current_population(10) = pars(1)
    current_intensity(10) = pars(1)
    t0 = 0.0_dp
    tout = times(2) - times(1)
    z0 = 0.0_dp
    zout = z_samples(2)
    do concurrent(pos_idx=1:size(z_positions))
      do concurrent(sample_idx=1:size(z_samples))
        pop(:, sample_idx) = initial_population
      end do
      do t_idx = 1, size(times)
        current_intensity(1) = intensities(pos_idx, t_idx)
        do sample_idx = 1, size(z_samples)
          current_population(1:5) = pop(:, sample_idx)
          current_population(6) = current_intensity(1)
          current_population = solve_rates(current_population, t0, tout)
          pop(:, sample_idx) = current_population(1:5)
          current_intensity(2:6) = current_population(1:5)
          current_intensity = solve_intensity(current_intensity, z0, zout)
        end do
        final_intensities(pos_idx, t_idx) = current_intensity(1)
      end do
    end do
    y = sum(final_intensities, dim=2)
    intensity_0 = sum(laser_intensities, dim=2)
    normalized_intensities = y / intensity_0
    y = y / intensity_0 + (1.0_dp - maxval(normalized_intensities))
  end function solve_system

  subroutine fit_scan(data_x, data_y, expr, pars, z_samples, &
                      times, initial_population, laser_intensities, fvec)
    real(dp), intent(in) :: data_x(:)
    real(dp), intent(in) :: data_y(:)
    real(dp), intent(inout) :: pars(:)
    real(dp), intent(inout) :: fvec(:)
    real(dp), intent(in) :: z_samples(:)
    real(dp), intent(in) :: times(:)
    real(dp), intent(in) :: initial_population(:)
    real(dp), intent(in) :: laser_intensities(:, :)
    real(dp) :: tol
    integer :: iwa(size(pars)), info, m, n
    procedure(expr_f) :: expr
    real(dp) :: wa(2 * size(fvec) * size(pars) + 5 * size(pars) + size(fvec))
    tol = 1e-3_dp
    m = size(fvec)
    n = size(pars)
    call lmdif1(fcn, m, n, pars, fvec, tol, info, iwa, wa, size(wa))
  contains

    subroutine fcn(m, n, x, fvec, iflag)
      integer, intent(in) :: m, n
      integer, intent(inout) :: iflag
      real(dp), intent(in) :: x(n)
      real(dp), intent(out) :: fvec(m)
      real(dp) :: y(size(data_x))
      fvec(1) = iflag
      y = expr(data_x, z_samples, times, x, initial_population, &
               laser_intensities)
      fvec = (data_y - y)
    end subroutine fcn
  end subroutine fit_scan

end module chem

