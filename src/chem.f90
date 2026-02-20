module chem
  use global
  implicit none

  private

  abstract interface
    pure function expr_f(z_positions, z_samples, times, initial_population, &
                         laser_intensities, frq, spec_pars) result(y)
      import :: c_double
      implicit none
      real(c_double), intent(in) :: z_positions(:)
      real(c_double), intent(in) :: z_samples(:)
      real(c_double), intent(in) :: times(:)
      real(c_double), intent(in) :: initial_population(:)
      real(c_double), intent(in) :: laser_intensities(:, :)
      real(c_double), intent(in) :: frq
      real(c_double), intent(in) :: spec_pars(8)
      real(c_double) :: y(size(z_positions))
    end function expr_f
  end interface

  abstract interface
    pure subroutine odepack_f(neq, time, y, ydot)
      import :: c_int, c_double
      implicit none
      integer(c_int), intent(in) :: neq
      real(c_double), intent(in) :: time
      real(c_double), intent(in) :: y(15)
      real(c_double), intent(out) :: ydot(neq)
    end subroutine

    pure subroutine odepack_jac()
    end subroutine
  end interface
  interface
    pure subroutine DLSODA(f, neq, y, t_in, t_out, itol, rtol, atol, itask, &
                           istate, iopt, rwork, lrw, iwork, liw, jac, jt)
      import :: c_int, c_double
      import :: odepack_f, odepack_jac
      implicit none
      integer(c_int), intent(in) :: neq, itol, itask, iopt, lrw, liw, jt
      integer(c_int), intent(inout) :: istate
      real(c_double), intent(in) :: rtol, atol
      real(c_double), intent(inout) :: t_in
      real(c_double), intent(in) :: t_out
      real(c_double), intent(inout) :: y(neq)
      real(c_double), intent(inout) :: rwork(lrw)
      integer(c_int), intent(inout) :: iwork(liw)
      procedure(odepack_f) :: f
      procedure(odepack_jac) :: jac
    end subroutine
  end interface

  public :: rhs_rates, rhs_intensity, jdum, solve_rates, solve_intensity, &
            solve_system, fit_scan

  ! Module-level variables for callback (to avoid nested procedure issue on Linux gfortran)
  real(c_double), pointer :: cb_data_x(:) => null()
  real(c_double), pointer :: cb_data_y(:) => null()
  real(c_double), pointer :: cb_spec_pars(:) => null()
  real(c_double), pointer :: cb_z_samples(:) => null()
  real(c_double), pointer :: cb_times(:) => null()
  real(c_double), pointer :: cb_initial_population(:) => null()
  real(c_double), pointer :: cb_laser_intensities(:,:,:) => null()
  real(c_double) :: cb_frq
  integer(c_int) :: cb_num_x_pts, cb_num_datasets
  procedure(expr_f), pointer :: cb_expr => null()

contains
  pure subroutine rhs_rates(neq, time, y, ydot)
    integer(c_int), intent(in) :: neq
    real(c_double), intent(in) :: time
    real(c_double), dimension(15), intent(in) :: y
    real(c_double), dimension(neq), intent(out) :: ydot
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
    integer(c_int), intent(in) :: neq
    real(c_double), intent(in) :: time
    real(c_double), dimension(15), intent(in) :: y
    real(c_double), dimension(neq), intent(out) :: ydot
    ydot(1) = -y(8) * y(2) * y(1) &
              - y(9) * y(3) * y(1) &
              - y(10) * y(5) * y(1)
  end subroutine rhs_intensity

  pure subroutine jdum()
  end subroutine jdum

  pure function solve_rates(y, t_in, t_out) result(y_ret)
    integer(c_int) :: neq, itol, itask, iopt, lrw, &
               liw, jt, istate
    real(c_double), intent(in) :: t_in, t_out
    real(c_double) :: t_ret, tout_ret, rtol, atol, rwork(102)
    real(c_double), intent(in) :: y(15)
    real(c_double) :: y_ret(15)
    integer(c_int) :: iwork(25)
    t_ret = t_in
    tout_ret = t_out
    neq = 5
    lrw = 102
    liw = 25
    rtol = 1e-6_c_double
    atol = 1e-6_c_double
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
    integer(c_int) :: neq, itol, itask, iopt, lrw, &
               liw, jt, istate
    real(c_double), intent(in) :: t_in, t_out
    real(c_double), intent(in) :: y(15)
    real(c_double) :: t_ret, tout_ret, rtol, atol, rwork(102)
    real(c_double) :: y_ret(15)
    integer(c_int) :: iwork(25)
    t_ret = t_in
    tout_ret = t_out
    neq = 1
    lrw = 102
    liw = 25
    rtol = 1e-6_c_double
    atol = 1e-6_c_double
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
  pure function solve_system(z_positions, z_samples, times, initial_population, &
                             laser_intensities, frq, spec_pars) result(y)
    !GCC$ attributes dllexport :: solve_system
    real(c_double), intent(in) :: z_positions(:)
    real(c_double), intent(in) :: z_samples(:)
    real(c_double), intent(in) :: initial_population(:)
    real(c_double), intent(in) :: laser_intensities(:, :)
    real(c_double), intent(in) :: times(:)
    real(c_double), intent(in) :: frq
    real(c_double), intent(in) :: spec_pars(8)
    real(c_double) :: pop(5, size(z_samples))
    real(c_double) :: y(size(z_positions)), normalized_intensities(size(z_positions)), &
                    final_intensities(size(z_positions), size(times)), &
                    intensity_0(size(z_positions)), current_population(15), current_intensity(15), &
                    t0, tout, z0, zout
    integer(c_int) :: pos_idx, t_idx, sample_idx
    
    current_population(1:5) = initial_population
    current_population(7) = frq
    current_population(8:15) = spec_pars(:)
    current_intensity(1) = 0.0_c_double
    current_intensity(2:6) = initial_population
    current_intensity(7) = frq
    current_intensity(8:15) = spec_pars(:)
    current_population(10) = spec_pars(3)
    current_intensity(10) = spec_pars(3)
    t0 = 0.0_c_double
    tout = times(2) - times(1)
    z0 = 0.0_c_double
    zout = z_samples(2)
    do pos_idx = 1, size(z_positions)
      do sample_idx = 1, size(z_samples)
        pop(:, sample_idx) = initial_population
      end do
      do t_idx = 1, size(times)
        current_intensity(1) = laser_intensities(pos_idx, t_idx)
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
    y = y / intensity_0 + (1.0_c_double - maxval(normalized_intensities))
  end function solve_system

  subroutine fit_scan(data_x, data_y, expr, z_samples, &
           times, initial_population, laser_intensities, fvec, frq, spec_pars, num_x_pts, num_datasets)
    real(c_double), intent(in), target :: data_x(:)
    real(c_double), intent(in), target :: data_y(:)
    real(c_double), intent(inout), target :: spec_pars(8)
    real(c_double), intent(inout) :: fvec(:)
    real(c_double), intent(in), target :: z_samples(:)
    real(c_double), intent(in), target :: times(:)
    real(c_double), intent(in), target :: initial_population(:)
    real(c_double), intent(in), target :: laser_intensities(:, :, :)
    real(c_double), intent(in) :: frq
    integer(c_int), intent(in) :: num_x_pts, num_datasets
    procedure(expr_f) :: expr
    real(c_double) :: tol
    integer(c_int) :: iwa(5), info, m, n
    real(c_double) :: wa(2 * size(fvec) * 5 + 5 * 5 + size(fvec))

    ! Set module-level callback variables
    cb_data_x => data_x
    cb_data_y => data_y
    cb_spec_pars => spec_pars
    cb_z_samples => z_samples
    cb_times => times
    cb_initial_population => initial_population
    cb_laser_intensities => laser_intensities
    cb_frq = frq
    cb_num_x_pts = num_x_pts
    cb_num_datasets = num_datasets
    cb_expr => expr

    tol = 1e-3_c_double
    m = size(fvec)
    n = 1
    call lmdif1(fcn, m, n, spec_pars(3), fvec, tol, info, iwa, wa, size(wa))

    ! Clear pointers
    cb_data_x => null()
    cb_data_y => null()
    cb_spec_pars => null()
    cb_z_samples => null()
    cb_times => null()
    cb_initial_population => null()
    cb_laser_intensities => null()
    cb_expr => null()
  end subroutine fit_scan

  ! Module-level callback for lmdif1
  subroutine fcn(m, n, x, fvec, iflag)
    integer(c_int), intent(in) :: m, n
    integer(c_int), intent(inout) :: iflag
    real(c_double), intent(in) :: x(n)
    real(c_double), intent(out) :: fvec(m)
    real(c_double) :: y(size(cb_data_x)), local_spec_pars(8)
    integer(c_int) :: i, bidx, eidx

    local_spec_pars = cb_spec_pars
    local_spec_pars(3) = x(1)
    do i = 1, cb_num_datasets
      bidx = (i - 1) * cb_num_x_pts + 1
      eidx = i * cb_num_x_pts
      y(bidx:eidx) = cb_expr(cb_data_x(bidx:eidx), cb_z_samples, cb_times, cb_initial_population, cb_laser_intensities(:, :, i), cb_frq, local_spec_pars)
      fvec(bidx:eidx) = (cb_data_y(bidx:eidx) - y(bidx:eidx))
    end do
  end subroutine

end module chem
