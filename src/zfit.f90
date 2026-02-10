module zfit
  use global
  use chem
  use laser
  implicit none

contains
  subroutine fit_zscan_data(x_data, y_data, populations, residuals, error, t_slices, z_slices, tau, wavelength, w0, M2, &
                                    pulse_energy, spec_pars, n_xdata, n_populations, n_residuals) bind(C, name="fit_zscan_data")
    !GCC$ attributes dllexport :: fit_zscan_data
    real(c_double), intent(in) :: x_data(*), y_data(*), populations(*), tau, wavelength, w0, M2
    real(c_double), intent(in) :: pulse_energy
    real(c_double), intent(inout) :: spec_pars(8)
    integer(c_int), intent(in) :: t_slices, z_slices
    real(c_double), intent(out) :: residuals(*), error
    integer(c_int), intent(in) :: n_xdata, n_populations, n_residuals
    integer(c_int) :: z_pos_slices, idx, zdx, tdx, loc(1), size_exp, i
    real(c_double) :: wid, zr, frq
    real(c_double), allocatable :: tot(:), int0(:), z(:), fvec(:), t(:), intensities(:, :), z_pos(:)
    real(c_double), allocatable :: data_x(:), data_y(:), data_pop(:)
    real(c_double) :: start, finish

    z_pos_slices = n_xdata

    allocate(tot(z_pos_slices), int0(z_pos_slices), z(z_slices), fvec(z_pos_slices))
    allocate(t(z_slices), intensities(z_pos_slices, t_slices), z_pos(z_pos_slices))
    allocate(data_x(z_pos_slices), data_y(z_pos_slices), data_pop(n_populations))

    do zdx = 1, z_pos_slices
      data_x(zdx) = x_data(zdx)
      data_y(zdx) = y_data(zdx)
      z_pos(zdx) = x_data(zdx)
    end do

    do i = 1, n_populations
      data_pop(i) = populations(i)
    end do

    wid = tau / (2 * (log(2.0_c_double))**0.5)
    zr = (pi * w0**2) / (M2 * wavelength)
    frq = c / wavelength

    loc = minloc(data_y)
    do zdx = 1, z_pos_slices
      z_pos(zdx) = z_pos(zdx) - z_pos(loc(1))
    end do

    do zdx = 1, z_slices
      z(zdx) = 0.0_c_double + (1.0e-1_c_double - 0.0_c_double) * (zdx - 1) / (z_slices - 1)
    end do

    do tdx = 1, t_slices
      t(tdx) = -4.0e-9_c_double + (4.0e-9_c_double - (-4.0e-9_c_double)) * (tdx - 1) / (t_slices - 1)
    end do

    do zdx = 1, z_pos_slices
      do tdx = 1, t_slices
        intensities(zdx, tdx) = irradiance(0.0_c_double, t(tdx), z_pos(zdx), pulse_energy, wid, w0, zr)
      end do
    end do

    call fit_scan(data_x, data_y, solve_system, z, &
                  t, data_pop, intensities, fvec, frq, spec_pars)
    do i = 1, z_pos_slices
      residuals(i) = fvec(i)
    end do
    error = enorm(z_pos_slices, fvec)

    call cpu_time(finish)
  end subroutine
end module zfit
