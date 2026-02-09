module zfit
  use global
  use chem
  use laser
  implicit none

contains
  subroutine fit_zscan_data_fortran(x_data, y_data, populations, residuals, error, t_slices, z_slices, tau, wavelength, w0, M2, &
                                    pulse_energy, spec_pars, frq)
    real(dp), intent(in) :: x_data(:), y_data(:), populations(:), tau, wavelength, w0, M2
    real(dp), intent(in) :: pulse_energy
    real(dp), intent(inout) :: spec_pars(8)
    real(dp), intent(in) :: frq
    integer(kind=4), intent(in) :: t_slices, z_slices
    real(dp), intent(out) :: residuals(:), error
    integer(kind=4) :: z_pos_slices, idx, zdx, tdx, loc(1), size_exp, i
    real(dp) :: wid, zr
    real(dp), allocatable :: tot(:), int0(:), z(:), fvec(:), t(:), z_sample(:), intensities(:, :), z_pos(:)
    real(dp) :: start, finish

    call cpu_time(start)

    wid = tau / (2 * (log(2.0_dp))**0.5)
    zr = (pi * w0**2) / (M2 * wavelength)

    z_pos_slices = size(x_data, dim=1)

    allocate (tot(z_pos_slices))
    allocate (int0(z_pos_slices))
    allocate (z(z_slices))
    allocate (fvec(z_pos_slices))
    allocate (t(t_slices))
    allocate (z_sample(z_slices))
    allocate (intensities(z_pos_slices, t_slices))
    allocate (z_pos(z_pos_slices))

    loc = minloc(y_data)
    z_pos = z_pos - z_pos(loc(1))

    z = linspace(0.0_dp, 1.0e-1_dp, z_slices)
    t = linspace(-4.0e-9_dp, 4.0e-9_dp, t_slices)

    do concurrent(zdx=1:z_pos_slices)
      do concurrent(tdx=1:t_slices)
        intensities(zdx, tdx) = irradiance(0.0_dp, t(tdx), z_pos(zdx), pulse_energy, wid, w0, zr)
      end do
    end do

    call fit_scan(x_data, y_data, solve_system, z_sample, &
                  t, populations, intensities, fvec, frq, spec_pars)
    residuals = fvec
    error = enorm(z_pos_slices, fvec)

    call cpu_time(finish)
  end subroutine

  subroutine fit_zscan_data(x_data, y_data, populations, residuals, error, t_slices, z_slices, tau, wavelength, w0, M2, &
                            pulse_energy, spec_pars, frq, x_size, y_size, pop_size, res_size) bind(C, name="fit_zscan_data")
    use iso_c_binding
    real(c_double), intent(in) :: x_data(*), y_data(*), populations(*), tau, wavelength, w0, M2
    real(c_double), intent(in) :: pulse_energy
    real(c_double), intent(inout) :: spec_pars(8)
    real(c_double), intent(in) :: frq
    integer(c_int), intent(in) :: t_slices, z_slices
    real(c_double), intent(out) :: residuals(*), error
    integer(c_int), intent(in) :: x_size, y_size, pop_size, res_size
    real(c_double), allocatable :: x_data_fort(:), y_data_fort(:), populations_fort(:), residuals_fort(:)

    allocate(x_data_fort(x_size), y_data_fort(y_size), populations_fort(pop_size), residuals_fort(res_size))

    x_data_fort = x_data(1:x_size)
    y_data_fort = y_data(1:y_size)
    populations_fort = populations(1:pop_size)

    call fit_zscan_data_fortran(x_data_fort, y_data_fort, populations_fort, residuals_fort, error, t_slices, z_slices, &
                        tau, wavelength, w0, M2, pulse_energy, spec_pars, frq)

    residuals(1:res_size) = residuals_fort

    deallocate (x_data_fort, y_data_fort, populations_fort, residuals_fort)
  end subroutine
end module zfit
