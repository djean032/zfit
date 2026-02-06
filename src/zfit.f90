module zfit
  ! TODO: Change this whole thing to one funciton that calls fit system and takes in positions, experimental values, and necessary
  ! chemical and laser data.
  use global
  use chem
  use laser
  use file_ops
  implicit none

contains
  subroutine fit_zscan_data(x_data, y_data, params, populations, residuals, error, t_slices, z_slices, tau, wavelength, w0, M2,
  pulse_energy)
    real(dp), intent(in) :: x_data(:), y_data(:), populations(:), tau, wavelength, w0, M2
  integer, intent(in) :: t_slices, z_slices
  real(dp), intent(out) :: params(:), residuals(:), error
  integer :: z_pos_slices, idx, zdx, tdx, loc(1), size_exp, i
    real(dp), allocatable :: tot(:), int0(:), z(:), fvec(:), t(:), z_sample(:), intensities(:, :), z_pos(:)

  call cpu_time(start)

  ! setup important variables
  wid = tau / (2 * (log(2.0_dp))**0.5)
  zr = (PI_dp * w0**2) / (M2 * wavelength)
  frq = c / wavelength

  estimated_param = params(3)
  initial_params = set_par_grid(estimated_param)

  z_pos_slices = size(x_data, dim=1)

  allocate tot(z_pos_slices)
  allocate int0(z_pos_slices)
  allocate z(z_slices)
  allocate fvec(z_pos_slices)
  allocate t(t_slices)
  allocate z_sample(z_slices)
  allocate intensities(z_pos_slices, t_slices)
  allocate z_pos(z_pos_slices)

  ! center the data if needed
  loc = minloc(y_data)
  z_pos = z_pos - z_pos(loc(1))

  ! Initialize t and z values for DLSODA
  z = linspace(0.0_dp, 1.0e-1_dp, z_slices)
  t = linspace(-4.0e-9_dp, 4.0e-9_dp, t_slices)

  do concurrent(zdx=1:z_pos_slices)
    z_position = z_pos(zdx)
    do concurrent(tdx=1:t_slices)
      intensities(zdx, tdx) = irradiance(0.0_dp, t(tdx), z_position)
    end do
  end do

    call fit_scan(z_pos, transmission, params, z, t, initial_pop, intensities, fvec)
  error = enorm(z_pos_slices, fvec)

!    initial_pop = [2.90869e17_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp]
!    spec_pars = [5.48e-18_dp, 16.0e-18_dp, 0.0_dp, 1.0e-12_dp, &
!                 1.0e-12_dp, 1.0e-12_dp, 1.05e-7_dp,1.0e-12_dp]
!    initial_params = [3.2e-17_dp, 3.0e-17_dp, 2.6e-17_dp, 3.4e-17_dp, 2.8e-17_dp]

  call cpu_time(finish)
  end subroutine

  function set_par_grid(par) result(par_grid)
    real(dp), intent(in) :: par
    real(dp) :: par_grid(5)
    par_grid = [par - 0.1_dp * par, par - 0.05_dp * par, par, &
                par + 0.05_dp * par, par + 0.1_dp * par]
  end function set_par_grid
end module zfit
