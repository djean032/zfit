module zfit
  ! TODO: Change this whole thing to one funciton that calls fit system and takes in positions, experimental values, and necessary
  ! chemical and laser data.
  use global
  use chem
  use laser
  use file_ops
  implicit none

  integer :: idx, zdx, tdx, loc(1), size_exp, i

  call cpu_time(start)

  ! Get command line arguments
  estimated_param = spec_pars(3)
  initial_params = set_par_grid(estimated_param)

  exp = read_exp(filename)

  z_pos_slices = size(exp, dim=1)
  allocate (tot(z_pos_slices))
  allocate (int0(z_pos_slices))
  allocate (z(z_slices) [*])
  allocate (fvec(z_pos_slices) [*])
  allocate (t(t_slices) [*])
  allocate (z_sample(z_slices) [*])
  allocate (intensities(z_pos_slices, t_slices) [*])
  allocate (z_pos(z_pos_slices) [*])
  allocate (transmission(z_pos_slices) [*])
  sync all

  ! Process experimental data into positions and transmissions

  if (this_image() == 1) then
    z_pos = exp(:, 1) * 1.0e-3_dp
    transmission = exp(:, 3) / exp(:, 2)
    transmission = transmission + (1.0_dp - maxval(transmission))
    loc = minloc(transmission)
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
  end if
  sync all

  call co_broadcast(intensities, source_image=1)
  call co_broadcast(z_pos, source_image=1)
  call co_broadcast(z_sample, source_image=1)
  call co_broadcast(transmission, source_image=1)
  call co_broadcast(z, source_image=1)
  call co_broadcast(t, source_image=1)
  call co_broadcast(fvec, source_image=1)

!    initial_pop = [2.90869e17_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp]
!    spec_pars = [5.48e-18_dp, 16.0e-18_dp, 0.0_dp, 1.0e-12_dp, &
!                 1.0e-12_dp, 1.0e-12_dp, 1.05e-7_dp,1.0e-12_dp]
!    initial_params = [3.2e-17_dp, 3.0e-17_dp, 2.6e-17_dp, 3.4e-17_dp, 2.8e-17_dp]

  sync all
  idx = this_image()
  params(1) = initial_params(idx)
  call fit_scan(z_pos, transmission, solve_system, params, z, t, &
                initial_pop, intensities, fvec)
  final_params = params(1)
  rnorm = enorm(z_pos_slices, fvec)
  sync all
  call co_min(final_params, result_image=1)
  call co_min(rnorm, result_image=1)

  if (this_image() == 1) then
    print *, "Fit Complete"
    print *, "Best parameters: ", final_params
    print *, "Best residuals: ", rnorm
  end if
  sync all

  call cpu_time(finish)
  call co_max(finish, result_image=1)
  call co_min(start, result_image=1)
  if (this_image() == 1) then
    print *, "Time taken: ", finish - start
    print *, final_params
    print *, rnorm
    tot = solve_system(z_pos, z, t, final_params, initial_pop, intensities)
    int0 = sum(intensities, dim=2)
    open (1, file="intensities.dat", status="unknown")
    open (2, file="exp.dat", status="unknown")
    do i = 1, z_pos_slices
      write (1, *) z_pos(i), tot(i)
      write (2, *) z_pos(i), transmission(i)
    end do
  end if
contains
  function set_par_grid(par) result(par_grid)
    real(dp), intent(in) :: par
    real(dp) :: par_grid(5)
    par_grid = [par - 0.1_dp * par, par - 0.05_dp * par, par, &
                par + 0.05_dp * par, par + 0.1_dp * par]
  end function set_par_grid
end module zfit
