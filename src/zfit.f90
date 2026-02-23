module zfit
   use global
   use chem
   use laser
   implicit none

contains
   subroutine fit_zscan_data(x_data, y_data, populations, residuals, error, t_slices, z_slices, &
                             sample_width, tau, wavelength, w0, M2, pulse_energy, spec_pars, &
                             num_x_pts, num_datasets, n_populations, n_residuals) bind(C, name="fit_zscan_data")
      integer(c_int), intent(in) :: n_populations, n_residuals, num_datasets, num_x_pts, t_slices, z_slices
      real(c_double), intent(in) :: M2, sample_width, tau, w0, wavelength, populations(*), &
                                    pulse_energy(*), x_data(*), y_data(*)
      real(c_double), intent(inout) :: spec_pars(8)
      real(c_double), intent(out) :: error, residuals(*)
      integer(c_int) :: bidx, eidx, idx, loc(1), tdx, zdx
      real(c_double) :: frq, wid, zr
      real(c_double), allocatable :: data_x(:), data_pop(:), data_y(:), fvec(:), &
                                     intensities(:, :, :), t(:), z(:)

      allocate (z(z_slices), fvec(num_x_pts*num_datasets))
      allocate (t(z_slices), intensities(num_x_pts, t_slices, num_datasets))
      allocate (data_x(num_x_pts*num_datasets), data_y(num_x_pts*num_datasets), data_pop(n_populations))

      data_x = x_data(1:num_x_pts*num_datasets)/1000
      data_y = y_data(1:num_x_pts*num_datasets)
      data_pop = populations(1:n_populations)

      !wid = tau / (2 * (log(2.0_c_double))**0.5)
      wid = tau
      zr = (pi*w0**2)/(M2*wavelength)
      frq = c/wavelength

      do idx = 1, num_datasets
         bidx = (idx - 1)*num_x_pts + 1
         eidx = idx*num_x_pts
         loc = minloc(data_y(bidx:eidx))
         data_x(bidx:eidx) = data_x(bidx:eidx) - data_x(bidx + loc(1) - 1)
      end do

      z = linspace(0.0_c_double, sample_width, z_slices)
      t = linspace(-tau/2.0_c_double, tau/2.0_c_double, t_slices)

      do idx = 1, num_datasets
         bidx = (idx - 1)*num_x_pts + 1
         do tdx = 1, t_slices
            do zdx = 1, num_x_pts
               intensities(zdx, tdx, idx) = irradiance(0.0_c_double, t(tdx), data_x(bidx + zdx - 1), &
                                                       pulse_energy(idx), wid, w0, zr)
            end do
         end do
      end do

      call fit_scan(data_x, data_y, solve_system, z, &
                    t, data_pop, intensities, fvec, frq, spec_pars, num_x_pts, num_datasets)
      error = enorm(num_x_pts*num_datasets, fvec)
      residuals(1:n_residuals) = fvec(1:n_residuals)

   end subroutine
end module zfit
