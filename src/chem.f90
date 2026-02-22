module chem
   use global
   implicit none

   private

   abstract interface
      function expr_f(z_positions, z_samples, times, initial_population, &
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
      subroutine odepack_f(neq, time, y, ydot)
         import :: c_int, c_double
         implicit none
         integer(c_int), intent(in) :: neq
         real(c_double), intent(in) :: time, y(*)
         real(c_double), intent(out) :: ydot(neq)
      end subroutine

      subroutine odepack_jac()
      end subroutine
   end interface

   interface
      subroutine DLSODA(f, neq, y, t_in, t_out, itol, rtol, atol, itask, &
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

   !$omp threadprivate(cb_data_x, cb_data_y, cb_spec_pars, cb_z_samples, &
   !$omp                cb_times, cb_initial_population, cb_laser_intensities, &
   !$omp                cb_frq, cb_num_x_pts, cb_num_datasets, cb_expr, cb_y_work)
   real(c_double), pointer :: cb_data_x(:) => null()
   real(c_double), pointer :: cb_data_y(:) => null()
   real(c_double), pointer :: cb_spec_pars(:) => null()
   real(c_double), pointer :: cb_z_samples(:) => null()
   real(c_double), pointer :: cb_times(:) => null()
   real(c_double), pointer :: cb_initial_population(:) => null()
   real(c_double), pointer :: cb_laser_intensities(:, :, :) => null()
   real(c_double) :: cb_frq
   integer(c_int) :: cb_num_x_pts, cb_num_datasets
   procedure(expr_f), pointer :: cb_expr => null()
   real(c_double), allocatable :: cb_y_work(:)

   !$omp threadprivate(pre_inv_hfrq, pre_k1, pre_k2, pre_k3, pre_inv_tau4, &
   !$omp               pre_inv_tau5, pre_inv_tau6, pre_inv_tau7, pre_inv_tau8)
   real(c_double) :: pre_inv_hfrq, pre_k1, pre_k2, pre_k3, pre_inv_tau4, &
                     pre_inv_tau5, pre_inv_tau6, pre_inv_tau7, pre_inv_tau8

contains

   subroutine rhs_rates(neq, time, y, ydot)
      integer(c_int), intent(in) :: neq
      real(c_double), intent(in) :: time, y(*)
      real(c_double), intent(out) :: ydot(neq)
      real(c_double) :: i1, i6, r1, r2, r4

      i1 = y(1)
      i6 = y(6)*pre_inv_hfrq
      r1 = pre_k1*i1*i6
      r2 = pre_k2*y(2)*i6
      r4 = pre_k3*y(4)*i6
      ydot(1) = -r1 + y(2)*pre_inv_tau4 + y(4)*pre_inv_tau7
      ydot(2) = r1 - y(2)*pre_inv_tau4 - r2 + y(3)*pre_inv_tau6 - y(2)*pre_inv_tau5
      ydot(3) = r2 - y(3)*pre_inv_tau6
      ydot(4) = -r4 + y(5)*pre_inv_tau8 + y(2)*pre_inv_tau5 - y(4)*pre_inv_tau7
      ydot(5) = r4 - y(5)*pre_inv_tau8
   end subroutine rhs_rates

   subroutine rhs_intensity(neq, time, y, ydot)
      integer(c_int), intent(in) :: neq
      real(c_double), intent(in) :: time, y(*)
      real(c_double), intent(out) :: ydot(neq)

      ydot(1) = -(pre_k1*y(2) + pre_k2*y(3) + pre_k3*y(5))*y(1)
   end subroutine rhs_intensity

   subroutine jdum()
   end subroutine jdum

   subroutine solve_rates(y, t_in, t_out, rwork, iwork, y_ret)
      real(c_double), intent(in) :: y(6), t_in, t_out
      real(c_double), intent(inout) :: rwork(102)
      integer(c_int), intent(inout) :: iwork(25)
      real(c_double), intent(out) :: y_ret(6)
      integer(c_int) :: istate, neq
      real(c_double) :: t_ret

      neq = 5
      t_ret = t_in
      istate = 1
      y_ret = y
      call DLSODA(rhs_rates, neq, y_ret, t_ret, t_out, 1, &
                  1e-6_c_double, 1e-6_c_double, 1, istate, 0, rwork, 102, iwork, &
                  25, jdum, 2)
   end subroutine solve_rates

   subroutine solve_intensity(y, t_in, t_out, rwork, iwork, y_ret)
      real(c_double), intent(in) :: y(6), t_in, t_out
      real(c_double), intent(inout) :: rwork(102)
      integer(c_int), intent(inout) :: iwork(25)
      real(c_double), intent(out) :: y_ret(6)
      integer(c_int) :: istate, neq
      real(c_double) :: t_ret

      neq = 1
      t_ret = t_in
      istate = 1
      y_ret = y
      call DLSODA(rhs_intensity, neq, y_ret, t_ret, t_out, 1, &
                  1e-6_c_double, 1e-6_c_double, 1, istate, 0, rwork, 102, iwork, &
                  25, jdum, 2)
   end subroutine solve_intensity

   function solve_system(z_positions, z_samples, times, initial_population, &
                         laser_intensities, frq, spec_pars) result(y)
      !GCC$ attributes dllexport :: solve_system
      real(c_double), intent(in) :: frq, initial_population(:), laser_intensities(:, :), spec_pars(8), times(:), &
                                    z_positions(:), z_samples(:)
      integer(c_int) :: pos_idx, sample_idx, t_idx
      real(c_double) :: current_intensity(6), current_population(6), final_intensities(size(z_positions), size(times)), &
                        intensity_0(size(z_positions)), normalized_intensities(size(z_positions)), pop(5, size(z_samples)), &
                        tout, y(size(z_positions)), zout
      real(c_double) :: rwork(102)
      integer(c_int) :: iwork(25)

      pre_inv_hfrq = 1.0_c_double/(h*frq)
      pre_k1 = spec_pars(1)
      pre_k2 = spec_pars(2)
      pre_k3 = spec_pars(3)
      pre_inv_tau4 = 1.0_c_double/spec_pars(4)
      pre_inv_tau5 = 1.0_c_double/spec_pars(5)
      pre_inv_tau6 = 1.0_c_double/spec_pars(6)
      pre_inv_tau7 = 1.0_c_double/spec_pars(7)
      pre_inv_tau8 = 1.0_c_double/spec_pars(8)

      current_intensity(1) = 0.0_c_double
      current_intensity(2:6) = initial_population
      tout = times(2) - times(1)
      zout = z_samples(2)
      do pos_idx = 1, size(z_positions)
         pop = spread(initial_population, 2, size(z_samples))
         do t_idx = 1, size(times)
            current_intensity(1) = laser_intensities(pos_idx, t_idx)
            do sample_idx = 1, size(z_samples)
               current_population(1:5) = pop(:, sample_idx)
               current_population(6) = current_intensity(1)
               call solve_rates(current_population, 0.0_c_double, tout, rwork, iwork, current_population)
               pop(:, sample_idx) = current_population(1:5)
               current_intensity(2:6) = current_population(1:5)
               call solve_intensity(current_intensity, 0.0_c_double, zout, rwork, iwork, current_intensity)
            end do
            final_intensities(pos_idx, t_idx) = current_intensity(1)
         end do
      end do
      y = sum(final_intensities, dim=2)
      intensity_0 = sum(laser_intensities, dim=2)
      normalized_intensities = y/intensity_0
      y = y/intensity_0 + (1.0_c_double - maxval(normalized_intensities))
   end function solve_system

   subroutine fit_scan(data_x, data_y, expr, z_samples, &
                       times, initial_population, laser_intensities, fvec, frq, spec_pars, num_x_pts, num_datasets)
      integer(c_int), intent(in) :: num_datasets, num_x_pts
      real(c_double), intent(in), target :: data_x(:), data_y(:), initial_population(:), laser_intensities(:, :, :), &
                                            times(:), z_samples(:)
      real(c_double), intent(inout), target :: spec_pars(8)
      real(c_double), intent(inout) :: fvec(:)
      real(c_double), intent(in) :: frq
      procedure(expr_f) :: expr
      integer(c_int) :: info, iwa(5)
      real(c_double) :: wa(2*size(fvec)*5 + 5*5 + size(fvec))

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
      allocate (cb_y_work(size(data_x)))

      call lmdif1(fcn, size(fvec), 1, spec_pars(3), fvec, 1e-3_c_double, info, iwa, wa, size(wa))

      cb_data_x => null()
      cb_data_y => null()
      cb_spec_pars => null()
      cb_z_samples => null()
      cb_times => null()
      cb_initial_population => null()
      cb_laser_intensities => null()
      cb_expr => null()
      deallocate (cb_y_work)
   end subroutine fit_scan

   subroutine fcn(m, n, x, fvec, iflag)
      integer(c_int), intent(in) :: m, n
      integer(c_int), intent(inout) :: iflag
      real(c_double), intent(in) :: x(n)
      real(c_double), intent(out) :: fvec(m)
      integer(c_int) :: bidx, eidx, i
      real(c_double) :: local_spec_pars(8)

      local_spec_pars = cb_spec_pars
      local_spec_pars(3) = x(1)
      do i = 1, cb_num_datasets
         bidx = (i - 1)*cb_num_x_pts + 1
         eidx = i*cb_num_x_pts
         cb_y_work(bidx:eidx) = cb_expr(cb_data_x(bidx:eidx), cb_z_samples, cb_times, cb_initial_population, &
                                        cb_laser_intensities(:, :, i), cb_frq, local_spec_pars)
         fvec(bidx:eidx) = (cb_data_y(bidx:eidx) - cb_y_work(bidx:eidx))
      end do
   end subroutine

end module chem
