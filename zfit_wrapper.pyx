# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from cython.parallel import prange
import numpy as np
cimport numpy as np

np.import_array()

cdef extern void fit_zscan_data(
    double* x_data, double* y_data, double* populations,
    double* residuals, double* error,
    int* t_slices, int* z_slices,
    double* sample_width, double* tau,
    double* wavelength, double* w0, double* M2,
    double* pulse_energy, double* spec_pars,
    int* num_x_pts, int* num_datasets,
    int* n_populations, int* n_residuals
) noexcept nogil

def fit_zscan(double[::1] x_data not None, double[::1] y_data not None,
              double[::1] populations not None, int t_slices, int z_slices,
              double sample_width, double tau, double wavelength, double w0, double M2,
              double[::1] pulse_energy not None, double[::1] spec_pars not None,
              int num_x_pts, int num_datasets, bint is_sa=False, int n_starts=3):
    """
    Fit z-scan data with multi-start parallel optimization.
    
    Parameters:
        is_sa: Whether sample is saturable absorber (SA) vs reverse saturable absorber (RSA).
               SA centers around max of y_data, RSA centers around min of y_data.
               Default False (RSA).
        n_starts: Number of starting points (default=3)
                  Starting values: spec_pars[2] * [0.7, 1.0, 1.3, ...]
    
    Returns:
        residuals, error, spec_pars (best result)
    """
    cdef int i, best_idx, n_pop, n_res
    cdef double best_error
    
    n_pop = populations.shape[0]
    n_res = y_data.shape[0]
    
    # Center x_data around the SA/RSA peak position
    # For SA: center around the maximum (minimum transmission)
    # For RSA: center around the minimum (maximum transmission)
    cdef double center_x
    if is_sa:
        center_x = x_data[np.argmax(y_data)]
    else:
        center_x = x_data[np.argmin(y_data)]
    cdef double[::1] centered_x = np.asarray(x_data) - center_x
    
    # Pre-allocate per-start results
    cdef double[:, ::1] all_spec_pars = np.tile(np.asarray(spec_pars), (n_starts, 1))
    cdef double[:, ::1] all_residuals = np.zeros((n_starts, n_res), dtype=np.float64)
    cdef double[::1] all_errors = np.zeros(n_starts, dtype=np.float64)
    
    # Generate starting points: two-sided logarithmic scale
    # Spans 0.3x to 3x of initial guess (closer to initial guess)
    # For n_starts=3: [0.3, 1.0, 3.0]
    # For n_starts=5: [0.3, 0.55, 1.0, 1.83, 3.0]
    cdef double[::1] start_multipliers = np.array(
        [10 ** (-0.52 + 1.04 * i / max(1, n_starts - 1)) for i in range(n_starts)],
        dtype=np.float64
    )
    for i in range(n_starts):
        all_spec_pars[i, 2] = spec_pars[2] * start_multipliers[i]
    
    # Parallel optimization - OpenMP manages thread count
    with nogil:
        for i in prange(n_starts):
            fit_zscan_data(
                &centered_x[0], &y_data[0], &populations[0],
                &all_residuals[i, 0], &all_errors[i],
                &t_slices, &z_slices,
                &sample_width, &tau,
                &wavelength, &w0, &M2,
                &pulse_energy[0], &all_spec_pars[i, 0],
                &num_x_pts, &num_datasets,
                &n_pop, &n_res
            )
    
    # Find best result
    best_idx = 0
    best_error = all_errors[0]
    for i in range(1, n_starts):
        if all_errors[i] < best_error:
            best_error = all_errors[i]
            best_idx = i
    
    return (
        np.asarray(all_residuals[best_idx]),
        all_errors[best_idx],
        np.asarray(all_spec_pars[best_idx])
    )
