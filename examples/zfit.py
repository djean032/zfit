import ctypes
import os
import sys
import numpy as np

dirname = os.getcwd()
lib_name = dirname + "\\libzfit.dll"
lib = ctypes.CDLL(lib_name, winmode=0)

lib.fit_zscan_data.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double * 8),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
lib.fit_zscan_data.restype = None


def fit_zscan_data_wrapper(
    x_data,
    y_data,
    populations,
    t_slices,
    z_slices,
    tau,
    wavelength,
    w0,
    M2,
    pulse_energy,
    spec_pars,
):
    x_data = np.ascontiguousarray(x_data, dtype=np.float64)
    y_data = np.ascontiguousarray(y_data, dtype=np.float64)
    populations = np.ascontiguousarray(populations, dtype=np.float64)
    spec_pars = np.ascontiguousarray(spec_pars, dtype=np.float64)

    residuals = np.zeros_like(y_data)
    error = ctypes.c_double()

    n_xdata = x_data.shape[0]
    n_populations = populations.shape[0]
    n_residuals = residuals.shape[0]

    lib.fit_zscan_data(
        x_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        populations.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        residuals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(error),
        ctypes.byref(ctypes.c_int(t_slices)),
        ctypes.byref(ctypes.c_int(z_slices)),
        ctypes.byref(ctypes.c_double(tau)),
        ctypes.byref(ctypes.c_double(wavelength)),
        ctypes.byref(ctypes.c_double(w0)),
        ctypes.byref(ctypes.c_double(M2)),
        ctypes.byref(ctypes.c_double(pulse_energy)),
        spec_pars.ctypes.data_as(ctypes.POINTER(ctypes.c_double * 8)),
        ctypes.byref(ctypes.c_int(n_xdata)),
        ctypes.byref(ctypes.c_int(n_populations)),
        ctypes.byref(ctypes.c_int(n_residuals)),
    )

    return residuals, error.value, spec_pars


if __name__ == "__main__":
    x_data = np.linspace(0, 1, 100)
    y_data = np.sin(x_data * np.pi)
    populations = np.array([1.0, 0.5, 0.3])
    spec_pars = np.zeros(8, dtype=np.float64)

    residuals, error, spec_pars = fit_zscan_data_wrapper(
        x_data, y_data, populations,
        t_slices=50, z_slices=50,
        tau=1e-9, wavelength=532e-9, w0=1e-6, M2=1.5, pulse_energy=1e-6,
        spec_pars=spec_pars
    )

    print(f"Error: {error}")
    print(f"Spec pars: {spec_pars}")
