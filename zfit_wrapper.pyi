import numpy as np
from numpy.typing import NDArray

def fit_zscan(
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    populations: NDArray[np.float64],
    t_slices: int,
    z_slices: int,
    sample_width: float,
    tau: float,
    wavelength: float,
    w0: float,
    M2: float,
    pulse_energy: NDArray[np.float64],
    spec_pars: NDArray[np.float64],
    num_x_pts: int,
    num_datasets: int,
    is_sa: bool = False,
    n_starts: int = 3,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """
    Fit Z-scan data to extract nonlinear optical parameters.

    Parameters
    ----------
    x_data : NDArray[np.float64]
        Z-positions (micrometers), shape (num_x_pts * num_datasets,).
    y_data : NDArray[np.float64]
        Normalized transmittance values, shape (num_x_pts * num_datasets,).
    populations : NDArray[np.float64]
        Initial population of each state [N1, N2, N3, N4, N5], shape (5,).
    t_slices : int
        Number of time slices for pulse integration.
    z_slices : int
        Number of z-slices for sample integration.
    sample_width : float
        Sample width (cm).
    tau : float
        Pulse duration (seconds).
    wavelength : float
        Laser wavelength (meters).
    w0 : float
        Beam waist radius (meters).
    M2 : float
        Beam quality factor.
    pulse_energy : NDArray[np.float64]
        Pulse energy for each dataset (Joules), shape (num_datasets,).
    spec_pars : NDArray[np.float64]
        Initial spectroscopic parameters [k1, k2, k3, tau1, tau2, tau3, sigma, tau0], shape (8,).
        k1, k2, k3: Two-photon absorption coefficients (m^4/W)
        tau1, tau2, tau3: Lifetimes (s)
        sigma: Cross-section (m^2)
        tau0: Ground state recovery time (s)
    num_x_pts : int
        Number of x-points per dataset.
    num_datasets : int
        Number of datasets (for multi-pulse-energy fitting).
    is_sa : bool, optional
        Whether the sample is a saturable absorber (SA) vs reverse saturable absorber (RSA).
        For SA, centers around the maximum of y_data (minimum transmission).
        For RSA, centers around the minimum of y_data (maximum transmission).
        Default is False (RSA).
    n_starts : int, optional
        Number of parallel optimization starts (default=3).
        Higher values give better global minimum finding but slower.

    Returns
    -------
    residuals : NDArray[np.float64]
        Fit residuals (data - model), same shape as y_data.
    error : float
        RMS fitting error.
    result : NDArray[np.float64]
        Fitted spectroscopic parameters, shape (8,).

    Examples
    --------
    >>> import numpy as np
    >>> import zfit_wrapper
    >>> x_data = np.linspace(-20, 20, 101)
    >>> y_data = np.ones(101)  # your transmittance data
    >>> populations = np.array([1.75e18, 0, 0, 0, 0])
    >>> pulse_energy = np.array([50e-9])
    >>> spec_pars = np.array([4.94e-18, 1.60e-17, 1.5e-17, 1e-12, 1e-12, 1e-12, 1.29e-7, 1e-12])
    >>> residuals, error, result = zfit_wrapper.fit_zscan(
    ...     x_data, y_data, populations,
    ...     t_slices=11, z_slices=11, sample_width=0.1, tau=8e-9,
    ...     wavelength=532e-9, w0=14.5e-6, M2=1.0,
    ...     pulse_energy=pulse_energy, spec_pars=spec_pars,
    ...     num_x_pts=101, num_datasets=1, n_starts=3
    ... )
    >>> print(f"Fit error: {error:.6e}")
    >>> print(f"Fitted k3: {result[2]:.2e} m^4/W")
    """
    ...
