# zfit

Z-scan fitting library for nonlinear optics. Fits Z-scan data to extract two-photon absorption coefficients and other nonlinear optical parameters.

## Installation

```bash
pip install zfit
```

Or install from source:

```bash
git clone https://github.com/yourname/zfit.git
cd zfit
pip install .
```

### Requirements

- Python >= 3.9
- NumPy >= 1.20
- gfortran or Intel Fortran compiler

### Prebuilt Wheels

PyPI wheels are built for Linux, Windows, and macOS (Apple Silicon `arm64`, macOS 14+) for Python 3.9 through 3.14.

## Quick Start

```python
import numpy as np
import zfit_wrapper

x_data = np.linspace(-20, 20, 101)
y_data = ...  # normalized transmittance values

populations = np.array([1.75e18, 0, 0, 0, 0], dtype=np.float64)
pulse_energy = np.array([50e-9], dtype=np.float64)
spec_pars = np.array(
    [k1, k2, k3, tau1, tau2, tau3, sigma, tau0],
    dtype=np.float64,
)

residuals, error, result = zfit_wrapper.fit_zscan(
    x_data, y_data, populations,
    t_slices=11, z_slices=11,
    sample_width=0.1, tau=8e-9,
    wavelength=532e-9, w0=14.5e-6, M2=1.0,
    pulse_energy=pulse_energy,
    spec_pars=spec_pars,
    num_x_pts=len(x_data),
    num_datasets=1,
    n_starts=3,
)

print(f"Fit error: {error:.6e}")
print(f"Fitted k3: {result[2]:.2e} m^4/W")
```

## Multi-Start Optimization

The `n_starts` parameter controls parallel multi-start optimization for finding the global minimum:

- `n_starts=1`: Single optimization run (fast but may find local minimum)
- `n_starts=3`: Default, good balance of speed and accuracy
- `n_starts=5+`: More thorough search for global minimum

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest test/ -v
```

### Building from Source

```bash
cmake -B build -S .
cmake --build build
```

## License

MIT
