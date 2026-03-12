"""Tests for zfit package."""

import numpy as np
import pytest

# Try to import zfit_wrapper
# Will be skipped if library not built
zfit_wrapper = pytest.importorskip("zfit_wrapper")


@pytest.fixture
def sample_data():
    """Sample z-scan data for testing."""
    x_data = np.array(
        [
            -20,
            -19.6,
            -19.2,
            -18.8,
            -18.4,
            -18,
            -17.6,
            -17.2,
            -16.8,
            -16.4,
            -16,
            -15.6,
            -15.2,
            -14.8,
            -14.4,
            -14,
            -13.6,
            -13.2,
            -12.8,
            -12.4,
            -12,
            -11.6,
            -11.2,
            -10.8,
            -10.4,
            -10,
            -9.6,
            -9.2,
            -8.8,
            -8.4,
            -8,
            -7.6,
            -7.2,
            -6.8,
            -6.4,
            -6,
            -5.6,
            -5.2,
            -4.8,
            -4.4,
            -4,
            -3.6,
            -3.2,
            -2.8,
            -2.4,
            -2,
            -1.6,
            -1.2,
            -0.8,
            -0.4,
            0,
            0.4,
            0.8,
            1.2,
            1.6,
            2,
            2.4,
            2.8,
            3.2,
            3.6,
            4,
            4.4,
            4.8,
            5.2,
            5.6,
            6,
            6.4,
            6.8,
            7.2,
            7.6,
            8,
            8.4,
            8.8,
            9.2,
            9.6,
            10,
            10.4,
            10.8,
            11.2,
            11.6,
            12,
            12.4,
            12.8,
            13.2,
            13.6,
            14,
            14.4,
            14.8,
            15.2,
            15.6,
            16,
            16.4,
            16.8,
            17.2,
            17.6,
            18,
            18.4,
            18.8,
            19.2,
            19.6,
            20,
        ],
        dtype=np.float64,
    )

    y_data = np.array(
        [
            0.99938049,
            0.999948711,
            0.999693921,
            0.999684822,
            1.000040719,
            1.000473457,
            0.999833449,
            1.000660505,
            0.999799073,
            0.99987187,
            1.000612984,
            1.000568497,
            0.999946689,
            1.000470424,
            1.000141826,
            0.999958822,
            1.000425936,
            0.99988198,
            0.999398689,
            0.999280394,
            0.999709087,
            0.999688866,
            0.999214674,
            0.999771774,
            0.998560511,
            0.999595848,
            1.000216645,
            0.999977021,
            0.999608991,
            0.998764748,
            0.99953215,
            0.999568549,
            0.998782947,
            0.998562534,
            0.998010489,
            0.997421035,
            0.996983241,
            0.997704135,
            0.997324983,
            0.996435241,
            0.995481802,
            0.995153204,
            0.993742761,
            0.990998715,
            0.987503445,
            0.984057716,
            0.97713795,
            0.969236434,
            0.964720993,
            0.971656937,
            0.976806319,
            0.984103214,
            0.989455822,
            0.991474929,
            0.993757927,
            0.995304864,
            0.99647164,
            0.996805293,
            0.997575729,
            0.99813384,
            0.998915397,
            0.998780925,
            0.999250061,
            0.999558438,
            0.999593825,
            0.998703072,
            0.998987183,
            0.999859737,
            0.999104467,
            0.999028637,
            0.998929552,
            0.999029648,
            0.99987187,
            1.000502778,
            0.99947553,
            0.999590792,
            0.999771774,
            0.999506873,
            0.999450253,
            0.999407788,
            0.99906908,
            0.999091323,
            0.999929501,
            0.999283427,
            0.999002349,
            0.998278423,
            0.999000327,
            0.999042792,
            0.999002349,
            0.999780873,
            0.999997243,
            0.998482659,
            0.999986121,
            0.999459353,
            0.999092334,
            1.000695892,
            0.999741442,
            0.999422954,
            0.999280394,
            1.00025911,
            0.999802106,
        ],
        dtype=np.float64,
    )

    populations = np.array([1.75e18, 0, 0, 0, 0], dtype=np.float64)
    pulse_energy = np.array([50e-9], dtype=np.float64)
    spec_pars = np.array(
        [4.94e-18, 1.60e-17, 1.5e-17, 1.00e-12, 1.00e-12, 1.00e-12, 1.29e-7, 1.00e-12],
        dtype=np.float64,
    )

    return x_data, y_data, populations, pulse_energy, spec_pars


def test_import():
    """Test that zfit_wrapper can be imported."""
    assert zfit_wrapper is not None


def test_fit_zscan_exists():
    """Test that fit_zscan function exists."""
    assert hasattr(zfit_wrapper, "fit_zscan")


def test_fit_zscan_basic(sample_data):
    """Test basic fitting functionality."""
    x_data, y_data, populations, pulse_energy, spec_pars = sample_data

    residuals, error, result = zfit_wrapper.fit_zscan(
        x_data,
        y_data,
        populations,
        t_slices=11,
        z_slices=11,
        sample_width=0.1,
        tau=9e-9,
        wavelength=532e-9,
        w0=14.5e-6,
        M2=1.2,
        pulse_energy=pulse_energy,
        spec_pars=spec_pars,
        num_x_pts=len(x_data),
        num_datasets=1,
        is_sa=False,
        n_starts=3,
    )

    # Check output shapes and types
    assert isinstance(residuals, np.ndarray)
    assert residuals.shape == y_data.shape
    assert isinstance(error, float)
    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)

    print(f"\n--- Basic fit ---")
    print(f"Error: {error:.6e}")
    print(f"Result: {result}")

    # Check that error is reasonable (should be < 0.1 for good fit)
    assert error < 0.1, f"Error {error} is too large"


def test_fit_zscan_multi_start(sample_data):
    """Test multi-start optimization improves results."""
    x_data, y_data, populations, pulse_energy, spec_pars = sample_data

    # Single start
    residuals_1, error_1, result_1 = zfit_wrapper.fit_zscan(
        x_data,
        y_data,
        populations,
        t_slices=11,
        z_slices=11,
        sample_width=0.1,
        tau=9e-9,
        wavelength=532e-9,
        w0=14.5e-6,
        M2=1.2,
        pulse_energy=pulse_energy,
        spec_pars=spec_pars.copy(),
        num_x_pts=len(x_data),
        num_datasets=1,
        is_sa=False,
        n_starts=1,
    )

    # Multiple starts
    residuals_3, error_3, result_3 = zfit_wrapper.fit_zscan(
        x_data,
        y_data,
        populations,
        t_slices=11,
        z_slices=11,
        sample_width=0.1,
        tau=9e-9,
        wavelength=532e-9,
        w0=14.5e-6,
        M2=1.2,
        pulse_energy=pulse_energy,
        spec_pars=spec_pars.copy(),
        num_x_pts=len(x_data),
        num_datasets=1,
        is_sa=False,
        n_starts=3,
    )

    # Multi-start should generally give equal or better results
    assert error_3 <= error_1 + 1e-6, (
        f"Multi-start error {error_3} should be <= single-start {error_1}"
    )

    print(f"\n--- Multi-start comparison ---")
    print(f"Single-start error: {error_1:.6e}, result: {result_1}")
    print(f"Multi-start (3) error: {error_3:.6e}, result: {result_3}")


def test_k3_in_expected_range(sample_data):
    """Test that fitted k3 is in physically reasonable range."""
    x_data, y_data, populations, pulse_energy, spec_pars = sample_data

    residuals, error, result = zfit_wrapper.fit_zscan(
        x_data,
        y_data,
        populations,
        t_slices=11,
        z_slices=11,
        sample_width=0.1,
        tau=9e-9,
        wavelength=532e-9,
        w0=14.5e-6,
        M2=1.2,
        pulse_energy=pulse_energy,
        spec_pars=spec_pars,
        num_x_pts=len(x_data),
        num_datasets=1,
        is_sa=False,
        n_starts=5,
    )

    # k3 should be in range 1e-18 to 1e-16
    k3 = result[2]
    print(f"\n--- k3 range test ---")
    print(f"Error: {error:.6e}")
    print(f"Result: {result}")
    print(f"k3 = {k3:.2e}")
    assert 1e-18 <= k3 <= 1e-16, f"k3 = {k3} is outside expected range [1e-18, 1e-16]"
