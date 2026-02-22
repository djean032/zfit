import numpy as np
import time
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import zfit_wrapper


def parse_pzy_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    pulse_energy = None
    z_data = []
    y_data = []

    for line in lines:
        line = line.strip()
        if line.startswith("pulseEnergy_J"):
            parts = line.split()
            pulse_energy = float(parts[1])
        elif line.startswith("$"):
            continue
        elif line and line[0] not in ("%", "$"):
            try:
                parts = line.split()
                if len(parts) >= 2:
                    z_data.append(float(parts[0]))
                    y_data.append(float(parts[1]))
            except:
                pass

    return np.array(z_data), np.array(y_data), pulse_energy


def plot_datasets(x_data, y_data, residuals, num_x_pts, num_datasets, filename):
    fig, axes = plt.subplots(
        num_datasets, 2, figsize=(14, 4 * num_datasets), squeeze=False
    )

    for i in range(num_datasets):
        bidx = i * num_x_pts
        eidx = (i + 1) * num_x_pts

        x = x_data[bidx:eidx]
        y = y_data[bidx:eidx]
        res = residuals[bidx:eidx]

        # Fit plot
        axes[i, 0].scatter(x, y, label="Data", alpha=0.7)
        axes[i, 0].plot(x, y - res, label="Fit", color="orange")
        axes[i, 0].set_xlabel("z position")
        axes[i, 0].set_ylabel("Normalized transmittance")
        axes[i, 0].set_title(f"Dataset {i + 1} - Fit")
        axes[i, 0].legend()
        axes[i, 0].grid(True)

        # Residuals plot
        axes[i, 1].scatter(x, res, alpha=0.7, color="red")
        axes[i, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[i, 1].set_xlabel("z position")
        axes[i, 1].set_ylabel("Residual")
        axes[i, 1].set_title(f"Dataset {i + 1} - Residuals")
        axes[i, 1].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")


if __name__ == "__main__":
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
    num_x_pts = len(x_data)
    num_datasets = 1

    # Test with 1, 3, and 5 starting points
    for n_starts in [1, 3, 5]:
        spec_pars = np.array(
            [
                4.94e-18,
                1.60e-17,
                1.5e-17,
                1.00e-12,
                1.00e-12,
                1.00e-12,
                1.29e-7,
                1.00e-12,
            ],
            dtype=np.float64,
        )

        print(f"\n{'=' * 60}")
        print(f"Running with {n_starts} starting point(s)")
        print(f"{'=' * 60}")

        start = time.time()
        residuals, error, spec_pars_result = zfit_wrapper.fit_zscan(
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
            num_x_pts=num_x_pts,
            num_datasets=num_datasets,
            n_starts=n_starts,
        )
        end = time.time()

        print(f"Elapsed time: {end - start:.4f} seconds")
        print(f"Error: {error}")
        print(f"spec_pars[2] (k3): {spec_pars[2]:.6e} -> {spec_pars_result[2]:.6e}")

        plot_datasets(
            x_data,
            y_data,
            residuals,
            num_x_pts,
            num_datasets,
            f"fit_nstarts_{n_starts}.png",
        )

    # # Multi-dataset test (commented out)
    # data_files = [
    #     "data/3-Br-pbt_532 nm_103.2 nJ.pzy",
    #     "data/3-Br-pbt_532 nm_204.1 nJ.pzy",
    #     "data/3-Br-pbt_532 nm_310 nj.pzy",
    # ]
    #
    # x_data_list = []
    # y_data_list = []
    # pulse_energies = []
    #
    # for filepath in data_files:
    #     z_data, y_data, pulse_energy = parse_pzy_file(filepath)
    #     x_data_list.append(z_data)
    #     y_data_list.append(y_data)
    #     pulse_energies.append(pulse_energy)
    #
    # x_data = np.concatenate(x_data_list)
    # y_data = np.concatenate(y_data_list)
    # pulse_energy = np.array(pulse_energies)
    #
    # populations = np.array([1.75e18, 0, 0, 0, 0], dtype=np.float64)
    # spec_pars = np.array(
    #     [4.94e-18, 1.60e-17, 1.95e-17, 1.00e-12, 1.00e-12, 1.00e-12, 1.29e-7, 1.00e-12],
    #     dtype=np.float64
    # )
    #
    # num_x_pts = len(x_data_list[0])
    # num_datasets = len(data_files)
    #
    # start = time.time()
    # print("\nRunning multiple datasets.\n")
    # residuals, error, spec_pars_result = zfit_wrapper.fit_zscan(
    #     x_data, y_data, populations,
    #     t_slices=11, z_slices=11,
    #     sample_width=0.1, tau=8e-9,
    #     wavelength=532e-9, w0=14.5e-6, M2=1.0,
    #     pulse_energy=pulse_energy,
    #     spec_pars=spec_pars,
    #     num_x_pts=num_x_pts,
    #     num_datasets=num_datasets,
    #     n_starts=3
    # )
    # end = time.time()
    #
    # print(f"Elapsed time: {end - start}")
    # print(f"Error: {error}")
    # print(f"Spec pars: {spec_pars_result}")
    #
    # plot_datasets(x_data, y_data, residuals, num_x_pts, num_datasets, "fit_multiple.png")
