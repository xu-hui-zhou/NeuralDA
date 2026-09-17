#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import pyvista as pv
import yaml


# ============================================================
# Paths
# ============================================================

POSTPROCESS_DIR = Path(__file__).resolve().parent
BASE = POSTPROCESS_DIR.parent

CONFIG_FILE = BASE / "input.yaml"
FIGURE_DIR = BASE / "figures"


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 18,
})


# ============================================================
# Colormaps
# ============================================================

CMAPS = {
    "rho": cmc.batlowW,
    "u": cmc.berlin,
    "p": cmc.lipari,
}


# ============================================================
# VTR utilities
# ============================================================

def as_1d(array):
    """Convert an array to a one-dimensional float array."""

    return np.asarray(
        array,
        dtype=float,
    ).ravel()


def read_vtr_scalar_field(
    vtr_file,
    variable,
):
    """Read one scalar flow variable from a 2D M2C VTR file."""

    vtr_file = Path(
        vtr_file
    )

    if not vtr_file.is_file():
        raise FileNotFoundError(
            f"VTR file not found: {vtr_file}"
        )

    solution = pv.read(
        str(vtr_file)
    )

    x = as_1d(
        solution.points[:, 0]
    )

    y = as_1d(
        solution.points[:, 1]
    )

    if variable == "rho":

        field = as_1d(
            solution.point_data["density"]
        )

    elif variable == "u":

        velocity = np.asarray(
            solution.point_data["velocity"]
        )

        field = as_1d(
            velocity[:, 0]
        )

    elif variable == "p":

        field = as_1d(
            solution.point_data["pressure"]
        )

    else:

        raise ValueError(
            f"Unknown flow variable: {variable}"
        )

    nx, ny, _ = (
        solution.dimensions
    )

    return (
        x.reshape(ny, nx),
        y.reshape(ny, nx),
        field.reshape(ny, nx),
    )


# ============================================================
# Plot utilities
# ============================================================

def imshow_field(
    ax,
    x,
    y,
    field,
    vmin,
    vmax,
    cmap,
):
    """Plot one two-dimensional flow field."""

    image = ax.imshow(
        field,
        origin="lower",
        extent=(
            x.min(),
            x.max(),
            y.min(),
            y.max(),
        ),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
        aspect="equal",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    return image


def round_max_label(value):
    """Round the upper colorbar limit for display."""

    if value > 10:

        rounded = int(
            np.round(value)
        )

        return (
            float(rounded),
            f"{rounded:d}",
        )

    rounded = float(
        np.round(
            value,
            1,
        )
    )

    return (
        rounded,
        f"{rounded:.1f}",
    )


def round_min_label(value):
    """Round the lower colorbar limit for display."""

    if abs(value) > 10:

        rounded = int(
            np.round(value)
        )

        return (
            float(rounded),
            f"{rounded:d}",
        )

    rounded = float(
        np.round(
            value,
            1,
        )
    )

    return (
        rounded,
        f"{rounded:.1f}",
    )


def sci_tex(
    value,
    sig=2,
):
    """Format a time value in scientific notation for LaTeX."""

    if value == 0.0:
        return r"$t = 0$"

    exponent = int(
        np.floor(
            np.log10(
                abs(value)
            )
        )
    )

    mantissa = (
        value
        / (10.0 ** exponent)
    )

    mantissa = float(
        np.round(
            mantissa,
            sig - 1,
        )
    )

    if abs(mantissa) >= 10.0:

        mantissa /= 10.0
        exponent += 1

    mantissa_string = (
        f"{mantissa:g}"
    )

    return (
        rf"$t = {mantissa_string} "
        rf"\times 10^{{{exponent}}}$"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Command-line options
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Plot truth, ensemble mean, and a representative "
            "sample for the 2D Neural EnKF example."
        )
    )

    parser.add_argument(
        "--sample",
        default=None,
        help=(
            "Ensemble-member directory used for the sample row. "
            "Default: <case_name>_004."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 2. Load configuration
    # --------------------------------------------------------

    with CONFIG_FILE.open("r") as file:

        config = yaml.safe_load(
            file
        )

    case_name = (
        config["case"]["name"]
    )

    da_interval = (
        config["assimilation"]["interval"]
    )

    sample_name = (
        args.sample
        if args.sample is not None
        else f"{case_name}_004"
    )

    # --------------------------------------------------------
    # 3. Input and output directories
    # --------------------------------------------------------

    truth_dir = (
        BASE
        / f"{case_name}.truth"
        / "results"
    )

    mean_dir = (
        BASE
        / f"{case_name}.mean"
        / "results"
    )

    sample_dir = (
        BASE
        / sample_name
        / "results"
    )

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"Truth directory not found: {truth_dir}"
        )

    if not mean_dir.is_dir():
        raise FileNotFoundError(
            f"Mean directory not found: {mean_dir}"
        )

    if not sample_dir.is_dir():
        raise FileNotFoundError(
            f"Sample directory not found: {sample_dir}"
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 4. Plot layout
    # --------------------------------------------------------

    # Each entry is (DA cycle, state type).
    #
    # k = -1:
    #     initial condition
    #
    # k >= 0, "forecast":
    #     forecast state at DA cycle k
    #
    # k >= 0, "analysis":
    #     analysis state after DA cycle k
    columns = [
        (-1, "initial"),
        (0, "forecast"),
        (0, "analysis"),
        (1, "analysis"),
        (2, "analysis"),
        (3, "analysis"),
        (4, "analysis"),
    ]

    # Physical time between observations.
    dt_obs = 4.0e-3

    column_titles = []

    for k, mode in columns:

        if k == -1:

            column_titles.append(
                "$t = 0$,\nInitial"
            )

        else:

            time_value = (
                (k + 1)
                * dt_obs
            )

            time_string = sci_tex(
                time_value,
                sig=2,
            )

            if mode == "forecast":

                column_titles.append(
                    f"{time_string},\nForecast"
                )

            else:

                column_titles.append(
                    f"{time_string},\nAnalysis"
                )

    row_titles = [
        "Truth",
        "Ensemble mean",
        "Farthest sample",
    ]

    # --------------------------------------------------------
    # 5. File mapping
    # --------------------------------------------------------

    def truth_file(
        k,
        mode,
    ):
        """Return the truth VTR corresponding to one column."""

        if k == -1:

            return (
                truth_dir
                / "solution_0000.vtr"
            )

        index = (
            (k + 1)
            * da_interval
        )

        return (
            truth_dir
            / f"solution_{index:04d}.vtr"
        )

    def member_file(
        results_dir,
        k,
        mode,
    ):
        """Return the ensemble-member VTR corresponding to one column."""

        if k == -1:

            return (
                results_dir
                / "solution_0000_0000.vtr"
            )

        if mode == "forecast":

            return (
                results_dir
                / (
                    f"solution_{k:04d}_"
                    f"{da_interval:04d}.vtr"
                )
            )

        if mode == "analysis":

            return (
                results_dir
                / (
                    f"solution_{k + 1:04d}_"
                    "0000.vtr"
                )
            )

        raise ValueError(
            f"Unknown state type: {mode}"
        )

    # --------------------------------------------------------
    # 6. Plot one flow variable
    # --------------------------------------------------------

    def plot_field(variable):

        panels = [
            [None] * len(columns)
            for _ in range(3)
        ]

        vmins = []
        vmaxs = []
        ticklabels = []

        # ----------------------------------------------------
        # Read fields and determine color limits
        # ----------------------------------------------------

        for j, (k, mode) in enumerate(
            columns
        ):

            truth_path = truth_file(
                k,
                mode,
            )

            mean_path = member_file(
                mean_dir,
                k,
                mode,
            )

            sample_path = member_file(
                sample_dir,
                k,
                mode,
            )

            if not truth_path.is_file():
                raise FileNotFoundError(
                    f"Missing truth file: {truth_path}"
                )

            if not mean_path.is_file():
                raise FileNotFoundError(
                    f"Missing mean file: {mean_path}"
                )

            if not sample_path.is_file():
                raise FileNotFoundError(
                    f"Missing sample file: {sample_path}"
                )

            (
                x_truth,
                y_truth,
                field_truth,
            ) = read_vtr_scalar_field(
                truth_path,
                variable,
            )

            (
                x_mean,
                y_mean,
                field_mean,
            ) = read_vtr_scalar_field(
                mean_path,
                variable,
            )

            (
                x_sample,
                y_sample,
                field_sample,
            ) = read_vtr_scalar_field(
                sample_path,
                variable,
            )

            panels[0][j] = (
                x_truth,
                y_truth,
                field_truth,
            )

            panels[1][j] = (
                x_mean,
                y_mean,
                field_mean,
            )

            panels[2][j] = (
                x_sample,
                y_sample,
                field_sample,
            )

            raw_min = float(
                min(
                    field_truth.min(),
                    field_mean.min(),
                    field_sample.min(),
                )
            )

            raw_max = float(
                max(
                    field_truth.max(),
                    field_mean.max(),
                    field_sample.max(),
                )
            )

            # ------------------------------------------------
            # Color limits
            # ------------------------------------------------

            if variable == "p":

                vmin = 0.1
                vmax = raw_max

            elif variable == "rho":

                if j == 0:

                    vmin = 0.9
                    vmax = 1.1

                else:

                    vmin = raw_min
                    vmax = raw_max

            elif variable == "u":

                if j == 0:

                    vmin = -0.1
                    vmax = 0.1

                else:

                    amplitude = float(
                        max(
                            abs(raw_min),
                            abs(raw_max),
                        )
                    )

                    vmin = -amplitude
                    vmax = amplitude

            else:

                raise ValueError(
                    f"Unknown flow variable: {variable}"
                )

            (
                vmin_rounded,
                vmin_label,
            ) = round_min_label(
                vmin
            )

            (
                vmax_rounded,
                vmax_label,
            ) = round_max_label(
                vmax
            )

            if vmin_rounded > vmax_rounded:

                (
                    vmin_rounded,
                    vmax_rounded,
                ) = (
                    vmax_rounded,
                    vmin_rounded,
                )

                (
                    vmin_label,
                    vmax_label,
                ) = (
                    vmax_label,
                    vmin_label,
                )

            vmins.append(
                vmin_rounded
            )

            vmaxs.append(
                vmax_rounded
            )

            ticklabels.append(
                (
                    vmin_label,
                    vmax_label,
                )
            )

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        cmap = (
            CMAPS[variable]
        )

        figure_width = (
            3.6
            * len(columns)
        )

        fig, axes = plt.subplots(
            3,
            len(columns),
            figsize=(
                figure_width,
                8.0,
            ),
        )

        fig.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.90,
            bottom=0.06,
            wspace=-0.5,
            hspace=0.10,
        )

        # ----------------------------------------------------
        # Column titles
        # ----------------------------------------------------

        for j in range(
            len(columns)
        ):

            axes[0, j].set_title(
                column_titles[j],
                pad=12,
            )

        # ----------------------------------------------------
        # Plot panels
        # ----------------------------------------------------

        for j in range(
            len(columns)
        ):

            vmin = (
                vmins[j]
            )

            vmax = (
                vmaxs[j]
            )

            (
                vmin_label,
                vmax_label,
            ) = ticklabels[j]

            # Truth
            image = imshow_field(
                axes[0, j],
                *panels[0][j],
                vmin,
                vmax,
                cmap,
            )

            # Colorbar
            position = (
                axes[0, j]
                .get_position()
            )

            colorbar_axis = (
                fig.add_axes([
                    position.x1 + 0.004,
                    position.y0
                    + position.height / 5,
                    0.005,
                    position.height * 2 / 3,
                ])
            )

            colorbar = fig.colorbar(
                image,
                cax=colorbar_axis,
            )

            colorbar.ax.tick_params(
                labelsize=14
            )

            colorbar.set_ticks([
                vmin,
                vmax,
            ])

            colorbar.set_ticklabels([
                vmin_label,
                vmax_label,
            ])

            # Ensemble mean
            imshow_field(
                axes[1, j],
                *panels[1][j],
                vmin,
                vmax,
                cmap,
            )

            # Farthest sample
            imshow_field(
                axes[2, j],
                *panels[2][j],
                vmin,
                vmax,
                cmap,
            )

        # ----------------------------------------------------
        # Row labels
        # ----------------------------------------------------

        for i in range(3):

            axes[i, 0].text(
                -0.10,
                0.5,
                row_titles[i],
                transform=axes[i, 0].transAxes,
                rotation=90,
                va="center",
                ha="center",
            )

        # ----------------------------------------------------
        # Save figure
        # ----------------------------------------------------

        output_png = (
            FIGURE_DIR
            / f"{case_name}_{variable}_posterior.png"
        )

        fig.savefig(
            output_png,
            dpi=250,
            bbox_inches="tight",
        )

        # Optional PDF output.
        # output_pdf = (
        #     FIGURE_DIR
        #     / f"{case_name}_{variable}_posterior.pdf"
        # )
        #
        # fig.savefig(
        #     output_pdf,
        #     bbox_inches="tight",
        # )

        plt.close(
            fig
        )

        print(
            f"Saved: {output_png}"
        )

    # --------------------------------------------------------
    # 7. Plot flow variables
    # --------------------------------------------------------

    for variable in [
        "rho",
        "u",
        "p",
    ]:

        plot_field(
            variable
        )


if __name__ == "__main__":
    main()