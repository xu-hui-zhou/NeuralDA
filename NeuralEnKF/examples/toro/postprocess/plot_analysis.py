#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pyvista as pv
import yaml


# ============================================================
# Paths
# ============================================================

POSTPROCESS_DIR = Path(__file__).resolve().parent
BASE = POSTPROCESS_DIR.parent

CONFIG_FILE = BASE / "input.yaml"
FIGURE_DIR = BASE / "figures"

DA_RESULTS = BASE / "da_results"
OBS_DIR = DA_RESULTS / "obs"
TRUTH_OBS_DIR = DA_RESULTS / "truth"


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


C_TRUTH = "black"
C_OBS = "0.6"

SAMPLE_ALPHA = 0.45

C_RHO = mcolors.to_rgba(
    "red",
    SAMPLE_ALPHA,
)

C_U = mcolors.to_rgba(
    "#1b8a6b",
    SAMPLE_ALPHA,
)

C_P = mcolors.to_rgba(
    "#365F96",
    SAMPLE_ALPHA,
)


# ============================================================
# Variable-specific axes
# ============================================================

Y_LIM = {
    "rho": (0.0, 35.0),
    "u": (-10.0, 25.0),
    "p": (-50.0, 2050.0),
}

Y_TICKS = {
    "rho": [0.0, 10.0, 20.0, 30.0],
    "u": [-10.0, 0.0, 10.0, 20.0],
    "p": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
}


# ============================================================
# Utilities
# ============================================================

def as_1d(array):
    """Convert an array to a one-dimensional float array."""

    return np.asarray(
        array,
        dtype=float,
    ).ravel()


def interpolate(
    x_source,
    y_source,
    x_target,
):
    """Interpolate a field onto the target grid."""

    x_source = as_1d(
        x_source
    )

    y_source = as_1d(
        y_source
    )

    order = np.argsort(
        x_source
    )

    return np.interp(
        x_target,
        x_source[order],
        y_source[order],
    )


# ============================================================
# Data readers
# ============================================================

def read_vtr(path):
    """Read density, velocity, and pressure from a VTR file."""

    if not path.is_file():
        raise FileNotFoundError(
            f"VTR file not found: {path}"
        )

    grid = pv.read(
        str(path)
    )

    return (
        as_1d(
            grid.points[:, 0]
        ),
        as_1d(
            grid.point_data["density"]
        ),
        as_1d(
            grid.point_data["velocity"][:, 0]
        ),
        as_1d(
            grid.point_data["pressure"]
        ),
    )


def read_member_solution(
    results_dir,
    cycle,
    subindex,
):
    """Read one ensemble-member solution."""

    path = (
        results_dir
        / f"solution_{cycle:04d}_{subindex:04d}.vtr"
    )

    if not path.is_file():
        return None

    return read_vtr(
        path
    )


def read_truth(
    truth_dir,
    index,
):
    """Read one truth solution."""

    path = (
        truth_dir
        / f"solution_{index:04d}.vtr"
    )

    return read_vtr(
        path
    )


def read_two_column_file(path):
    """Read a two-column text file."""

    if not path.is_file():
        return None

    data = np.loadtxt(
        path
    )

    if data.ndim == 1:
        data = data[None, :]

    return (
        as_1d(data[:, 0]),
        as_1d(data[:, 1]),
    )


# ============================================================
# Plot rows
# ============================================================

def build_rows(
    da_interval,
):
    """
    Define the six rows shown in the figure.

    The analysis after DA cycle n is represented by
    solution_n_0000.vtr.
    """

    return [
        {
            "label": "Initial",
            "truth_index": 0,
            "cycle": 0,
            "subindex": 0,
            "obs_cycle": None,
        },
        {
            "label": "Forecast\n1st DA",
            "truth_index": da_interval,
            "cycle": 0,
            "subindex": da_interval,
            "obs_cycle": None,
        },
        {
            "label": "Analysis\n1st DA",
            "truth_index": da_interval,
            "cycle": 1,
            "subindex": 0,
            "obs_cycle": 0,
        },
        {
            "label": "Analysis\n2nd DA",
            "truth_index": 2 * da_interval,
            "cycle": 2,
            "subindex": 0,
            "obs_cycle": 1,
        },
        {
            "label": "Analysis\n4th DA",
            "truth_index": 4 * da_interval,
            "cycle": 4,
            "subindex": 0,
            "obs_cycle": 3,
        },
        {
            "label": "Analysis\n7th DA",
            "truth_index": 7 * da_interval,
            "cycle": 7,
            "subindex": 0,
            "obs_cycle": 6,
        },
    ]


# ============================================================
# Plot
# ============================================================

def plot_analysis(
    ensemble_results,
    truth_dir,
    rows,
    relative_noise,
    absolute_noise,
    output_file,
):

    n_rows = len(
        rows
    )

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(10.0, 12.0),
        sharex=True,
        sharey="col",
    )

    titles = [
        r"Density, $\rho$",
        r"Velocity, $u$",
        r"Pressure, $p$",
    ]

    for j, title in enumerate(
        titles
    ):
        axes[0, j].set_title(
            title,
            pad=7,
        )

    linewidth_member = 0.9
    linewidth_truth = 2.0

    obs_legend_handle = None

    # --------------------------------------------------------
    # Plot rows
    # --------------------------------------------------------

    for row_index, spec in enumerate(
        rows
    ):

        ax_rho, ax_u, ax_p = (
            axes[row_index]
        )

        # ----------------------------------------------------
        # Truth
        # ----------------------------------------------------

        (
            x_truth,
            rho_truth,
            u_truth,
            p_truth,
        ) = read_truth(
            truth_dir,
            spec["truth_index"],
        )

        # ----------------------------------------------------
        # Ensemble
        # ----------------------------------------------------

        for results_dir in ensemble_results:

            sample = read_member_solution(
                results_dir,
                spec["cycle"],
                spec["subindex"],
            )

            if sample is None:
                continue

            (
                x_sample,
                rho_sample,
                u_sample,
                p_sample,
            ) = sample

            ax_rho.plot(
                x_truth,
                interpolate(
                    x_sample,
                    rho_sample,
                    x_truth,
                ),
                color=C_RHO,
                lw=linewidth_member,
            )

            ax_u.plot(
                x_truth,
                interpolate(
                    x_sample,
                    u_sample,
                    x_truth,
                ),
                color=C_U,
                lw=linewidth_member,
            )

            ax_p.plot(
                x_truth,
                interpolate(
                    x_sample,
                    p_sample,
                    x_truth,
                ),
                color=C_P,
                lw=linewidth_member,
            )

        # ----------------------------------------------------
        # Truth
        # ----------------------------------------------------

        ax_rho.plot(
            x_truth,
            rho_truth,
            color=C_TRUTH,
            lw=linewidth_truth,
        )

        ax_u.plot(
            x_truth,
            u_truth,
            color=C_TRUTH,
            lw=linewidth_truth,
        )

        ax_p.plot(
            x_truth,
            p_truth,
            color=C_TRUTH,
            lw=linewidth_truth,
        )

        # ----------------------------------------------------
        # Observations
        # ----------------------------------------------------

        obs_cycle = spec[
            "obs_cycle"
        ]

        if obs_cycle is not None:

            obs_data = read_two_column_file(
                OBS_DIR
                / f"y_obs_{obs_cycle:04d}.txt"
            )

            truth_obs_data = read_two_column_file(
                TRUTH_OBS_DIR
                / f"y_gt_{obs_cycle:04d}.txt"
            )

            if (
                obs_data is not None
                and truth_obs_data is not None
            ):

                x_obs, obs = (
                    obs_data
                )

                _, truth_obs = (
                    truth_obs_data
                )

                sigma_obs = (
                    relative_noise
                    * np.abs(truth_obs)
                    + absolute_noise
                )

                errorbar = ax_p.errorbar(
                    x_obs,
                    obs,
                    yerr=2.0 * sigma_obs,
                    fmt="o",
                    color=C_OBS,
                    ecolor=C_OBS,
                    capsize=3,
                    elinewidth=1.0,
                    markersize=4.0,
                    zorder=10,
                )

                if obs_legend_handle is None:
                    obs_legend_handle = (
                        errorbar
                    )

        # ----------------------------------------------------
        # Variable-specific y axes
        # ----------------------------------------------------

        ax_rho.set_ylim(
            *Y_LIM["rho"]
        )

        ax_rho.set_yticks(
            Y_TICKS["rho"]
        )

        ax_u.set_ylim(
            *Y_LIM["u"]
        )

        ax_u.set_yticks(
            Y_TICKS["u"]
        )

        ax_p.set_ylim(
            *Y_LIM["p"]
        )

        ax_p.set_yticks(
            Y_TICKS["p"]
        )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------

        for ax in (
            ax_rho,
            ax_u,
            ax_p,
        ):
            ax.grid(
                True,
                alpha=0.25,
            )

            # Show y ticks and labels in every column.
            ax.tick_params(
                axis="y",
                left=True,
                labelleft=True,
            )

        # Row label on the left.
        ax_rho.set_ylabel(
            spec["label"],
            labelpad=12,
        )

        # Only show x labels on the bottom row.
        if row_index == n_rows - 1:

            for ax in (
                ax_rho,
                ax_u,
                ax_p,
            ):
                ax.set_xlabel(
                    r"$x$"
                )

                ax.set_xticks([
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                ])

        else:

            for ax in (
                ax_rho,
                ax_u,
                ax_p,
            ):
                ax.tick_params(
                    axis="x",
                    bottom=False,
                    labelbottom=False,
                )

    # --------------------------------------------------------
    # Legends
    # --------------------------------------------------------

    main_handles = [
        Line2D(
            [0],
            [0],
            color=C_RHO,
            lw=linewidth_member,
            label=r"Sample, $\rho$",
        ),
        Line2D(
            [0],
            [0],
            color=C_U,
            lw=linewidth_member,
            label=r"Sample, $u$",
        ),
        Line2D(
            [0],
            [0],
            color=C_P,
            lw=linewidth_member,
            label=r"Sample, $p$",
        ),
        Line2D(
            [0],
            [0],
            color=C_TRUTH,
            lw=linewidth_truth,
            label="Truth",
        ),
    ]

    axes[0, 0].legend(
        handles=main_handles,
        loc="best",
        frameon=True,
    )

    if obs_legend_handle is not None:

        axes[0, 2].legend(
            handles=[
                obs_legend_handle
            ],
            labels=[
                "Observation"
            ],
            loc="best",
            frameon=True,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.subplots_adjust(
        wspace=0.28,
        hspace=0.12,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    with CONFIG_FILE.open("r") as file:
        config = yaml.safe_load(
            file
        )

    case_name = config[
        "case"
    ]["name"]

    da_interval = config[
        "assimilation"
    ]["interval"]

    relative_noise = config[
        "observations"
    ]["relative_noise"]

    absolute_noise = config[
        "observations"
    ]["absolute_noise"]

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    ensemble_results = sorted(
        BASE.glob(
            f"{case_name}_[0-9][0-9][0-9]/results"
        )
    )

    if not ensemble_results:
        raise RuntimeError(
            "No ensemble result directories found."
        )

    truth_dir = (
        BASE
        / f"{case_name}.truth"
        / "results"
    )

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"Truth directory not found: {truth_dir}"
        )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    rows = build_rows(
        da_interval
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    output_file = (
        FIGURE_DIR
        / f"{case_name}_analysis.png"
    )

    plot_analysis(
        ensemble_results=ensemble_results,
        truth_dir=truth_dir,
        rows=rows,
        relative_noise=relative_noise,
        absolute_noise=absolute_noise,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()