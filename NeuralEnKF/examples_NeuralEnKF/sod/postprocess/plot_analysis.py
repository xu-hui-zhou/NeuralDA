#!/usr/bin/env python3

import argparse
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


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
})


C_TRUTH = "k"
C_OBS = "0.6"

SAMPLE_ALPHA = 0.32

C_SAMPLES_RHO = mcolors.to_rgba(
    "red",
    SAMPLE_ALPHA,
)

C_SAMPLES_U = mcolors.to_rgba(
    "#1b8a6b",
    SAMPLE_ALPHA,
)

C_SAMPLES_P = mcolors.to_rgba(
    "#365F96",
    SAMPLE_ALPHA,
)


# ============================================================
# Command-line options
# ============================================================

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--col-width",
        type=float,
        default=3.3,
    )

    parser.add_argument(
        "--row-height",
        type=float,
        default=1.8,
    )

    parser.add_argument(
        "--wspace",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--hspace",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--pad-inches",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--title-pad",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--da-dt",
        type=float,
        default=0.025,
        help="Physical time between DA cycles.",
    )

    parser.add_argument(
        "--no-plot-obs",
        dest="plot_obs",
        action="store_false",
        help="Disable pressure observations.",
    )

    parser.set_defaults(
        plot_obs=True
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def format_time(t, max_decimals=3):
    """Format time without unnecessary trailing zeros."""

    return (
        f"{t:.{max_decimals}f}"
        .rstrip("0")
        .rstrip(".")
    )


def as_1d(array):
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

    idx = np.argsort(
        x_source
    )

    return np.interp(
        x_target,
        x_source[idx],
        y_source[idx],
    )


# ============================================================
# Data readers
# ============================================================

def read_vtr(path):
    """Read density, velocity, and pressure from a VTR file."""

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


def read_two_column_file(path):

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


def read_observation(
    obs_dir,
    cycle,
):
    """Read pressure observations from one DA cycle."""

    return read_two_column_file(
        obs_dir
        / f"y_obs_{cycle:04d}.txt"
    )


def read_truth_observation(
    truth_obs_dir,
    cycle,
):
    """Read truth values at the observation locations."""

    return read_two_column_file(
        truth_obs_dir
        / f"y_gt_{cycle:04d}.txt"
    )


# ============================================================
# Plot utilities
# ============================================================

def apply_axes(
    ax_rho,
    ax_u,
    ax_p,
):

    for ax in (
        ax_rho,
        ax_u,
        ax_p,
    ):
        ax.set_ylim(
            -0.08,
            1.28,
        )

        ax.set_yticks(
            np.arange(
                0.0,
                1.21,
                0.4,
            )
        )

        ax.grid(
            True,
            alpha=0.25,
        )


def build_rows(
    da_interval,
    da_dt,
):
    """
    Define the six rows:

    1. Initial state
    2. Forecast at first DA
    3. Analysis at first DA
    4. Analysis at second DA
    5. Analysis at fourth DA
    6. Analysis at eighth DA
    """

    rows = [
        {
            "label": r"Initial, $t=0$",
            "truth_idx": 0,
            "cycle": 0,
            "subindex": 0,
            "obs_cycle": None,
        },
        {
            "label": (
                "Forecast, "
                rf"$t={format_time(da_dt)}$"
            ),
            "truth_idx": da_interval,
            "cycle": 0,
            "subindex": da_interval,
            "obs_cycle": None,
        },
        {
            "label": (
                "Analysis, "
                rf"$t={format_time(da_dt)}$"
            ),
            "truth_idx": da_interval,
            "cycle": 1,
            "subindex": 0,
            "obs_cycle": 0,
        },
        {
            "label": (
                "Analysis, "
                rf"$t={format_time(2 * da_dt)}$"
            ),
            "truth_idx": 2 * da_interval,
            "cycle": 2,
            "subindex": 0,
            "obs_cycle": 1,
        },
        {
            "label": (
                "Analysis, "
                rf"$t={format_time(4 * da_dt)}$"
            ),
            "truth_idx": 4 * da_interval,
            "cycle": 4,
            "subindex": 0,
            "obs_cycle": 3,
        },
        {
            "label": (
                "Analysis, "
                rf"$t={format_time(8 * da_dt)}$"
            ),
            "truth_idx": 8 * da_interval,
            "cycle": 8,
            "subindex": 0,
            "obs_cycle": 7,
        },
    ]

    return rows


# ============================================================
# Main plotting routine
# ============================================================

def plot_analysis(
    run_results,
    truth_dir,
    obs_dir,
    truth_obs_dir,
    rows,
    relative_noise,
    absolute_noise,
    figsize,
    dpi,
    output_file,
    wspace,
    hspace,
    title_pad,
    pad_inches,
    plot_obs,
):

    n_rows = len(
        rows
    )

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=figsize,
        sharex=True,
    )

    axes = np.atleast_2d(
        axes
    )

    linewidth_member = 1.0
    linewidth_truth = 2.2

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
            pad=title_pad,
        )

    obs_legend_handle = None

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    for row_index, spec in enumerate(
        rows
    ):

        ax_rho, ax_u, ax_p = (
            axes[row_index]
        )

        # Truth solution
        (
            x_truth,
            rho_truth,
            u_truth,
            p_truth,
        ) = read_truth(
            truth_dir,
            spec["truth_idx"],
        )

        # Ensemble members
        for results_dir in run_results:

            member = read_member_solution(
                results_dir,
                spec["cycle"],
                spec["subindex"],
            )

            if member is None:
                continue

            (
                x_member,
                rho_member,
                u_member,
                p_member,
            ) = member

            ax_rho.plot(
                x_truth,
                interpolate(
                    x_member,
                    rho_member,
                    x_truth,
                ),
                color=C_SAMPLES_RHO,
                lw=linewidth_member,
            )

            ax_u.plot(
                x_truth,
                interpolate(
                    x_member,
                    u_member,
                    x_truth,
                ),
                color=C_SAMPLES_U,
                lw=linewidth_member,
            )

            ax_p.plot(
                x_truth,
                interpolate(
                    x_member,
                    p_member,
                    x_truth,
                ),
                color=C_SAMPLES_P,
                lw=linewidth_member,
            )

        # Truth
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
        # Pressure observations
        # ----------------------------------------------------

        obs_cycle = spec["obs_cycle"]

        if (
            plot_obs
            and obs_cycle is not None
        ):

            obs = read_observation(
                obs_dir,
                obs_cycle,
            )

            truth_obs = read_truth_observation(
                truth_obs_dir,
                obs_cycle,
            )

            if (
                obs is not None
                and truth_obs is not None
            ):

                x_obs, p_obs = obs
                _, p_obs_truth = truth_obs

                sigma = (
                    relative_noise
                    * np.abs(p_obs_truth)
                    + absolute_noise
                )

                errorbar = ax_p.errorbar(
                    x_obs,
                    p_obs,
                    yerr=2.0 * sigma,
                    fmt="o",
                    color=C_OBS,
                    ecolor=C_OBS,
                    capsize=3,
                    markersize=4.0,
                    zorder=10,
                )

                if obs_legend_handle is None:
                    obs_legend_handle = errorbar

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        apply_axes(
            ax_rho,
            ax_u,
            ax_p,
        )

        ax_rho.set_ylabel(
            spec["label"],
            labelpad=15,
        )

        # Only the first column shows y tick labels.
        ax_u.tick_params(
            axis="y",
            left=False,
            labelleft=False,
        )

        ax_p.tick_params(
            axis="y",
            left=False,
            labelleft=False,
        )

        # Only the bottom row shows x ticks and labels.
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
            color=C_SAMPLES_RHO,
            lw=linewidth_member,
            label=r"Sample, $\rho$",
        ),
        Line2D(
            [0],
            [0],
            color=C_SAMPLES_U,
            lw=linewidth_member,
            label=r"Sample, $u$",
        ),
        Line2D(
            [0],
            [0],
            color=C_SAMPLES_P,
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

    axes[0, 1].legend(
        handles=main_handles,
        loc="upper right",
        frameon=True,
    )

    if obs_legend_handle is not None:

        axes[0, 2].legend(
            handles=[
                obs_legend_handle
            ],
            labels=[
                "Obs."
            ],
            loc="upper right",
            frameon=True,
        )

    fig.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
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

    args = get_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    with CONFIG_FILE.open("r") as file:
        config = yaml.safe_load(
            file
        )

    case_name = config["case"]["name"]

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
    # Input directories
    # --------------------------------------------------------

    run_results = sorted(
        BASE.glob(
            f"{case_name}_[0-9][0-9][0-9]/results"
        )
    )

    truth_dir = (
        BASE
        / f"{case_name}.truth"
        / "results"
    )

    obs_dir = (
        BASE
        / "da_results"
        / "obs"
    )

    truth_obs_dir = (
        BASE
        / "da_results"
        / "truth"
    )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    rows = build_rows(
        da_interval=da_interval,
        da_dt=args.da_dt,
    )

    figsize = (
        args.col_width * 3,
        args.row_height * len(rows),
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    output_file = (
        FIGURE_DIR
        / f"{case_name}_analysis.png"
    )

    plot_analysis(
        run_results=run_results,
        truth_dir=truth_dir,
        obs_dir=obs_dir,
        truth_obs_dir=truth_obs_dir,
        rows=rows,
        relative_noise=relative_noise,
        absolute_noise=absolute_noise,
        figsize=figsize,
        dpi=args.dpi,
        output_file=output_file,
        wspace=args.wspace,
        hspace=args.hspace,
        title_pad=args.title_pad,
        pad_inches=args.pad_inches,
        plot_obs=args.plot_obs,
    )


if __name__ == "__main__":
    main()