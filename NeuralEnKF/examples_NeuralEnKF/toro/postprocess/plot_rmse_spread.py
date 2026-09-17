#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
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
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


MARKER = "o"
LINE_STYLE = "-"

MARK_SIZE = 7.0
LINE_WIDTH = 2.2
EDGE_WIDTH = 1.8

C_RMSE = "#3171B7"
C_SPREAD = "#C95D2E"


# ============================================================
# Variable-specific y-axis ranges
# ============================================================

Y_LIM = {
    "rho": (0.0, 4.0),
    "u": (0.0, 5.1),
    "p": (0.0, 410.0),
}

Y_TICKS = {
    "rho": np.arange(0.0, 4.1, 1.0),
    "u": np.arange(0.0, 5.1, 1.0),
    "p": np.arange(0.0, 400.1, 100.0),
}


# ============================================================
# Command-line options
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Compute and plot RMSE and ensemble spread."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--k-min",
        type=int,
        default=-1,
        help="First evaluation index. k=-1 denotes the first forecast.",
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=6,
        help="Last analysis index.",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def as_1d(array):
    """Convert an array to a one-dimensional float array."""

    return np.asarray(
        array,
        dtype=float,
    ).ravel()


def interpolate_to_grid(
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

    x_target = as_1d(
        x_target
    )

    order = np.argsort(
        x_source
    )

    return np.interp(
        x_target,
        x_source[order],
        y_source[order],
    )


def rmse(
    truth,
    estimate,
):
    """Compute root-mean-square error."""

    truth = as_1d(
        truth
    )

    estimate = as_1d(
        estimate
    )

    if truth.shape != estimate.shape:
        raise ValueError(
            f"RMSE shape mismatch: "
            f"{truth.shape} vs {estimate.shape}"
        )

    return float(
        np.sqrt(
            np.mean(
                (truth - estimate) ** 2
            )
        )
    )


def spread(samples):
    """
    Compute ensemble spread.

    Parameters
    ----------
    samples : ndarray, shape (n_ensemble, n_points)
        Ensemble samples.
    """

    samples = np.asarray(
        samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2D, got {samples.shape}"
        )

    if samples.shape[0] < 2:
        raise ValueError(
            "At least two ensemble members are required."
        )

    return float(
        np.sqrt(
            np.mean(
                np.var(
                    samples,
                    axis=0,
                    ddof=1,
                )
            )
        )
    )


# ============================================================
# Data readers
# ============================================================

def read_vtr(path):
    """Read density, velocity, and pressure from a VTR file."""

    if not path.is_file():
        return None

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


def sample_path(
    results_dir,
    k,
    da_interval,
):
    """
    Return the ensemble-sample path for one evaluation index.

    k = -1:
        forecast at the first DA time
        solution_0000_DA.vtr

    k >= 0:
        analysis after DA cycle k+1
        solution_(k+1)_0000.vtr
    """

    if k == -1:
        return (
            results_dir
            / f"solution_0000_{da_interval:04d}.vtr"
        )

    return (
        results_dir
        / f"solution_{k + 1:04d}_0000.vtr"
    )


def truth_path(
    truth_dir,
    k,
    da_interval,
):
    """Return the corresponding truth solution path."""

    if k == -1:
        truth_index = da_interval
    else:
        truth_index = (
            (k + 1)
            * da_interval
        )

    return (
        truth_dir
        / f"solution_{truth_index:04d}.vtr"
    )


# ============================================================
# Metric computation
# ============================================================

def compute_metrics(
    ensemble_results,
    truth_dir,
    k_values,
    da_interval,
):

    records = []

    for plot_index, k in enumerate(
        k_values
    ):

        # ----------------------------------------------------
        # Truth
        # ----------------------------------------------------

        truth_file = truth_path(
            truth_dir,
            k,
            da_interval,
        )

        truth = read_vtr(
            truth_file
        )

        if truth is None:
            raise FileNotFoundError(
                f"Truth solution not found: {truth_file}"
            )

        (
            x_truth,
            rho_truth_fine,
            u_truth_fine,
            p_truth_fine,
        ) = truth

        # ----------------------------------------------------
        # Ensemble
        # ----------------------------------------------------

        samples = []

        for results_dir in ensemble_results:

            path = sample_path(
                results_dir,
                k,
                da_interval,
            )

            sample = read_vtr(
                path
            )

            if sample is not None:
                samples.append(
                    sample
                )

        if not samples:
            raise RuntimeError(
                f"No ensemble samples found for k={k}."
            )

        x_samples = samples[0][0]

        rho_samples = np.asarray([
            sample[1]
            for sample in samples
        ])

        u_samples = np.asarray([
            sample[2]
            for sample in samples
        ])

        p_samples = np.asarray([
            sample[3]
            for sample in samples
        ])

        # ----------------------------------------------------
        # Interpolate truth onto the ensemble grid
        # ----------------------------------------------------

        rho_truth = interpolate_to_grid(
            x_truth,
            rho_truth_fine,
            x_samples,
        )

        u_truth = interpolate_to_grid(
            x_truth,
            u_truth_fine,
            x_samples,
        )

        p_truth = interpolate_to_grid(
            x_truth,
            p_truth_fine,
            x_samples,
        )

        # ----------------------------------------------------
        # Ensemble mean
        # ----------------------------------------------------

        rho_mean = np.mean(
            rho_samples,
            axis=0,
        )

        u_mean = np.mean(
            u_samples,
            axis=0,
        )

        p_mean = np.mean(
            p_samples,
            axis=0,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        records.append({
            "k": int(k),
            "step_plot": int(plot_index),
            "n_ensemble": int(
                rho_samples.shape[0]
            ),
            "rmse_rho": rmse(
                rho_truth,
                rho_mean,
            ),
            "rmse_u": rmse(
                u_truth,
                u_mean,
            ),
            "rmse_p": rmse(
                p_truth,
                p_mean,
            ),
            "spread_rho": spread(
                rho_samples
            ),
            "spread_u": spread(
                u_samples
            ),
            "spread_p": spread(
                p_samples
            ),
        })

    return pd.DataFrame.from_records(
        records
    )


# ============================================================
# Plot
# ============================================================

def plot_metrics(
    dataframe,
    k_values,
    output_file,
    dpi,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 2.5),
        sharey=False,
    )

    variables = [
        "rho",
        "u",
        "p",
    ]

    ylabels = [
        r"$\rho$",
        r"$u$",
        r"$p$",
    ]

    x_values = dataframe[
        "step_plot"
    ].to_numpy()

    xticks = np.arange(
        len(k_values)
    )

    for j, variable in enumerate(
        variables
    ):

        ax = axes[j]

        # ----------------------------------------------------
        # RMSE
        # ----------------------------------------------------

        ax.plot(
            x_values,
            dataframe[
                f"rmse_{variable}"
            ].to_numpy(),
            linestyle=LINE_STYLE,
            marker=MARKER,
            markersize=MARK_SIZE,
            markerfacecolor="none",
            markeredgecolor=C_RMSE,
            markeredgewidth=EDGE_WIDTH,
            linewidth=LINE_WIDTH,
            color=C_RMSE,
            label="RMSE",
            zorder=3,
            clip_on=False,
        )

        # ----------------------------------------------------
        # Spread
        # ----------------------------------------------------

        ax.plot(
            x_values,
            dataframe[
                f"spread_{variable}"
            ].to_numpy(),
            linestyle=LINE_STYLE,
            marker=MARKER,
            markersize=MARK_SIZE,
            markerfacecolor="none",
            markeredgecolor=C_SPREAD,
            markeredgewidth=EDGE_WIDTH,
            linewidth=LINE_WIDTH,
            color=C_SPREAD,
            label="Spread",
            zorder=3,
            clip_on=False,
        )

        # ----------------------------------------------------
        # Axes
        # ----------------------------------------------------

        ax.set_xlabel(
            "DA cycle"
        )

        ax.set_ylabel(
            ylabels[j]
        )

        ax.set_xticks(
            xticks
        )

        ax.set_xlim(
            xticks[0] - 0.05,
            xticks[-1] + 0.05,
        )

        ax.set_ylim(
            *Y_LIM[variable]
        )

        ax.set_yticks(
            Y_TICKS[variable]
        )

        ax.tick_params(
            axis="both",
            direction="in",
            top=True,
            right=True,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_zorder(1)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    axes[2].legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        loc="upper right",
    )

    # --------------------------------------------------------
    # Layout and save
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.20,
        top=0.95,
        wspace=0.34,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
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

    args = parse_args()

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

    # --------------------------------------------------------
    # Input directories
    # --------------------------------------------------------

    truth_dir = (
        BASE
        / f"{case_name}.truth"
        / "results"
    )

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"Truth directory not found: {truth_dir}"
        )

    ensemble_results = sorted(
        BASE.glob(
            f"{case_name}_[0-9][0-9][0-9]/results"
        )
    )

    if not ensemble_results:
        raise RuntimeError(
            "No ensemble result directories found."
        )

    # --------------------------------------------------------
    # Evaluation points
    # --------------------------------------------------------

    k_values = list(
        range(
            args.k_min,
            args.k_max + 1,
        )
    )

    # --------------------------------------------------------
    # Compute RMSE and spread
    # --------------------------------------------------------

    dataframe = compute_metrics(
        ensemble_results=ensemble_results,
        truth_dir=truth_dir,
        k_values=k_values,
        da_interval=da_interval,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    figure_file = (
        FIGURE_DIR
        / f"{case_name}_rmse_spread.png"
    )

    plot_metrics(
        dataframe=dataframe,
        k_values=k_values,
        output_file=figure_file,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()