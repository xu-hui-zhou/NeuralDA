#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
# Y-axis ranges
# ============================================================

Y_LIM = {
    "rho": (-0.01, 0.20),
    "uv": (-0.05, 1.07),
    "p": (-1.0, 20.0),
}

Y_TICKS = {
    "rho": np.arange(
        0.0,
        0.2001,
        0.05,
    ),
    "uv": np.arange(
        0.0,
        1.1,
        0.2,
    ),
    "p": np.arange(
        0.0,
        20.0001,
        5.0,
    ),
}


# ============================================================
# Command-line options
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot RMSE and ensemble spread "
            "for the 2D Neural EnKF example."
        )
    )

    parser.add_argument(
        "--k-min",
        type=int,
        default=-1,
        help=(
            "First evaluation index. "
            "k=-1 denotes the first forecast."
        ),
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=4,
        help="Last analysis index.",
    )

    parser.add_argument(
        "--print-mapping",
        action="store_true",
        help=(
            "Print the mapping between evaluation indices "
            "and VTR filenames."
        ),
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


def stack_uv(
    u,
    v,
):
    """
    Combine the x- and y-velocity components into one state vector.
    """

    u = as_1d(
        u
    )

    v = as_1d(
        v
    )

    if u.shape != v.shape:
        raise ValueError(
            f"u/v shape mismatch: "
            f"{u.shape} vs {v.shape}"
        )

    return np.concatenate(
        [
            u,
            v,
        ],
        axis=0,
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
    samples : ndarray, shape (n_ensemble, n_dof)
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

def read_fields(path):
    """Read density, velocity, and pressure from a 2D VTR file."""

    path = Path(
        path
    )

    if not path.is_file():
        return None

    grid = pv.read(
        str(path)
    )

    rho = as_1d(
        grid.point_data["density"]
    )

    velocity = np.asarray(
        grid.point_data["velocity"]
    )

    u = as_1d(
        velocity[:, 0]
    )

    v = as_1d(
        velocity[:, 1]
    )

    p = as_1d(
        grid.point_data["pressure"]
    )

    return (
        rho,
        u,
        v,
        p,
    )


def truth_path(
    truth_dir,
    k,
    da_interval,
):
    """
    Return the truth solution path for one evaluation index.

    k = -1:
        truth at the first DA time.

    k >= 0:
        truth corresponding to the analysis after DA cycle k.
    """

    if k == -1:

        truth_index = (
            da_interval
        )

    else:

        truth_index = (
            (k + 1)
            * da_interval
        )

    return (
        truth_dir
        / f"solution_{truth_index:04d}.vtr"
    )


def sample_path(
    results_dir,
    k,
    da_interval,
):
    """
    Return the ensemble-member path for one evaluation index.

    k = -1:
        forecast at the first DA time:
        solution_0000_DA.vtr

    k >= 0:
        analysis after DA cycle k:
        solution_(k+1)_0000.vtr
    """

    if k == -1:

        return (
            results_dir
            / (
                f"solution_0000_"
                f"{da_interval:04d}.vtr"
            )
        )

    return (
        results_dir
        / f"solution_{k + 1:04d}_0000.vtr"
    )


# ============================================================
# Metric computation
# ============================================================

def compute_metrics(
    ensemble_results,
    mean_results,
    truth_dir,
    k_values,
    da_interval,
):

    metrics = {
        "rmse_rho": [],
        "rmse_uv": [],
        "rmse_p": [],
        "spread_rho": [],
        "spread_uv": [],
        "spread_p": [],
    }

    for k in k_values:

        # ----------------------------------------------------
        # Truth
        # ----------------------------------------------------

        truth_file = truth_path(
            truth_dir,
            k,
            da_interval,
        )

        truth = read_fields(
            truth_file
        )

        if truth is None:
            raise FileNotFoundError(
                f"Truth solution not found: {truth_file}"
            )

        (
            rho_truth,
            u_truth,
            v_truth,
            p_truth,
        ) = truth

        uv_truth = stack_uv(
            u_truth,
            v_truth,
        )

        # ----------------------------------------------------
        # Ensemble mean
        # ----------------------------------------------------

        mean_file = sample_path(
            mean_results,
            k,
            da_interval,
        )

        mean = read_fields(
            mean_file
        )

        if mean is None:
            raise FileNotFoundError(
                f"Mean solution not found: {mean_file}"
            )

        (
            rho_mean,
            u_mean,
            v_mean,
            p_mean,
        ) = mean

        uv_mean = stack_uv(
            u_mean,
            v_mean,
        )

        # ----------------------------------------------------
        # Ensemble members
        # ----------------------------------------------------

        rho_samples = []
        u_samples = []
        v_samples = []
        p_samples = []

        for results_dir in ensemble_results:

            path = sample_path(
                results_dir,
                k,
                da_interval,
            )

            sample = read_fields(
                path
            )

            if sample is None:
                continue

            rho, u, v, p = sample

            rho_samples.append(
                rho
            )

            u_samples.append(
                u
            )

            v_samples.append(
                v
            )

            p_samples.append(
                p
            )

        if not rho_samples:
            raise RuntimeError(
                f"No ensemble samples found for k={k}."
            )

        rho_samples = np.stack(
            rho_samples,
            axis=0,
        )

        u_samples = np.stack(
            u_samples,
            axis=0,
        )

        v_samples = np.stack(
            v_samples,
            axis=0,
        )

        p_samples = np.stack(
            p_samples,
            axis=0,
        )

        uv_samples = np.concatenate(
            [
                u_samples,
                v_samples,
            ],
            axis=1,
        )

        # ----------------------------------------------------
        # RMSE and spread
        # ----------------------------------------------------

        metrics["rmse_rho"].append(
            rmse(
                rho_truth,
                rho_mean,
            )
        )

        metrics["rmse_uv"].append(
            rmse(
                uv_truth,
                uv_mean,
            )
        )

        metrics["rmse_p"].append(
            rmse(
                p_truth,
                p_mean,
            )
        )

        metrics["spread_rho"].append(
            spread(
                rho_samples
            )
        )

        metrics["spread_uv"].append(
            spread(
                uv_samples
            )
        )

        metrics["spread_p"].append(
            spread(
                p_samples
            )
        )

    for key in metrics:
        metrics[key] = np.asarray(
            metrics[key],
            dtype=float,
        )

    return metrics


# ============================================================
# Plot
# ============================================================

def plot_metrics(
    metrics,
    k_values,
    output_file,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.0, 2.5),
        sharey=False,
    )

    variables = [
        "rho",
        "uv",
        "p",
    ]

    ylabels = [
        r"$\rho$",
        r"$[u,v]$",
        r"$p$",
    ]

    x_values = np.arange(
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
            metrics[
                f"rmse_{variable}"
            ],
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
            metrics[
                f"spread_{variable}"
            ],
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
            x_values
        )

        ax.set_xlim(
            x_values[0] - 0.05,
            x_values[-1] + 0.05,
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

            spine.set_linewidth(
                1.0
            )

            spine.set_zorder(
                1
            )

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
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    # Optional PDF output.
    # output_pdf = output_file.with_suffix(".pdf")
    #
    # fig.savefig(
    #     output_pdf,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )

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
    # 1. Load configuration
    # --------------------------------------------------------

    with CONFIG_FILE.open("r") as file:

        config = yaml.safe_load(
            file
        )

    case_name = (
        config["case"]["name"]
    )

    n_ensemble = (
        config["initial_ensemble"]["size"]
    )

    da_interval = (
        config["assimilation"]["interval"]
    )

    # --------------------------------------------------------
    # 2. Input directories
    # --------------------------------------------------------

    truth_dir = (
        BASE
        / f"{case_name}.truth"
        / "results"
    )

    mean_results = (
        BASE
        / f"{case_name}.mean"
        / "results"
    )

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"Truth directory not found: {truth_dir}"
        )

    if not mean_results.is_dir():
        raise FileNotFoundError(
            f"Mean directory not found: {mean_results}"
        )

    ensemble_results = []

    for i in range(
        n_ensemble
    ):

        results_dir = (
            BASE
            / f"{case_name}_{i:03d}"
            / "results"
        )

        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"Ensemble results directory not found: "
                f"{results_dir}"
            )

        ensemble_results.append(
            results_dir
        )

    # --------------------------------------------------------
    # 3. Evaluation points
    # --------------------------------------------------------

    k_values = list(
        range(
            args.k_min,
            args.k_max + 1,
        )
    )

    # --------------------------------------------------------
    # 4. Print file mapping if requested
    # --------------------------------------------------------

    if args.print_mapping:

        print(
            "---- File mapping "
            "(k -> truth / sample / mean) ----"
        )

        for k in k_values:

            truth_name = truth_path(
                truth_dir,
                k,
                da_interval,
            ).name

            sample_name = sample_path(
                ensemble_results[0],
                k,
                da_interval,
            ).name

            mean_name = sample_path(
                mean_results,
                k,
                da_interval,
            ).name

            print(
                f"k={k:>2d}: "
                f"truth={truth_name:>18s} | "
                f"sample={sample_name:>22s} | "
                f"mean={mean_name:>22s}"
            )

        print(
            "----------------------------------------------"
        )

    # --------------------------------------------------------
    # 5. Compute RMSE and spread
    # --------------------------------------------------------

    metrics = compute_metrics(
        ensemble_results=ensemble_results,
        mean_results=mean_results,
        truth_dir=truth_dir,
        k_values=k_values,
        da_interval=da_interval,
    )

    # --------------------------------------------------------
    # 6. Plot
    # --------------------------------------------------------

    figure_file = (
        FIGURE_DIR
        / f"{case_name}_rmse_spread.png"
    )

    plot_metrics(
        metrics=metrics,
        k_values=k_values,
        output_file=figure_file,
    )


if __name__ == "__main__":
    main()