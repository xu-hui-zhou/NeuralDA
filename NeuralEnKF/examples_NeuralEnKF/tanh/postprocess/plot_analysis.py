#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

POSTPROCESS_DIR = Path(__file__).resolve().parent
BASE = POSTPROCESS_DIR.parent

DA_RESULTS = BASE / "da_results"
FIGURE_DIR = BASE / "figures"


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


C_FORECAST = "#1b8a6b"
C_ANALYSIS = "darkorange"
C_TRUTH = "black"
C_OBS = "0.6"

SAMPLE_ALPHA = 0.66


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    truth_data = np.load(
        DA_RESULTS / "truth.npy"
    )

    forecast_data = np.load(
        DA_RESULTS / "forecast.npy"
    )

    analysis_data = np.load(
        DA_RESULTS / "analysis.npy"
    )

    observation_data = np.load(
        DA_RESULTS / "observations.npy"
    )

    # truth.npy:
    # [x, h_truth]
    x = truth_data[:, 0]
    h_truth = truth_data[:, 1]

    # forecast.npy:
    # [x, forecast_0, ..., forecast_Ne-1]
    x_forecast = forecast_data[:, 0]
    h_forecast = forecast_data[:, 1:]

    # analysis.npy:
    # [x, analysis_0, ..., analysis_Ne-1]
    x_analysis = analysis_data[:, 0]
    h_analysis = analysis_data[:, 1:]

    # observations.npy:
    # [x_obs, obs, sigma_obs]
    x_obs = observation_data[:, 0]
    obs = observation_data[:, 1]
    sigma_obs = observation_data[:, 2]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.5),
        sharex=True,
        sharey=True,
    )

    # --------------------------------------------------------
    # (a) Forecast ensemble
    # --------------------------------------------------------

    ax = axes[0]

    for i in range(
        h_forecast.shape[1]
    ):
        ax.plot(
            x_forecast,
            h_forecast[:, i],
            lw=0.8,
            alpha=SAMPLE_ALPHA,
            color=C_FORECAST,
            label="Forecast" if i == 0 else None,
        )

    ax.plot(
        x,
        h_truth,
        lw=2.0,
        color=C_TRUTH,
        label="Truth",
    )

    ax.legend(
        frameon=True,
    )

    ax.text(
        0.0,
        1.12,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
    )

    # --------------------------------------------------------
    # (b) Analysis ensemble
    # --------------------------------------------------------

    ax = axes[1]

    for i in range(
        h_analysis.shape[1]
    ):
        ax.plot(
            x_analysis,
            h_analysis[:, i],
            lw=0.8,
            alpha=SAMPLE_ALPHA,
            color=C_ANALYSIS,
            label="Analysis" if i == 0 else None,
        )

    ax.plot(
        x,
        h_truth,
        lw=2.0,
        color=C_TRUTH,
        label="Truth",
    )

    ax.errorbar(
        x_obs,
        obs,
        yerr=2.0 * sigma_obs,
        fmt="o",
        color=C_OBS,
        ecolor=C_OBS,
        capsize=3,
        elinewidth=1.0,
        markersize=4.5,
        label="Obs.",
    )

    ax.legend(
        frameon=True,
    )

    ax.text(
        0.0,
        1.12,
        "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
    )

    # --------------------------------------------------------
    # Shared formatting
    # --------------------------------------------------------

    for ax in axes:

        ax.set_xlim(
            -1.0,
            1.0,
        )

        ax.set_ylim(
            -0.08,
            1.5,
        )

        ax.set_xticks([
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ])

        ax.set_yticks([
            0.0,
            0.3,
            0.6,
            0.9,
            1.2,
            1.5,
        ])

        ax.set_xlabel(
            r"$x$"
        )

        ax.grid(
            True,
            alpha=0.25,
            linestyle="--",
        )

    axes[0].set_ylabel(
        r"$h(x)$"
    )

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    fig.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        FIGURE_DIR
        / "tanh_analysis.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":
    main()