#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv


# ============================================================
# Configuration
# ============================================================

SOLUTION_FILE = Path("results/solution_0125.vtr")
OUTPUT_FILE = Path("figures/solution_0125.png")

FIGSIZE = (10.0, 8.0)
DPI = 300

N_LEVELS = 100

RHO_LIM = (0.0, 5.0)
VEL_LIM = (-26.0, 26.0)
P_LIM = (0.0, 1000.0)

XY_LIM = (0.0, 2.0)
XY_TICKS = [0, 1, 2]


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": DPI,

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
})


# ============================================================
# Load VTR solution
# ============================================================

def load_solution(path: Path):
    """Load 2D flow fields from an M2C VTR file."""

    if not path.is_file():
        raise FileNotFoundError(f"Solution file not found: {path}")

    data = pv.read(str(path))

    x = np.asarray(data.points[:, 0])
    y = np.asarray(data.points[:, 1])

    rho = np.asarray(data.point_data["density"])

    velocity = np.asarray(data.point_data["velocity"])
    u = velocity[:, 0]
    v = velocity[:, 1]

    p = np.asarray(data.point_data["pressure"])

    return x, y, rho, u, v, p


# ============================================================
# Main
# ============================================================

def main():

    x, y, rho, u, v, p = load_solution(SOLUTION_FILE)

    fields = [
        (
            rho,
            r"Density, $\rho$",
            "viridis",
            RHO_LIM,
            [0.0, 2.5, 5.0],
        ),
        (
            u,
            r"Velocity, $u$",
            "coolwarm",
            VEL_LIM,
            [-25, 0, 25],
        ),
        (
            v,
            r"Velocity, $v$",
            "coolwarm",
            VEL_LIM,
            [-25, 0, 25],
        ),
        (
            p,
            r"Pressure, $p$",
            "viridis",
            P_LIM,
            [0, 500, 1000],
        ),
    ]

    fig, axs = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
    )

    for ax, (field, title, cmap, limits, cbar_ticks) in zip(
        axs.flat, fields
    ):

        vmin, vmax = limits
        levels = np.linspace(vmin, vmax, N_LEVELS + 1)

        contour = ax.tricontourf(
            x,
            y,
            field,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        ax.set_xlim(*XY_LIM)
        ax.set_ylim(*XY_LIM)

        ax.set_xticks(XY_TICKS)
        ax.set_yticks(XY_TICKS)

        ax.set_aspect("equal")

        cbar = fig.colorbar(
            contour,
            ax=ax,
            ticks=cbar_ticks,
            pad=0.03,
            fraction=0.046,
        )

        cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        bottom=0.08,
        top=0.95,
        wspace=0.28,
        hspace=0.25,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()