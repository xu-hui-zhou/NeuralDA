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

SOLUTION_FILE = Path("results/solution_0350.vtr")
OUTPUT_FILE = Path("figures/solution_0350.png")

FIGSIZE = (12.0, 3.2)
DPI = 300

LINE_COLOR = "k"
LINE_WIDTH = 2.5

X_LIM = (0.0, 1.0)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "lines.antialiased": True,
    "path.simplify": True,

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.labelsize": 14,
})


# ============================================================
# Load VTR solution
# ============================================================

def load_solution(path: Path):
    """Load density, x-velocity, and pressure from an M2C VTR file."""

    if not path.is_file():
        raise FileNotFoundError(f"Solution file not found: {path}")

    data = pv.read(str(path))

    x = np.asarray(data.points[:, 0])
    rho = np.asarray(data.point_data["density"])
    u = np.asarray(data.point_data["velocity"][:, 0])
    p = np.asarray(data.point_data["pressure"])

    # Ensure increasing x coordinate
    idx = np.argsort(x)

    return x[idx], rho[idx], u[idx], p[idx]


# ============================================================
# Main
# ============================================================

def main():

    x, rho, u, p = load_solution(SOLUTION_FILE)

    fields = [
        (rho, r"Density, $\rho$"),
        (u,   r"Velocity, $u$"),
        (p,   r"Pressure, $p$"),
    ]

    fig, axs = plt.subplots(
        1,
        3,
        figsize=FIGSIZE,
        sharex=True,
    )

    for ax, (field, ylabel) in zip(axs, fields):

        ax.plot(
            x,
            field,
            color=LINE_COLOR,
            lw=LINE_WIDTH,
        )

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)

        ax.set_xlim(*X_LIM)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.20,
        top=0.96,
        wspace=0.30,
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