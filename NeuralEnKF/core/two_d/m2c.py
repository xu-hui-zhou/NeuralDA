from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pyvista as pv


PathLike = Union[str, Path]


# ============================================================
# Shell utilities
# ============================================================

def _run(
    cmd: Iterable[str],
    *,
    cwd: PathLike | None = None,
) -> None:
    """Run a shell command and raise an error if it fails."""

    cmd = list(cmd)

    if cwd is not None:
        cwd = Path(cwd)

    print(
        f"[run] ({cwd or Path.cwd()})$ "
        f"{' '.join(cmd)}"
    )

    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
    )


def run_allclean(base_dir: PathLike) -> None:
    """Run ``Allclean`` in the specified directory if it exists."""

    base_dir = Path(base_dir)
    script = base_dir / "Allclean"

    if not script.exists():
        print("[Allclean] Not found; skipping.")
        return

    if not os.access(script, os.X_OK):
        os.chmod(script, 0o755)

    _run(
        ["./Allclean"],
        cwd=base_dir,
    )


# ============================================================
# M2C restart utilities
# ============================================================

def update_userdefinedstate(
    out_path: PathLike,
    case_dir: PathLike = ".",
    cpp_relpath: str = "IC/UserDefinedState.cpp",
) -> None:
    """Update the restart-state filename in ``UserDefinedState.cpp``."""

    case_dir = Path(case_dir)
    cpp_file = case_dir / cpp_relpath

    if not cpp_file.is_file():
        raise FileNotFoundError(
            f"UserDefinedState.cpp not found: {cpp_file}"
        )

    relative_path = f"IC/{Path(out_path).name}"

    lines = cpp_file.read_text().splitlines(
        keepends=True
    )

    new_lines = []
    replaced = False

    for line in lines:

        if (
            "std::string filename" in line
            and not replaced
        ):
            indent = line[
                : len(line) - len(line.lstrip())
            ]

            new_lines.append(
                f'{indent}std::string filename = '
                f'"{relative_path}";\n'
            )

            replaced = True

        else:
            new_lines.append(line)

    if not replaced:
        raise RuntimeError(
            "Target line 'std::string filename = ...' "
            "not found."
        )

    cpp_file.write_text(
        "".join(new_lines),
        encoding="utf-8",
    )

    print(
        f"[update_userdefinedstate] "
        f"{cpp_file} <- {relative_path}"
    )


def build_case(
    case_dir: PathLike = ".",
    ic_subdir: str = "IC",
    *,
    force_cmake: bool = False,
) -> None:
    """Build the user-defined restart state in the ``IC`` directory."""

    build_dir = Path(case_dir) / ic_subdir

    if not build_dir.is_dir():
        raise FileNotFoundError(
            f"Build directory not found: {build_dir}"
        )

    cache = build_dir / "CMakeCache.txt"

    if force_cmake or not cache.exists():
        _run(
            ["cmake", "."],
            cwd=build_dir,
        )
    else:
        print(
            "[build_case] CMakeCache.txt found; "
            "skipping cmake."
        )

    _run(
        ["make"],
        cwd=build_dir,
    )

    print(
        f"[build_case] OK: {build_dir}"
    )


def update_input_for_forecast(
    case_dir: PathLike,
    t_idx: int,
    filename: str = "input.st",
) -> None:
    """Update the solution filename in ``input.st`` for a forecast cycle."""

    case_dir = Path(case_dir)
    input_file = case_dir / filename

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    tag = f"{t_idx:04d}"

    lines = input_file.read_text().splitlines()
    new_lines = []

    for line in lines:

        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]

        if stripped.startswith("Solution"):
            new_lines.append(
                f'{indent}Solution = "solution_{tag}";'
            )

        else:
            new_lines.append(line)

    input_file.write_text(
        "\n".join(new_lines) + "\n"
    )

    print(
        f"[update_input_for_forecast] "
        f"Updated {input_file.name} for DA step {t_idx}"
    )


# ============================================================
# Observation utilities
# ============================================================

def nearest_indices(
    points_xy: np.ndarray,
    obs_xy: np.ndarray,
) -> np.ndarray:
    """
    Return grid-point indices associated with 2D observations.

    For each observation, the selected point is the last grid point
    whose x and y coordinates do not exceed the observation
    coordinates. Thus, the lower-left grid point is selected rather
    than the geometrically nearest point.

    The grid is assumed to be a uniform Cartesian grid with x-fast
    flattening.
    """

    points_xy = np.asarray(
        points_xy,
        dtype=float,
    )

    obs_xy = np.atleast_2d(
        np.asarray(
            obs_xy,
            dtype=float,
        )
    )

    if (
        points_xy.ndim != 2
        or points_xy.shape[1] != 2
    ):
        raise ValueError(
            "points_xy must have shape (N, 2)."
        )

    if (
        obs_xy.ndim != 2
        or obs_xy.shape[1] != 2
    ):
        raise ValueError(
            "obs_xy must have shape (M, 2)."
        )

    x_axis = np.unique(
        points_xy[:, 0]
    )

    y_axis = np.unique(
        points_xy[:, 1]
    )

    nx = x_axis.size
    ny = y_axis.size

    if nx * ny != points_xy.shape[0]:
        raise ValueError(
            "points_xy does not form a complete "
            "Cartesian grid."
        )

    tx = np.clip(
        obs_xy[:, 0],
        x_axis[0],
        x_axis[-1],
    )

    ty = np.clip(
        obs_xy[:, 1],
        y_axis[0],
        y_axis[-1],
    )

    ix = (
        np.searchsorted(
            x_axis,
            tx,
            side="right",
        )
        - 1
    )

    iy = (
        np.searchsorted(
            y_axis,
            ty,
            side="right",
        )
        - 1
    )

    ix = np.clip(
        ix,
        0,
        nx - 1,
    )

    iy = np.clip(
        iy,
        0,
        ny - 1,
    )

    return ix + nx * iy


# ============================================================
# M2C solution I/O
# ============================================================

def load_solution(
    vtr_file: PathLike,
):
    """
    Load a 2D M2C solution.

    Returns
    -------
    solution : dict
        Dictionary containing ``x``, ``y``, ``rho``, ``u``,
        ``v``, and ``p``.
    """

    vtr_file = Path(vtr_file)

    if not vtr_file.is_file():
        raise FileNotFoundError(
            f"VTR file not found: {vtr_file}"
        )

    grid = pv.read(
        str(vtr_file)
    )

    try:
        x = np.asarray(
            grid.points[:, 0]
        )

        y = np.asarray(
            grid.points[:, 1]
        )

        rho = np.asarray(
            grid.point_data["density"]
        )

        velocity = np.asarray(
            grid.point_data["velocity"]
        )

        u = velocity[:, 0]
        v = velocity[:, 1]

        p = np.asarray(
            grid.point_data["pressure"]
        )

    except KeyError as error:
        raise KeyError(
            f"Missing array in {vtr_file}. "
            f"Available point data: "
            f"{list(grid.point_data.keys())}"
        ) from error

    return {
        "x": x,
        "y": y,
        "rho": rho,
        "u": u,
        "v": v,
        "p": p,
    }


def write_vtk_analysis(
    *,
    template_vtr: PathLike,
    x_full: np.ndarray,
    y_full: np.ndarray,
    rho_full: np.ndarray,
    u_full: np.ndarray,
    v_full: np.ndarray,
    p_full: np.ndarray,
    out_vtr_path: PathLike,
):
    """
    Write an analyzed 2D state back to an M2C VTR file.

    The analyzed density, velocity, and pressure fields are written
    using the point ordering of the template VTR file.
    """

    template_vtr = Path(template_vtr)
    out_vtr_path = Path(out_vtr_path)

    if not template_vtr.is_file():
        raise FileNotFoundError(
            f"Template VTR not found: {template_vtr}"
        )

    x_full = np.asarray(x_full)
    y_full = np.asarray(y_full)
    rho_full = np.asarray(rho_full)
    u_full = np.asarray(u_full)
    v_full = np.asarray(v_full)
    p_full = np.asarray(p_full)

    if not (
        x_full.shape
        == y_full.shape
        == rho_full.shape
        == u_full.shape
        == v_full.shape
        == p_full.shape
    ):
        raise ValueError(
            "x_full, y_full, rho_full, u_full, "
            "v_full, and p_full must have the same shape."
        )

    grid = pv.read(
        str(template_vtr)
    )

    n_points = rho_full.size

    if grid.n_points != n_points:
        raise ValueError(
            f"Grid has {grid.n_points} points, "
            f"but analysis arrays contain {n_points} values."
        )

    if "density" not in grid.point_data:
        raise KeyError(
            f"'density' not found in {template_vtr}"
        )

    if "velocity" not in grid.point_data:
        raise KeyError(
            f"'velocity' not found in {template_vtr}"
        )

    if "pressure" not in grid.point_data:
        raise KeyError(
            f"'pressure' not found in {template_vtr}"
        )

    grid.point_data["density"] = np.asarray(
        rho_full,
        dtype=np.float64,
    )

    velocity = np.zeros(
        (n_points, 3),
        dtype=np.float64,
    )

    velocity[:, 0] = np.asarray(
        u_full,
        dtype=np.float64,
    )

    velocity[:, 1] = np.asarray(
        v_full,
        dtype=np.float64,
    )

    grid.point_data["velocity"] = velocity

    grid.point_data["pressure"] = np.asarray(
        p_full,
        dtype=np.float64,
    )

    out_vtr_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid.save(
        str(out_vtr_path)
    )

    return out_vtr_path