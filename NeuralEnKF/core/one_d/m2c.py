from __future__ import annotations

import os
import re
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
    """Update output filenames in ``input.st`` for a forecast cycle."""

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

        elif stripped.startswith("FileName"):
            new_lines.append(
                f'{indent}FileName = "line_{tag}";'
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
    x: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Return the nearest grid-point index for each target coordinate."""

    x = np.asarray(
        x,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    if not np.all(np.diff(x) >= 0):
        raise ValueError(
            "Grid coordinates must be in ascending order."
        )

    targets = np.clip(
        targets,
        x[0],
        x[-1],
    )

    indices = np.searchsorted(
        x,
        targets,
        side="left",
    )

    indices = np.clip(
        indices,
        0,
        len(x) - 1,
    )

    left = np.maximum(
        indices - 1,
        0,
    )

    use_left = (
        np.abs(x[left] - targets)
        <= np.abs(x[indices] - targets)
    )

    indices[use_left] = left[use_left]

    return indices


# ============================================================
# M2C solution I/O
# ============================================================

def load_solution(
    vtr_file: PathLike,
    input_st_file: PathLike,
):
    """
    Load a 1D M2C solution including boundary values.

    The interior state is read from the VTR file. Boundary values
    at x=0 and x=1 are read from the ``Inlet`` and ``Inlet2``
    blocks of ``input.st``.

    Returns
    -------
    solution : dict
        Dictionary containing ``x``, ``rho``, ``u``, and ``p``.
    """

    vtr_file = Path(vtr_file)
    input_st_file = Path(input_st_file)

    if not vtr_file.is_file():
        raise FileNotFoundError(
            f"VTR file not found: {vtr_file}"
        )

    if not input_st_file.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_st_file}"
        )

    # --------------------------------------------------------
    # Interior solution
    # --------------------------------------------------------

    grid = pv.read(str(vtr_file))

    try:
        x_inner = np.asarray(
            grid.points[:, 0]
        )

        rho_inner = np.asarray(
            grid.point_data["density"]
        )

        u_inner = np.asarray(
            grid.point_data["velocity"]
        )[:, 0]

        p_inner = np.asarray(
            grid.point_data["pressure"]
        )

    except KeyError as error:
        raise KeyError(
            f"Missing array in {vtr_file}. "
            f"Available point data: "
            f"{list(grid.point_data.keys())}"
        ) from error

    # --------------------------------------------------------
    # Boundary values
    # --------------------------------------------------------

    text = input_st_file.read_text()

    def get_block_value(
        label: str,
        key: str,
    ) -> float:

        block_pattern = re.compile(
            rf"under\s+{label}\s*\{{(.*?)\}}",
            re.S,
        )

        block = block_pattern.search(text)

        if block is None:
            raise ValueError(
                f"Block '{label}' not found in "
                f"{input_st_file}"
            )

        field_pattern = re.compile(
            rf"\b{key}\s*=\s*([-+0-9.eE]+)"
        )

        value = field_pattern.search(
            block.group(1)
        )

        if value is None:
            raise ValueError(
                f"Key '{key}' not found in "
                f"block '{label}'"
            )

        return float(
            value.group(1)
        )

    rho_left = get_block_value(
        "Inlet",
        "Density",
    )

    u_left = get_block_value(
        "Inlet",
        "VelocityX",
    )

    p_left = get_block_value(
        "Inlet",
        "Pressure",
    )

    rho_right = get_block_value(
        "Inlet2",
        "Density",
    )

    u_right = get_block_value(
        "Inlet2",
        "VelocityX",
    )

    p_right = get_block_value(
        "Inlet2",
        "Pressure",
    )

    # --------------------------------------------------------
    # Combine boundaries and interior
    # --------------------------------------------------------

    x = np.concatenate([
        [0.0],
        x_inner,
        [1.0],
    ])

    rho = np.concatenate([
        [rho_left],
        rho_inner,
        [rho_right],
    ])

    u = np.concatenate([
        [u_left],
        u_inner,
        [u_right],
    ])

    p = np.concatenate([
        [p_left],
        p_inner,
        [p_right],
    ])

    return {
        "x": x,
        "rho": rho,
        "u": u,
        "p": p,
    }


def write_vtk_analysis(
    *,
    template_vtr: PathLike,
    input_st_path: PathLike,
    rho_full: np.ndarray,
    u_full: np.ndarray,
    p_full: np.ndarray,
    out_vtr_path: PathLike,
):
    """
    Write an analyzed 1D state back to M2C.

    Interior values are written to the VTR file, while the left and
    right boundary values are written to ``input.st``.
    """

    template_vtr = Path(template_vtr)
    input_st_path = Path(input_st_path)
    out_vtr_path = Path(out_vtr_path)

    if not template_vtr.is_file():
        raise FileNotFoundError(
            f"Template VTR not found: {template_vtr}"
        )

    if not input_st_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_st_path}"
        )

    rho_full = np.asarray(rho_full)
    u_full = np.asarray(u_full)
    p_full = np.asarray(p_full)

    if not (
        rho_full.shape
        == u_full.shape
        == p_full.shape
    ):
        raise ValueError(
            "rho_full, u_full, and p_full must have the same shape."
        )

    # --------------------------------------------------------
    # Separate boundaries and interior
    # --------------------------------------------------------

    rho_left = float(rho_full[0])
    u_left = float(u_full[0])
    p_left = float(p_full[0])

    rho_right = float(rho_full[-1])
    u_right = float(u_full[-1])
    p_right = float(p_full[-1])

    rho_inner = np.asarray(
        rho_full[1:-1],
        dtype=np.float64,
    )

    u_inner = np.asarray(
        u_full[1:-1],
        dtype=np.float64,
    )

    p_inner = np.asarray(
        p_full[1:-1],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Write interior VTR state
    # --------------------------------------------------------

    grid = pv.read(str(template_vtr))

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

    n_inner = rho_inner.size

    if grid.point_data["density"].size != n_inner:
        raise ValueError(
            "Density size mismatch."
        )

    if grid.point_data["pressure"].size != n_inner:
        raise ValueError(
            "Pressure size mismatch."
        )

    if grid.point_data["velocity"].shape[0] != n_inner:
        raise ValueError(
            "Velocity size mismatch."
        )

    grid.point_data["density"] = rho_inner

    velocity = np.zeros(
        (n_inner, 3),
        dtype=np.float64,
    )

    velocity[:, 0] = u_inner

    grid.point_data["velocity"] = velocity
    grid.point_data["pressure"] = p_inner

    out_vtr_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid.save(
        str(out_vtr_path)
    )

    # --------------------------------------------------------
    # Update boundary values in input.st
    # --------------------------------------------------------

    text = input_st_path.read_text()

    def set_field(
        body: str,
        key: str,
        value: float,
    ) -> str:

        pattern = re.compile(
            rf"(\b{key}\s*=\s*)([-+0-9.eE]+)"
        )

        if pattern.search(body) is None:
            raise ValueError(
                f"Key '{key}' not found in boundary block."
            )

        return pattern.sub(
            lambda match:
            f"{match.group(1)}{value:.8e}",
            body,
            count=1,
        )

    def replace_block(
        full_text: str,
        label: str,
        updates: dict,
    ) -> str:

        block_pattern = re.compile(
            rf"(under\s+{label}\s*\{{)"
            rf"(.*?)"
            rf"(\}})",
            re.S,
        )

        match = block_pattern.search(
            full_text
        )

        if match is None:
            raise ValueError(
                f"Block '{label}' not found in "
                f"{input_st_path}"
            )

        start = match.group(1)
        body = match.group(2)
        end = match.group(3)

        for key, value in updates.items():
            body = set_field(
                body,
                key,
                float(value),
            )

        return (
            full_text[:match.start()]
            + start
            + body
            + end
            + full_text[match.end():]
        )

    text = replace_block(
        text,
        "Inlet",
        {
            "Density": rho_left,
            "VelocityX": u_left,
            "Pressure": p_left,
        },
    )

    text = replace_block(
        text,
        "Inlet2",
        {
            "Density": rho_right,
            "VelocityX": u_right,
            "Pressure": p_right,
        },
    )

    input_st_path.write_text(
        text
    )

    return out_vtr_path