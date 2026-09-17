#!/usr/bin/env python3

import csv
import re
import shutil
from pathlib import Path

import numpy as np


# ============================================================
# Regular expressions
# ============================================================

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
WS = r"[ \t]*"

PAT_CENTER_X = re.compile(
    rf"(\bCenter_x{WS}={WS}){NUM}({WS};)"
)

PAT_CENTER_Y = re.compile(
    rf"(\bCenter_y{WS}={WS}){NUM}({WS};)"
)

PAT_RADIUS = re.compile(
    rf"(\bRadius{WS}={WS}){NUM}({WS};)"
)

PAT_DENSITY = re.compile(
    rf"(\bDensity{WS}={WS}){NUM}({WS};)"
)

PAT_PRESSURE = re.compile(
    rf"(\bPressure{WS}={WS}){NUM}({WS};)"
)

PAT_SPHERE0_HEADER = re.compile(
    r"under\s+Sphere\s*\[\s*0\s*\]\s*",
    re.I,
)

PAT_INITIAL_STATE_HEADER = re.compile(
    r"under\s+InitialState\s*",
    re.I,
)


# ============================================================
# Text utilities
# ============================================================

def _format_value(value):
    """Format a scalar value for the M2C input file."""

    return f"{value:g}"


def _replace_value(
    pattern,
    text,
    value,
    label,
):
    """Replace the first value matched by a regular expression."""

    match = pattern.search(text)

    if match is None:
        raise ValueError(
            f"Expected field '{label}' not found."
        )

    return pattern.sub(
        lambda m: (
            f"{m.group(1)}"
            f"{_format_value(value)}"
            f"{m.group(2)}"
        ),
        text,
        count=1,
    )


def _find_block(
    text,
    header_pattern,
    label,
):
    """
    Find a brace-balanced M2C input block.

    Returns
    -------
    body_start : int
        Index of the first character inside the block.

    body_end : int
        Index of the closing brace.
    """

    match = header_pattern.search(text)

    if match is None:
        raise ValueError(
            f"Expected block '{label}' not found."
        )

    open_brace = text.find(
        "{",
        match.end(),
    )

    if open_brace < 0:
        raise ValueError(
            f"Opening brace not found for block '{label}'."
        )

    depth = 0

    for i in range(
        open_brace,
        len(text),
    ):

        if text[i] == "{":
            depth += 1

        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return (
                    open_brace + 1,
                    i,
                )

    raise ValueError(
        f"Matching closing brace not found for block '{label}'."
    )


# ============================================================
# Ensemble sampling
# ============================================================

def sample_ensemble(
    n,
    mean,
    std,
    center_bounds,
    radius_bounds,
    seed=42,
):
    """
    Sample the initial ensemble.

    The sampled variables are ordered as

        [center_x, center_y, radius, density, pressure].

    Sphere-center coordinates and radius are clipped to their
    prescribed bounds.
    """

    mean = np.asarray(
        mean,
        dtype=float,
    )

    std = np.asarray(
        std,
        dtype=float,
    )

    if mean.shape != (5,) or std.shape != (5,):
        raise ValueError(
            "mean and std must each contain five values."
        )

    center_min, center_max = center_bounds
    radius_min, radius_max = radius_bounds

    if center_min >= center_max:
        raise ValueError(
            "center_bounds must satisfy min < max."
        )

    if radius_min >= radius_max:
        raise ValueError(
            "radius_bounds must satisfy min < max."
        )

    rng = np.random.default_rng(
        seed
    )

    # Draw each variable separately to preserve the random-number
    # ordering used in the original blast-wave implementation.
    samples = np.column_stack([
        rng.normal(
            mean[0],
            std[0],
            size=n,
        ),
        rng.normal(
            mean[1],
            std[1],
            size=n,
        ),
        rng.normal(
            mean[2],
            std[2],
            size=n,
        ),
        rng.normal(
            mean[3],
            std[3],
            size=n,
        ),
        rng.normal(
            mean[4],
            std[4],
            size=n,
        ),
    ])

    # Clip geometric parameters to their prescribed bounds.
    samples[:, 0] = np.clip(
        samples[:, 0],
        center_min,
        center_max,
    )

    samples[:, 1] = np.clip(
        samples[:, 1],
        center_min,
        center_max,
    )

    samples[:, 2] = np.clip(
        samples[:, 2],
        radius_min,
        radius_max,
    )

    return samples


# ============================================================
# M2C input-file modifications
# ============================================================

def _update_initial_input(
    text,
    center_x,
    center_y,
    radius,
    density,
    pressure,
):
    """
    Update ``input0.st`` for one blast-wave ensemble member.

    The geometry of ``Sphere[0]`` and the density and pressure in
    its nested ``InitialState`` block are modified.
    """

    # --------------------------------------------------------
    # Locate Sphere[0]
    # --------------------------------------------------------

    sphere_start, sphere_end = _find_block(
        text,
        PAT_SPHERE0_HEADER,
        "Sphere[0]",
    )

    sphere_body = text[
        sphere_start:sphere_end
    ]

    # --------------------------------------------------------
    # Update sphere geometry
    # --------------------------------------------------------

    sphere_body = _replace_value(
        PAT_CENTER_X,
        sphere_body,
        center_x,
        "Center_x",
    )

    sphere_body = _replace_value(
        PAT_CENTER_Y,
        sphere_body,
        center_y,
        "Center_y",
    )

    sphere_body = _replace_value(
        PAT_RADIUS,
        sphere_body,
        radius,
        "Radius",
    )

    # --------------------------------------------------------
    # Locate InitialState inside the updated Sphere[0] block
    # --------------------------------------------------------

    initial_start, initial_end = _find_block(
        sphere_body,
        PAT_INITIAL_STATE_HEADER,
        "Sphere[0]/InitialState",
    )

    initial_body = sphere_body[
        initial_start:initial_end
    ]

    # --------------------------------------------------------
    # Update thermodynamic state
    # --------------------------------------------------------

    initial_body = _replace_value(
        PAT_DENSITY,
        initial_body,
        density,
        "Density",
    )

    initial_body = _replace_value(
        PAT_PRESSURE,
        initial_body,
        pressure,
        "Pressure",
    )

    # --------------------------------------------------------
    # Reassemble InitialState and Sphere[0]
    # --------------------------------------------------------

    sphere_body = (
        sphere_body[:initial_start]
        + initial_body
        + sphere_body[initial_end:]
    )

    text = (
        text[:sphere_start]
        + sphere_body
        + text[sphere_end:]
    )

    return text


# ============================================================
# Initial ensemble construction
# ============================================================

def make_initial_ensemble(
    base_dir,
    template_dir,
    case_name,
    n,
    seed,
    mean,
    std,
    center_bounds,
    radius_bounds,
):
    """
    Generate the initial blast-wave ensemble from an M2C template.

    No solver execution is performed. For a case named ``bw``,
    the generated directories are

        bw_000, bw_001, ..., bw_{n-1}.

    Parameters
    ----------
    base_dir : str or Path
        Directory in which ensemble-member folders are created.

    template_dir : str or Path
        M2C template directory.

    case_name : str
        Prefix used for ensemble-member directory names.

    n : int
        Ensemble size.

    seed : int
        Random seed.

    mean : sequence of float
        Means of
        [center_x, center_y, radius, density, pressure].

    std : sequence of float
        Standard deviations of the same variables.

    center_bounds : tuple of float
        Minimum and maximum allowed sphere-center coordinates.

    radius_bounds : tuple of float
        Minimum and maximum allowed sphere radii.
    """

    base_dir = Path(
        base_dir
    ).resolve()

    template_dir = Path(
        template_dir
    ).resolve()

    if n <= 0:
        raise ValueError(
            "Ensemble size must be positive."
        )

    if not template_dir.is_dir():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}"
        )

    input0_template = (
        template_dir
        / "input0.st"
    )

    if not input0_template.is_file():
        raise FileNotFoundError(
            f"Missing input0.st: {input0_template}"
        )

    samples = sample_ensemble(
        n=n,
        mean=mean,
        std=std,
        center_bounds=center_bounds,
        radius_bounds=radius_bounds,
        seed=seed,
    )

    csv_path = (
        base_dir
        / "ensemble_samples.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "member",
            "center_x",
            "center_y",
            "radius",
            "density",
            "pressure",
        ])

        for i, sample in enumerate(samples):

            (
                center_x,
                center_y,
                radius,
                density,
                pressure,
            ) = sample

            member_name = (
                f"{case_name}_{i:03d}"
            )

            member_dir = (
                base_dir
                / member_name
            )

            print(
                f"[{member_name}] "
                "Creating initial ensemble member"
            )

            shutil.copytree(
                template_dir,
                member_dir,
                dirs_exist_ok=False,
            )

            # ------------------------------------------------
            # Initial simulation input
            # ------------------------------------------------

            input0_path = (
                member_dir
                / "input0.st"
            )

            input0_text = (
                input0_path.read_text()
            )

            input0_text = _update_initial_input(
                input0_text,
                center_x=float(center_x),
                center_y=float(center_y),
                radius=float(radius),
                density=float(density),
                pressure=float(pressure),
            )

            input0_path.write_text(
                input0_text
            )

            # ------------------------------------------------
            # Record sampled parameters
            # ------------------------------------------------

            writer.writerow([
                member_name,
                center_x,
                center_y,
                radius,
                density,
                pressure,
            ])

    print(
        f"\nCreated {n} initial ensemble members."
    )

    print(
        f"Sample log: {csv_path}"
    )