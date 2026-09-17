#!/usr/bin/env python3

import csv
import re
import shutil
from pathlib import Path

import numpy as np


# ============================================================
# Regex patterns
# ============================================================

NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
WS = r"[ \t]*"

PAT_POINT_X = re.compile(
    rf"(Point_x{WS}={WS}){NUM}({WS};)",
    re.M,
)

PAT_INLET_BLOCK = re.compile(
    r"under\s+Inlet\s*\{(?P<body>.*?)}",
    re.S,
)

PAT_INLET2_BLOCK = re.compile(
    r"under\s+Inlet2\s*\{(?P<body>.*?)}",
    re.S,
)

PAT_INITIAL_BLOCK = re.compile(
    r"under\s+InitialState\s*\{(?P<body>.*?)}",
    re.S,
)

PAT_DENSITY = re.compile(
    rf"(\bDensity{WS}={WS}){NUM}({WS};)"
)

PAT_PRESSURE = re.compile(
    rf"(\bPressure{WS}={WS}){NUM}({WS};)"
)


# ============================================================
# Sampling
# ============================================================

def sample_ensemble(
    n,
    mean,
    std,
    x_bounds,
    seed,
):
    """
    Sample the initial ensemble parameters.

    Parameters are ordered as
    [x_diaphragm, rho_left, p_left, rho_right, p_right].

    The diaphragm location is resampled until it lies within
    the prescribed bounds.
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
            "mean and std must each contain five parameters."
        )

    if n <= 0:
        raise ValueError(
            "Ensemble size must be positive."
        )

    x_min, x_max = x_bounds

    if x_min >= x_max:
        raise ValueError(
            "x_diaphragm bounds must satisfy min < max."
        )

    rng = np.random.default_rng(
        seed
    )

    samples = rng.normal(
        loc=mean,
        scale=std,
        size=(n, 5),
    )

    # Resample only the diaphragm location until it lies
    # within the prescribed interval. This avoids artificial
    # probability mass at the bounds.
    for i in range(n):

        while not (
            x_min
            <= samples[i, 0]
            <= x_max
        ):
            samples[i, 0] = rng.normal(
                loc=mean[0],
                scale=std[0],
            )

    return samples


# ============================================================
# Text utilities
# ============================================================

def format_value(value):
    """Format a scalar for writing to an M2C input file."""

    return f"{value:g}"


def replace_one(
    pattern,
    text,
    value,
):
    """Replace the first value matched by a regex pattern."""

    if pattern.search(text) is None:
        raise ValueError(
            "Expected field not found."
        )

    return pattern.sub(
        lambda match:
        (
            f"{match.group(1)}"
            f"{format_value(value)}"
            f"{match.group(2)}"
        ),
        text,
        count=1,
    )


def replace_in_block(
    block_pattern,
    field_pattern,
    text,
    value,
    label,
):
    """Replace one field inside a uniquely identified block."""

    matches = list(
        block_pattern.finditer(text)
    )

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{label}' block, "
            f"found {len(matches)}."
        )

    match = matches[0]
    body = match.group("body")

    if field_pattern.search(body) is None:
        raise ValueError(
            f"Expected field not found in '{label}'."
        )

    new_body = field_pattern.sub(
        lambda field_match:
        (
            f"{field_match.group(1)}"
            f"{format_value(value)}"
            f"{field_match.group(2)}"
        ),
        body,
        count=1,
    )

    start, end = match.span(
        "body"
    )

    return (
        text[:start]
        + new_body
        + text[end:]
    )


# ============================================================
# M2C input modification
# ============================================================

def update_initial_input(
    text,
    x_diaphragm,
    rho_left,
    p_left,
    rho_right,
    p_right,
):
    """
    Update ``input0.st`` with one sampled initial condition.
    """

    # Diaphragm location
    text = replace_one(
        PAT_POINT_X,
        text,
        x_diaphragm,
    )

    # Left state
    text = replace_in_block(
        PAT_INLET_BLOCK,
        PAT_DENSITY,
        text,
        rho_left,
        "Inlet",
    )

    text = replace_in_block(
        PAT_INLET_BLOCK,
        PAT_PRESSURE,
        text,
        p_left,
        "Inlet",
    )

    # Right initial state
    text = replace_in_block(
        PAT_INITIAL_BLOCK,
        PAT_DENSITY,
        text,
        rho_right,
        "InitialState",
    )

    text = replace_in_block(
        PAT_INITIAL_BLOCK,
        PAT_PRESSURE,
        text,
        p_right,
        "InitialState",
    )

    # Right boundary
    text = replace_in_block(
        PAT_INLET2_BLOCK,
        PAT_DENSITY,
        text,
        rho_right,
        "Inlet2",
    )

    text = replace_in_block(
        PAT_INLET2_BLOCK,
        PAT_PRESSURE,
        text,
        p_right,
        "Inlet2",
    )

    return text


def update_boundary_conditions(
    text,
    rho_left,
    p_left,
    rho_right,
    p_right,
):
    """
    Update the left and right boundary conditions in ``input.st``.
    """

    # Left boundary
    text = replace_in_block(
        PAT_INLET_BLOCK,
        PAT_DENSITY,
        text,
        rho_left,
        "Inlet",
    )

    text = replace_in_block(
        PAT_INLET_BLOCK,
        PAT_PRESSURE,
        text,
        p_left,
        "Inlet",
    )

    # Right boundary
    text = replace_in_block(
        PAT_INLET2_BLOCK,
        PAT_DENSITY,
        text,
        rho_right,
        "Inlet2",
    )

    text = replace_in_block(
        PAT_INLET2_BLOCK,
        PAT_PRESSURE,
        text,
        p_right,
        "Inlet2",
    )

    return text


# ============================================================
# Initial ensemble generation
# ============================================================

def make_initial_ensemble(
    base_dir,
    template_dir,
    case_name,
    n,
    seed,
    mean,
    std,
    x_bounds,
):
    """
    Create the initial M2C ensemble directories.

    The template directory is copied once for each ensemble member.
    ``input0.st`` is updated with the sampled initial condition,
    while ``input.st`` is updated with the corresponding boundary
    conditions.

    The solver is not executed by this function.
    """

    base_dir = Path(
        base_dir
    ).resolve()

    template_dir = Path(
        template_dir
    ).resolve()

    if not template_dir.is_dir():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}"
        )

    input0_template = (
        template_dir
        / "input0.st"
    )

    input_template = (
        template_dir
        / "input.st"
    )

    if not input0_template.is_file():
        raise FileNotFoundError(
            f"Missing input0.st: {input0_template}"
        )

    if not input_template.is_file():
        raise FileNotFoundError(
            f"Missing input.st: {input_template}"
        )

    # --------------------------------------------------------
    # Sample ensemble parameters
    # --------------------------------------------------------

    samples = sample_ensemble(
        n=n,
        mean=mean,
        std=std,
        x_bounds=x_bounds,
        seed=seed,
    )

    # --------------------------------------------------------
    # Create ensemble-member directories
    # --------------------------------------------------------

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
            "run_id",
            "x_diaphragm",
            "rho_left",
            "p_left",
            "rho_right",
            "p_right",
        ])

        for i, sample in enumerate(
            samples
        ):

            (
                x_diaphragm,
                rho_left,
                p_left,
                rho_right,
                p_right,
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
                "copying template and updating initial state"
            )

            shutil.copytree(
                template_dir,
                member_dir,
                dirs_exist_ok=False,
            )

            # ------------------------------------------------
            # input0.st
            # ------------------------------------------------

            input0_path = (
                member_dir
                / "input0.st"
            )

            text = input0_path.read_text()

            text = update_initial_input(
                text=text,
                x_diaphragm=float(x_diaphragm),
                rho_left=float(rho_left),
                p_left=float(p_left),
                rho_right=float(rho_right),
                p_right=float(p_right),
            )

            input0_path.write_text(
                text
            )

            # ------------------------------------------------
            # input.st
            # ------------------------------------------------

            input_path = (
                member_dir
                / "input.st"
            )

            text = input_path.read_text()

            text = update_boundary_conditions(
                text=text,
                rho_left=float(rho_left),
                p_left=float(p_left),
                rho_right=float(rho_right),
                p_right=float(p_right),
            )

            input_path.write_text(
                text
            )

            # ------------------------------------------------
            # Record sampled parameters
            # ------------------------------------------------

            writer.writerow([
                member_name,
                x_diaphragm,
                rho_left,
                p_left,
                rho_right,
                p_right,
            ])

    print(
        f"\nCreated {n} ensemble members: "
        f"{case_name}_000 .. {case_name}_{n - 1:03d}"
    )

    print(
        f"Sample log: {csv_path}"
    )