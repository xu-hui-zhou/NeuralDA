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
# Text utilities
# ============================================================

def _format_value(value):
    """Format a scalar value for the M2C input file."""

    return f"{value:g}"


def _replace_value(pattern, text, value):
    """Replace the first value matched by a regular expression."""

    match = pattern.search(text)

    if match is None:
        raise ValueError(
            f"Expected pattern not found: {pattern.pattern}"
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


def _replace_block_value(
    block_pattern,
    field_pattern,
    text,
    value,
    label,
):
    """Replace one field inside a specified M2C input block."""

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
            f"Expected field not found in '{label}' block."
        )

    new_body = field_pattern.sub(
        lambda m: (
            f"{m.group(1)}"
            f"{_format_value(value)}"
            f"{m.group(2)}"
        ),
        body,
        count=1,
    )

    start, end = match.span("body")

    return (
        text[:start]
        + new_body
        + text[end:]
    )


# ============================================================
# Ensemble sampling
# ============================================================

def sample_ensemble(
    n,
    mean,
    std,
    x_bounds,
    seed=0,
):
    """
    Sample the initial ensemble.

    The sampled variables are ordered as

        [x_diaphragm, rho_left, p_left, rho_right, p_right].

    The diaphragm location is resampled until it lies inside the
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

    x_min, x_max = x_bounds

    if x_min >= x_max:
        raise ValueError(
            "x_diaphragm_bounds must satisfy min < max."
        )

    rng = np.random.default_rng(seed)

    samples = rng.normal(
        loc=mean,
        scale=std,
        size=(n, 5),
    )

    # Resample the diaphragm location instead of clipping it,
    # avoiding artificial probability mass at the bounds.
    for i in range(n):

        while not (
            x_min <= samples[i, 0] <= x_max
        ):
            samples[i, 0] = rng.normal(
                loc=mean[0],
                scale=std[0],
            )

    return samples


# ============================================================
# M2C input-file modifications
# ============================================================

def _update_initial_input(
    text,
    x_diaphragm,
    rho_left,
    p_left,
    rho_right,
    p_right,
):
    """
    Update ``input0.st`` for one ensemble member.
    """

    text = _replace_value(
        PAT_POINT_X,
        text,
        x_diaphragm,
    )

    text = _replace_block_value(
        PAT_INLET_BLOCK,
        PAT_DENSITY,
        text,
        rho_left,
        "Inlet",
    )

    text = _replace_block_value(
        PAT_INLET_BLOCK,
        PAT_PRESSURE,
        text,
        p_left,
        "Inlet",
    )

    text = _replace_block_value(
        PAT_INITIAL_BLOCK,
        PAT_DENSITY,
        text,
        rho_right,
        "InitialState",
    )

    text = _replace_block_value(
        PAT_INITIAL_BLOCK,
        PAT_PRESSURE,
        text,
        p_right,
        "InitialState",
    )

    text = _replace_block_value(
        PAT_INLET2_BLOCK,
        PAT_DENSITY,
        text,
        rho_right,
        "Inlet2",
    )

    text = _replace_block_value(
        PAT_INLET2_BLOCK,
        PAT_PRESSURE,
        text,
        p_right,
        "Inlet2",
    )

    return text


def _update_forecast_input(
    text,
    rho_left,
    p_left,
    rho_right,
    p_right,
):
    """
    Update boundary conditions in ``input.st`` for one member.
    """

    text = _replace_block_value(
        PAT_INLET_BLOCK,
        PAT_DENSITY,
        text,
        rho_left,
        "Inlet",
    )

    text = _replace_block_value(
        PAT_INLET_BLOCK,
        PAT_PRESSURE,
        text,
        p_left,
        "Inlet",
    )

    text = _replace_block_value(
        PAT_INLET2_BLOCK,
        PAT_DENSITY,
        text,
        rho_right,
        "Inlet2",
    )

    text = _replace_block_value(
        PAT_INLET2_BLOCK,
        PAT_PRESSURE,
        text,
        p_right,
        "Inlet2",
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
    x_bounds,
):
    """
    Generate the initial ensemble from an M2C template directory.

    No solver execution is performed. For a case named ``sod``,
    the generated directories are

        sod_000, sod_001, ..., sod_{n-1}.

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
        [x_diaphragm, rho_left, p_left, rho_right, p_right].

    std : sequence of float
        Standard deviations of the same variables.

    x_bounds : tuple of float
        Minimum and maximum allowed diaphragm locations.
    """

    base_dir = Path(base_dir).resolve()
    template_dir = Path(template_dir).resolve()

    if n <= 0:
        raise ValueError(
            "Ensemble size must be positive."
        )

    if not template_dir.is_dir():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}"
        )

    input0_template = template_dir / "input0.st"
    input_template = template_dir / "input.st"

    if not input0_template.is_file():
        raise FileNotFoundError(
            f"Missing input0.st: {input0_template}"
        )

    if not input_template.is_file():
        raise FileNotFoundError(
            f"Missing input.st: {input_template}"
        )

    samples = sample_ensemble(
        n=n,
        mean=mean,
        std=std,
        x_bounds=x_bounds,
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
            "x_diaphragm",
            "rho_left",
            "p_left",
            "rho_right",
            "p_right",
        ])

        for i, sample in enumerate(samples):

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
                x_diaphragm=float(x_diaphragm),
                rho_left=float(rho_left),
                p_left=float(p_left),
                rho_right=float(rho_right),
                p_right=float(p_right),
            )

            input0_path.write_text(
                input0_text
            )

            # ------------------------------------------------
            # Forecast/restart input
            # ------------------------------------------------

            input_path = (
                member_dir
                / "input.st"
            )

            input_text = (
                input_path.read_text()
            )

            input_text = _update_forecast_input(
                input_text,
                rho_left=float(rho_left),
                p_left=float(p_left),
                rho_right=float(rho_right),
                p_right=float(p_right),
            )

            input_path.write_text(
                input_text
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
        f"\nCreated {n} initial ensemble members."
    )

    print(
        f"Sample log: {csv_path}"
    )