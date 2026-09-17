#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pyvista as pv
import yaml


# ============================================================
# Paths
# ============================================================

POSTPROCESS_DIR = Path(__file__).resolve().parent
BASE = POSTPROCESS_DIR.parent

CONFIG_FILE = BASE / "input.yaml"


# ============================================================
# Configuration
# ============================================================

FLOW_ARRAYS = [
    "density",
    "velocity",
    "pressure",
]


# ============================================================
# Ensemble utilities
# ============================================================

def get_case_dirs(
    base_dir,
    case_name,
    n_ensemble,
):
    """
    Return the expected ensemble-member directories.

    For example, for ``case_name='bw'`` and ``n_ensemble=30``:

        bw_000, bw_001, ..., bw_029.
    """

    case_dirs = []

    for i in range(n_ensemble):

        case_dir = (
            Path(base_dir)
            / f"{case_name}_{i:03d}"
        )

        if not case_dir.is_dir():
            raise FileNotFoundError(
                f"Ensemble-member directory not found: {case_dir}"
            )

        case_dirs.append(
            case_dir
        )

    return case_dirs


def collect_result_filenames(
    case_dirs,
):
    """
    Collect the union of VTR filenames available across ensemble members.
    """

    filenames = set()

    for case_dir in case_dirs:

        results_dir = (
            case_dir
            / "results"
        )

        if not results_dir.is_dir():
            continue

        for path in results_dir.iterdir():

            if (
                path.is_file()
                and path.suffix.lower() == ".vtr"
            ):
                filenames.add(
                    path.name
                )

    return sorted(
        filenames
    )


# ============================================================
# Mesh utilities
# ============================================================

def check_mesh_compatibility(
    reference,
    dataset,
):
    """
    Check whether two VTR datasets can be averaged pointwise.
    """

    if (
        reference.number_of_points
        != dataset.number_of_points
    ):
        raise ValueError(
            "Point-count mismatch: "
            f"{reference.number_of_points} vs "
            f"{dataset.number_of_points}."
        )

    if not np.allclose(
        reference.points,
        dataset.points,
    ):
        raise ValueError(
            "Point coordinates differ between ensemble members."
        )


# ============================================================
# Ensemble averaging
# ============================================================

def average_flow_arrays(
    datasets,
):
    """
    Compute ensemble means of the M2C flow variables.

    The averaged quantities are

        density,
        velocity,
        pressure.

    The velocity array is averaged component-wise, giving the
    ensemble means of both the x- and y-velocity components.
    """

    if not datasets:
        raise ValueError(
            "No datasets were provided."
        )

    mean_arrays = {}

    for name in FLOW_ARRAYS:

        for dataset in datasets:

            if name not in dataset.point_data:
                raise KeyError(
                    f"Required array '{name}' is missing."
                )

        arrays = [
            np.asarray(
                dataset.point_data[name]
            )
            for dataset in datasets
        ]

        reference_shape = (
            arrays[0].shape
        )

        if not all(
            array.shape == reference_shape
            for array in arrays
        ):
            raise ValueError(
                f"Inconsistent shapes for array '{name}'."
            )

        mean_arrays[name] = (
            np.stack(
                arrays,
                axis=0,
            )
            .mean(axis=0)
        )

    return mean_arrays


def write_mean_dataset(
    template_dataset,
    mean_arrays,
    output_path,
):
    """
    Write ensemble-mean flow variables using a template VTR mesh.
    """

    dataset = (
        template_dataset.copy(
            deep=True
        )
    )

    # Remove all original point-data arrays.
    for name in list(
        dataset.point_data.keys()
    ):
        del dataset.point_data[name]

    # Write only the ensemble-mean flow variables.
    for name, array in mean_arrays.items():

        dataset.point_data[name] = (
            array
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save(
        str(output_path)
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------------

    with CONFIG_FILE.open("r") as f:
        config = yaml.safe_load(f)

    case_name = (
        config["case"]["name"]
    )

    n_ensemble = (
        config["initial_ensemble"]["size"]
    )

    # --------------------------------------------------------
    # 2. Locate ensemble members
    # --------------------------------------------------------

    case_dirs = get_case_dirs(
        base_dir=BASE,
        case_name=case_name,
        n_ensemble=n_ensemble,
    )

    mean_dir = (
        BASE
        / f"{case_name}.mean"
    )

    mean_results_dir = (
        mean_dir
        / "results"
    )

    mean_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Found {len(case_dirs)} ensemble members."
    )

    # --------------------------------------------------------
    # 3. Collect available VTR files
    # --------------------------------------------------------

    filenames = collect_result_filenames(
        case_dirs
    )

    if not filenames:
        print(
            "No VTR files found in the ensemble results directories."
        )
        return

    # --------------------------------------------------------
    # 4. Compute ensemble means
    # --------------------------------------------------------

    summary = []

    for filename in filenames:

        datasets = []

        for case_dir in case_dirs:

            vtr_path = (
                case_dir
                / "results"
                / filename
            )

            if not vtr_path.is_file():
                continue

            dataset = pv.read(
                str(vtr_path)
            )

            datasets.append(
                dataset
            )

        if not datasets:
            continue

        reference = datasets[0]

        try:

            for dataset in datasets[1:]:

                check_mesh_compatibility(
                    reference,
                    dataset,
                )

        except ValueError as error:

            print(
                f"[skip] {filename}: {error}"
            )

            continue

        try:

            mean_arrays = average_flow_arrays(
                datasets
            )

        except (KeyError, ValueError) as error:

            print(
                f"[skip] {filename}: {error}"
            )

            continue

        output_path = (
            mean_results_dir
            / filename
        )

        write_mean_dataset(
            template_dataset=reference,
            mean_arrays=mean_arrays,
            output_path=output_path,
        )

        n_used = len(
            datasets
        )

        print(
            f"[mean] {filename}: "
            f"{n_used}/{n_ensemble} members "
            f"-> {output_path}"
        )

        summary.append(
            (
                filename,
                n_used,
            )
        )

    # --------------------------------------------------------
    # 5. Write summary
    # --------------------------------------------------------

    if summary:

        summary_file = (
            mean_dir
            / "mean_summary.txt"
        )

        with summary_file.open(
            "w"
        ) as f:

            for filename, n_used in summary:

                f.write(
                    f"{filename}, "
                    f"{n_used} contributing members\n"
                )

        print(
            f"\nSummary written to {summary_file}"
        )

    else:

        print(
            "\nNo ensemble-mean VTR files were produced."
        )


if __name__ == "__main__":
    main()
