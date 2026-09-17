#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml


# ============================================================
# Repository path
# ============================================================

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]

# Allow this script to be run directly from examples/bw/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from initial_ensemble import make_initial_ensemble
from core.two_d import assimilation, chain, m2c, neural


# ============================================================
# Paths
# ============================================================

CONFIG_FILE = BASE / "input.yaml"

DA_RESULTS = BASE / "da_results"
CHAINS_DIR = DA_RESULTS / "chains"
TRUTH_DIR = DA_RESULTS / "truth"
OBS_DIR = DA_RESULTS / "obs"
HX_DIR = DA_RESULTS / "Hx"


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------------

    with CONFIG_FILE.open("r") as f:
        config = yaml.safe_load(f)

    case_name = config["case"]["name"]

    initial_cfg = config["initial_ensemble"]
    assimilation_cfg = config["assimilation"]
    observation_cfg = config["observations"]
    network_cfg = config["network"]
    training_cfg = config["training"]
    physical_cfg = config["physical_constraints"]

    # Initial ensemble
    n_ensemble = initial_cfg["size"]
    seed = initial_cfg["seed"]

    mean = np.array([
        initial_cfg["mean"]["center_x"],
        initial_cfg["mean"]["center_y"],
        initial_cfg["mean"]["radius"],
        initial_cfg["mean"]["density"],
        initial_cfg["mean"]["pressure"],
    ], dtype=float)

    std = np.array([
        initial_cfg["std"]["center_x"],
        initial_cfg["std"]["center_y"],
        initial_cfg["std"]["radius"],
        initial_cfg["std"]["density"],
        initial_cfg["std"]["pressure"],
    ], dtype=float)

    center_bounds = (
        initial_cfg["center_bounds"]["min"],
        initial_cfg["center_bounds"]["max"],
    )

    radius_bounds = (
        initial_cfg["radius_bounds"]["min"],
        initial_cfg["radius_bounds"]["max"],
    )

    # Data assimilation
    num_cycles = assimilation_cfg["num_cycles"]
    da_interval = assimilation_cfg["interval"]

    # Observation locations
    obs_x = np.arange(
        observation_cfg["x_start"],
        observation_cfg["x_end"],
        observation_cfg["spacing"],
    )

    obs_y = np.arange(
        observation_cfg["y_start"],
        observation_cfg["y_end"],
        observation_cfg["spacing"],
    )

    x_grid, y_grid = np.meshgrid(
        obs_x,
        obs_y,
        indexing="xy",
    )

    obs_xy = np.column_stack([
        x_grid.ravel(),
        y_grid.ravel(),
    ])

    # Observation noise
    relative_noise = observation_cfg["relative_noise"]
    absolute_noise = observation_cfg["absolute_noise"]

    # Neural network
    hidden_dim = network_cfg["hidden_dim"]
    num_layers = network_cfg["num_layers"]

    # Neural training
    learning_rate = training_cfg["learning_rate"]
    first_learning_rate = training_cfg["first_learning_rate"]
    max_epochs = training_cfg["max_epochs"]
    print_every = training_cfg["print_every"]
    target_loss = training_cfg["target_loss"]

    # Physical constraints
    pressure_min = physical_cfg["pressure_min"]

    # --------------------------------------------------------
    # 2. Initialization
    # --------------------------------------------------------

    np.random.seed(seed)
    torch.manual_seed(seed)

    m2c.run_allclean(
        BASE
    )

    for directory in [
        TRUTH_DIR,
        OBS_DIR,
        HX_DIR,
        CHAINS_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Generate initial ensemble-member directories.
    make_initial_ensemble(
        base_dir=BASE,
        template_dir=BASE / f"{case_name}.template",
        case_name=case_name,
        n=n_ensemble,
        seed=seed,
        mean=mean,
        std=std,
        center_bounds=center_bounds,
        radius_bounds=radius_bounds,
    )

    # Propagate initial ensemble from t = 0 to the first DA step.
    for i in range(n_ensemble):

        member_dir = (
            BASE
            / f"{case_name}_{i:03d}"
        )

        subprocess.run(
            ["./Allrun", "input0.st"],
            cwd=member_dir,
            check=True,
        )

    # ========================================================
    # Data assimilation loop
    # ========================================================

    for t_idx in range(num_cycles):

        print(
            f"\n{'=' * 60}\n"
            f"DA STEP {t_idx}\n"
            f"{'=' * 60}"
        )

        # ----------------------------------------------------
        # 3. Forecast
        # ----------------------------------------------------

        if t_idx > 0:

            for i in range(n_ensemble):

                member_dir = (
                    BASE
                    / f"{case_name}_{i:03d}"
                )

                m2c.update_input_for_forecast(
                    case_dir=member_dir,
                    t_idx=t_idx,
                )

                subprocess.run(
                    ["./Allrun", "input.st"],
                    cwd=member_dir,
                    check=True,
                )

        # ----------------------------------------------------
        # 4. Generate physical observations
        # ----------------------------------------------------

        truth_vtr = (
            BASE
            / f"{case_name}.truth"
            / "results"
            / f"solution_{(t_idx + 1) * da_interval:04d}.vtr"
        )

        truth_solution = m2c.load_solution(
            truth_vtr
        )

        truth_xy = np.column_stack((
            truth_solution["x"],
            truth_solution["y"],
        ))

        # Use the truth grid to determine observation indices.
        # The same indices are used for all ensemble members.
        obs_indices = m2c.nearest_indices(
            truth_xy,
            obs_xy,
        )

        # Pressure observations
        y_truth = (
            truth_solution["p"][obs_indices]
        )

        sigma_obs = (
            relative_noise
            * np.abs(y_truth)
            + absolute_noise
        )

        noise = np.random.normal(
            0.0,
            sigma_obs,
        )

        y_obs = (
            y_truth
            + noise
        )

        # Preserve the original positivity treatment.
        y_obs = np.where(
            y_obs > 0.0,
            y_obs,
            y_truth - noise,
        )

        # ----------------------------------------------------
        # 5. Save truth and observations
        # ----------------------------------------------------

        np.savetxt(
            TRUTH_DIR
            / f"y_gt_{t_idx:04d}.txt",
            np.column_stack([
                obs_xy,
                y_truth,
            ]),
            header="x y pressure_truth",
            fmt="%.6e",
        )

        np.savetxt(
            OBS_DIR
            / f"y_obs_{t_idx:04d}.txt",
            np.column_stack([
                obs_xy,
                y_obs,
            ]),
            header="x y pressure_obs",
            fmt="%.6e",
        )

        # ----------------------------------------------------
        # 6. Gather forecast ensemble
        # ----------------------------------------------------

        rho_columns = []
        u_columns = []
        v_columns = []
        p_columns = []
        predicted_obs = []

        for i in range(n_ensemble):

            member_dir = (
                BASE
                / f"{case_name}_{i:03d}"
            )

            forecast_vtr = (
                member_dir
                / "results"
                / f"solution_{t_idx:04d}_{da_interval:04d}.vtr"
            )

            solution = m2c.load_solution(
                forecast_vtr
            )

            rho_columns.append(
                solution["rho"]
            )

            u_columns.append(
                solution["u"]
            )

            v_columns.append(
                solution["v"]
            )

            p_columns.append(
                solution["p"]
            )

            # Use the same observation indices obtained from
            # the truth grid, matching the original workflow.
            predicted_obs.append(
                solution["p"][obs_indices]
            )

        rho_matrix = np.column_stack(
            rho_columns
        )

        u_matrix = np.column_stack(
            u_columns
        )

        v_matrix = np.column_stack(
            v_columns
        )

        p_matrix = np.column_stack(
            p_columns
        )

        Hx = np.column_stack(
            predicted_obs
        )

        # ----------------------------------------------------
        # 7. Save forecast ensemble in observation space
        # ----------------------------------------------------

        header = " ".join(
            ["x", "y"]
            + [
                f"{case_name}_{i:03d}"
                for i in range(n_ensemble)
            ]
        )

        np.savetxt(
            HX_DIR
            / (
                f"Hx_{t_idx:04d}_"
                f"{da_interval:04d}.txt"
            ),
            np.column_stack([
                obs_xy,
                Hx,
            ]),
            header=header,
            fmt="%.6e",
        )

        # ----------------------------------------------------
        # 8. Compute normalization statistics
        # ----------------------------------------------------

        norm_stats = (
            neural.compute_normalization_stats(
                rho_matrix,
                u_matrix,
                v_matrix,
                p_matrix,
            )
        )

        # ----------------------------------------------------
        # 9. Build training chain
        # ----------------------------------------------------

        order, parent = chain.build_chain(
            p_matrix,
            jitter=True,
            seed=seed,
        )

        chain.save_chain(
            CHAINS_DIR
            / f"chain_p_{t_idx:04d}.txt",
            order,
            parent,
        )

        # ----------------------------------------------------
        # 10. Train forecast neural representations
        # ----------------------------------------------------

        trained_models = neural.train_chain(
            t_idx=t_idx,
            da_interval=da_interval,
            base_dir=BASE,
            order=order,
            parent=parent,
            norm_stats=norm_stats,
            case_name=case_name,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=learning_rate,
            first_lr=first_learning_rate,
            epochs=max_epochs,
            print_every=print_every,
            target_loss=target_loss,
        )

        ensemble_models = [
            trained_models[i]
            for i in range(n_ensemble)
        ]

        # ----------------------------------------------------
        # 11. Neural EnKF analysis
        # ----------------------------------------------------

        state_f = (
            assimilation.ensemble_to_matrix(
                ensemble_models
            )
        )

        R = np.diag(
            sigma_obs ** 2
        )

        # Preserve the original random-number draw and
        # positivity treatment for perturbed observations.
        eps = np.random.normal(
            0.0,
            sigma_obs[:, None],
            (
                len(obs_indices),
                n_ensemble,
            ),
        )

        obs_ensemble = (
            y_obs[:, None]
            + eps
        )

        obs_ensemble = np.where(
            obs_ensemble > 0.0,
            obs_ensemble,
            y_obs[:, None] - eps,
        )

        state_a = (
            assimilation.analysis_EnKF(
                state_f,
                Hx,
                obs_ensemble,
                R,
            )
        )

        # ----------------------------------------------------
        # 12. Reconstruct analysis and update M2C
        # ----------------------------------------------------

        for i in range(n_ensemble):

            model_a = (
                assimilation.state_to_model(
                    state_a[:, i],
                    ensemble_models[i],
                )
            )

            member_dir = (
                BASE
                / f"{case_name}_{i:03d}"
            )

            forecast_vtr = (
                member_dir
                / "results"
                / f"solution_{t_idx:04d}_{da_interval:04d}.vtr"
            )

            solution = m2c.load_solution(
                forecast_vtr
            )

            # Prepare spatial coordinates for inference.
            xy_tensor = torch.from_numpy(
                np.column_stack((
                    solution["x"],
                    solution["y"],
                )).astype(np.float32)
            )

            # Perform reconstruction on GPU when available.
            model_a.to(
                neural.device
            )

            model_a.eval()

            with torch.no_grad():

                output = (
                    model_a(
                        xy_tensor.to(
                            neural.device
                        )
                    )
                    .cpu()
                    .numpy()
                )

            # Move model back to CPU after inference.
            model_a.cpu()

            # ------------------------------------------------
            # Convert normalized output to physical variables
            # ------------------------------------------------

            (
                rho_analysis,
                u_analysis,
                v_analysis,
                p_analysis,
            ) = neural.inverse_scale_solution(
                output[:, 0],
                output[:, 1],
                output[:, 2],
                output[:, 3],
                norm_stats,
            )

            # Preserve the original pressure-floor treatment.
            p_analysis = np.maximum(
                p_analysis,
                pressure_min,
            )

            # ------------------------------------------------
            # Write analyzed state for the next forecast
            # ------------------------------------------------

            analysis_vtr = (
                member_dir
                / "IC"
                / (
                    f"solution_{t_idx:04d}_"
                    f"{da_interval:04d}_analysis.vtr"
                )
            )

            analysis_vtr.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            m2c.write_vtk_analysis(
                template_vtr=forecast_vtr,
                x_full=solution["x"],
                y_full=solution["y"],
                rho_full=rho_analysis,
                u_full=u_analysis,
                v_full=v_analysis,
                p_full=p_analysis,
                out_vtr_path=analysis_vtr,
            )

            m2c.update_userdefinedstate(
                analysis_vtr,
                member_dir,
            )

            m2c.build_case(
                member_dir,
                force_cmake=False,
            )

        print(
            f"\n>>> DA step {t_idx} complete."
        )

    print(
        "\nSUCCESS: Full Neural EnKF workflow complete."
    )


if __name__ == "__main__":
    main()