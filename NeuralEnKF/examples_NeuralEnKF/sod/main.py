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

# Allow this script to be run directly from examples/sod/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from initial_ensemble import make_initial_ensemble
from core.one_d import assimilation, chain, m2c, neural


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

    # Initial ensemble
    n_ensemble = initial_cfg["size"]
    seed = initial_cfg["seed"]

    mean = np.array([
        initial_cfg["mean"]["x_diaphragm"],
        initial_cfg["mean"]["rho_left"],
        initial_cfg["mean"]["p_left"],
        initial_cfg["mean"]["rho_right"],
        initial_cfg["mean"]["p_right"],
    ], dtype=float)

    std = np.array([
        initial_cfg["std"]["x_diaphragm"],
        initial_cfg["std"]["rho_left"],
        initial_cfg["std"]["p_left"],
        initial_cfg["std"]["rho_right"],
        initial_cfg["std"]["p_right"],
    ], dtype=float)

    x_bounds = (
        initial_cfg["x_diaphragm_bounds"]["min"],
        initial_cfg["x_diaphragm_bounds"]["max"],
    )

    # Data assimilation
    num_cycles = assimilation_cfg["num_cycles"]
    da_interval = assimilation_cfg["interval"]

    # Observation locations and noise
    obs_x = np.arange(
        observation_cfg["x_start"],
        observation_cfg["x_end"],
        observation_cfg["spacing"],
    )

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

    # --------------------------------------------------------
    # 2. Initialization
    # --------------------------------------------------------

    np.random.seed(seed)
    torch.manual_seed(seed)

    m2c.run_allclean(BASE)

    for directory in [
        CHAINS_DIR,
        TRUTH_DIR,
        OBS_DIR,
        HX_DIR,
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
        x_bounds=x_bounds,
    )

    # Propagate initial ensemble from t = 0 to the first DA step
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
                    member_dir,
                    t_idx,
                )

                subprocess.run(
                    ["./Allrun", "input.st"],
                    cwd=member_dir,
                    check=True,
                )

        # ----------------------------------------------------
        # 4. Gather forecast ensemble
        # ----------------------------------------------------

        rho_columns = []
        u_columns = []
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
                forecast_vtr,
                member_dir / "input.st",
            )

            rho_columns.append(
                solution["rho"]
            )

            u_columns.append(
                solution["u"]
            )

            p_columns.append(
                solution["p"]
            )

            obs_indices = m2c.nearest_indices(
                solution["x"],
                obs_x,
            )

            predicted_obs.append(
                solution["p"][obs_indices]
            )

        rho_matrix = np.column_stack(
            rho_columns
        )

        u_matrix = np.column_stack(
            u_columns
        )

        p_matrix = np.column_stack(
            p_columns
        )

        Hx = np.column_stack(
            predicted_obs
        )

        # ----------------------------------------------------
        # 5. Generate observations
        # ----------------------------------------------------

        truth_vtr = (
            BASE
            / f"{case_name}.truth"
            / "results"
            / f"solution_{(t_idx + 1) * da_interval:04d}.vtr"
        )

        truth_solution = m2c.load_solution(
            truth_vtr,
            BASE
            / f"{case_name}.truth"
            / "input.st",
        )

        truth_indices = m2c.nearest_indices(
            truth_solution["x"],
            obs_x,
        )

        y_truth = (
            truth_solution["p"][truth_indices]
        )

        sigma_obs = (
            relative_noise * np.abs(y_truth)
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

        # Ensure physical pressure observations remain positive.
        y_obs = np.where(
            y_obs > 0.0,
            y_obs,
            y_truth - noise,
        )

        # Save truth and observations.
        np.savetxt(
            TRUTH_DIR
            / f"y_gt_{t_idx:04d}.txt",
            np.column_stack([
                obs_x,
                y_truth,
            ]),
            header="x pressure_truth",
            fmt="%.6e",
        )

        np.savetxt(
            OBS_DIR
            / f"y_obs_{t_idx:04d}.txt",
            np.column_stack([
                obs_x,
                y_obs,
            ]),
            header="x pressure_obs",
            fmt="%.6e",
        )

        # Save forecast ensemble in observation space.
        header = " ".join(
            ["x"]
            + [
                f"{case_name}_{i:03d}"
                for i in range(n_ensemble)
            ]
        )

        np.savetxt(
            HX_DIR
            / f"Hx_{t_idx:04d}.txt",
            np.column_stack([
                obs_x,
                Hx,
            ]),
            header=header,
            fmt="%.6e",
        )

        # ----------------------------------------------------
        # 6. Normalize forecast states for NN training
        # ----------------------------------------------------

        # One common normalization is used by the full ensemble
        # within the current DA cycle.
        norm_stats = (
            neural.compute_normalization_stats(
                rho_matrix,
                u_matrix,
                p_matrix,
            )
        )

        # ----------------------------------------------------
        # 7. Build training chain
        # ----------------------------------------------------

        order, parent = chain.build_chain(
            p_matrix,
            jitter=True,
            seed=seed,
        )

        chain.save_chain(
            CHAINS_DIR
            / f"chain_{t_idx:04d}.txt",
            order,
            parent,
        )

        # ----------------------------------------------------
        # 8. Train forecast neural representations
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
        # 9. Neural EnKF analysis
        # ----------------------------------------------------

        state_f = (
            assimilation.ensemble_to_matrix(
                ensemble_models
            )
        )

        obs_ensemble = (
            y_obs[:, None]
            + np.random.normal(
                0.0,
                sigma_obs[:, None],
                size=(
                    len(obs_x),
                    n_ensemble,
                ),
            )
        )

        R = np.diag(
            sigma_obs ** 2
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
        # 10. Write analysis back to M2C
        # ----------------------------------------------------

        for i in range(n_ensemble):

            member_dir = (
                BASE
                / f"{case_name}_{i:03d}"
            )

            model_a = (
                assimilation.state_to_model(
                    state_a[:, i],
                    ensemble_models[i],
                )
            )

            forecast_vtr = (
                member_dir
                / "results"
                / f"solution_{t_idx:04d}_{da_interval:04d}.vtr"
            )

            solution = m2c.load_solution(
                forecast_vtr,
                member_dir / "input.st",
            )

            # Neural-network output is in normalized state space.
            with torch.no_grad():

                x_tensor = torch.from_numpy(
                    solution["x"].astype(
                        np.float32
                    )
                ).view(-1, 1)

                output = (
                    model_a(x_tensor)
                    .numpy()
                )

            # Convert the analyzed state back to physical variables.
            (
                rho_analysis,
                u_analysis,
                p_analysis,
            ) = neural.inverse_scale_solution(
                output[:, 0],
                output[:, 1],
                output[:, 2],
                norm_stats,
            )

            analysis_vtr = (
                member_dir
                / "IC"
                / (
                    f"solution_{t_idx:04d}_"
                    f"{da_interval:04d}_analysis.vtr"
                )
            )

            m2c.write_vtk_analysis(
                template_vtr=forecast_vtr,
                input_st_path=member_dir
                / "input.st",
                rho_full=rho_analysis,
                u_full=u_analysis,
                p_full=p_analysis,
                out_vtr_path=analysis_vtr,
            )

            # Update and rebuild the M2C restart state.
            m2c.update_userdefinedstate(
                analysis_vtr,
                member_dir,
            )

            m2c.build_case(
                member_dir
            )

    print(
        "\nSUCCESS: Full Neural EnKF workflow complete."
    )


if __name__ == "__main__":
    main()