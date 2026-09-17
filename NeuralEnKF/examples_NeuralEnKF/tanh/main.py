#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Repository path
# ============================================================

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.one_d import assimilation, chain
from neural_scalar import train_chain


# ============================================================
# Paths
# ============================================================

DA_RESULTS = BASE / "da_results"


# ============================================================
# Configuration
# ============================================================

SEED = 42

# Grid
DX = 0.01
X_MIN = -1.0
X_MAX = 1.0

# Ensemble
N_ENSEMBLE = 40

# Truth
H_LEFT_TRUE = 1.0
H_RIGHT_TRUE = 0.2
SHARPNESS_TRUE = 25.0
LOCATION_TRUE = 0.0

# Prior distribution
H_LEFT_MEAN = 1.02
H_LEFT_STD = 0.10

H_RIGHT_MEAN = 0.22
H_RIGHT_STD = 0.03

SHARPNESS_STD = 4.0
LOCATION_STD = 0.10

# Observations
OBS_START_INDEX = 10
OBS_SPACING = 20
SIGMA_OBS = 0.05


# ============================================================
# Tanh profile
# ============================================================

def tanh_profile(
    x,
    amplitude,
    center,
    sharpness,
    location,
):
    """Hyperbolic-tangent profile h(x)."""

    return (
        center
        - amplitude
        * np.tanh(
            sharpness
            * (x - location)
        )
    )


# ============================================================
# Forecast ensemble
# ============================================================

def sample_prior_parameters(
    n_ensemble,
    rng,
):
    """Sample parameters defining the forecast ensemble."""

    h_left = (
        H_LEFT_MEAN
        + H_LEFT_STD
        * rng.standard_normal(n_ensemble)
    )

    h_right = (
        H_RIGHT_MEAN
        + H_RIGHT_STD
        * rng.standard_normal(n_ensemble)
    )

    # Keep the left and right states positive.
    h_left = np.clip(
        h_left,
        0.01,
        None,
    )

    h_right = np.clip(
        h_right,
        0.01,
        None,
    )

    amplitude = 0.5 * (
        h_left - h_right
    )

    center = 0.5 * (
        h_left + h_right
    )

    sharpness = (
        SHARPNESS_TRUE
        + SHARPNESS_STD
        * rng.standard_normal(n_ensemble)
    )

    location = (
        LOCATION_TRUE
        + LOCATION_STD
        * rng.standard_normal(n_ensemble)
    )

    return (
        amplitude,
        center,
        sharpness,
        location,
    )


def build_forecast_ensemble(
    x,
    amplitude,
    center,
    sharpness,
    location,
):
    """Construct the physical forecast ensemble."""

    n_ensemble = len(amplitude)

    h_forecast = np.zeros(
        (len(x), n_ensemble),
        dtype=float,
    )

    for i in range(n_ensemble):

        h_forecast[:, i] = tanh_profile(
            x,
            amplitude[i],
            center[i],
            sharpness[i],
            location[i],
        )

    return h_forecast


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rng = np.random.default_rng(
        SEED
    )

    DA_RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    x = np.arange(
        X_MIN,
        X_MAX + DX,
        DX,
    )

    # --------------------------------------------------------
    # Truth
    # --------------------------------------------------------

    amplitude_true = 0.5 * (
        H_LEFT_TRUE
        - H_RIGHT_TRUE
    )

    center_true = 0.5 * (
        H_LEFT_TRUE
        + H_RIGHT_TRUE
    )

    h_truth = tanh_profile(
        x,
        amplitude_true,
        center_true,
        SHARPNESS_TRUE,
        LOCATION_TRUE,
    )

    # --------------------------------------------------------
    # Forecast ensemble
    # --------------------------------------------------------

    (
        amplitude,
        center,
        sharpness,
        location,
    ) = sample_prior_parameters(
        N_ENSEMBLE,
        rng,
    )

    h_forecast = build_forecast_ensemble(
        x,
        amplitude,
        center,
        sharpness,
        location,
    )

    # --------------------------------------------------------
    # Observations
    # --------------------------------------------------------

    obs_idx = np.arange(
        OBS_START_INDEX,
        len(x),
        OBS_SPACING,
    )

    x_obs = x[
        obs_idx
    ]

    obs = (
        h_truth[obs_idx]
        + SIGMA_OBS
        * rng.standard_normal(
            len(obs_idx)
        )
    )

    obs_ensemble = (
        obs[:, None]
        + SIGMA_OBS
        * rng.standard_normal(
            (
                len(obs_idx),
                N_ENSEMBLE,
            )
        )
    )

    R = (
        SIGMA_OBS ** 2
        * np.eye(
            len(obs_idx)
        )
    )

    # --------------------------------------------------------
    # Training chain
    # --------------------------------------------------------

    order, parent = chain.build_chain(
        h_forecast,
        jitter=False,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Train neural forecast ensemble
    # --------------------------------------------------------

    trained_models = train_chain(
        x=x,
        samples=h_forecast,
        order=order,
        parent=parent,
    )

    ensemble_models = [
        trained_models[i]
        for i in range(N_ENSEMBLE)
    ]

    # --------------------------------------------------------
    # Neural EnKF analysis
    # --------------------------------------------------------

    state_forecast = (
        assimilation.ensemble_to_matrix(
            ensemble_models
        )
    )

    # Forecast ensemble mapped to observation space.
    state_hx = h_forecast[
        obs_idx,
        :
    ]

    state_analysis = (
        assimilation.analysis_EnKF(
            state_forecast,
            state_hx,
            obs_ensemble,
            R,
        )
    )

    # --------------------------------------------------------
    # Map analyzed parameters back to h(x)
    # --------------------------------------------------------

    x_tensor = torch.as_tensor(
        x,
        dtype=torch.float32,
    ).view(-1, 1)

    h_analysis = np.zeros_like(
        h_forecast,
        dtype=np.float32,
    )

    for i in range(N_ENSEMBLE):

        model_analysis = (
            assimilation.state_to_model(
                state_analysis[:, i],
                ensemble_models[i],
            )
        )

        model_analysis.eval()

        with torch.no_grad():

            h_analysis[:, i] = (
                model_analysis(
                    x_tensor
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    # truth.npy:
    # [x, h_truth]
    np.save(
        DA_RESULTS / "truth.npy",
        np.column_stack([
            x,
            h_truth,
        ]),
    )

    # observations.npy:
    # [x_obs, obs, sigma_obs]
    np.save(
        DA_RESULTS / "observations.npy",
        np.column_stack([
            x_obs,
            obs,
            np.full_like(
                x_obs,
                SIGMA_OBS,
            ),
        ]),
    )

    # forecast.npy:
    # [x, forecast_0, ..., forecast_Ne-1]
    np.save(
        DA_RESULTS / "forecast.npy",
        np.column_stack([
            x,
            h_forecast,
        ]),
    )

    # analysis.npy:
    # [x, analysis_0, ..., analysis_Ne-1]
    np.save(
        DA_RESULTS / "analysis.npy",
        np.column_stack([
            x,
            h_analysis,
        ]),
    )

    print(
        "\nSaved Neural EnKF results to:"
    )

    print(
        f"  {DA_RESULTS}"
    )


if __name__ == "__main__":
    main()