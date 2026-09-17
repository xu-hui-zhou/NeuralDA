import copy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# ============================================================
# Model-state conversion
# ============================================================

def model_to_state(model):
    """Flatten model parameters into a one-dimensional NumPy array."""

    return (
        parameters_to_vector(model.parameters())
        .detach()
        .cpu()
        .numpy()
    )


def state_to_model(state, template):
    """Construct a model from a flattened parameter vector."""

    model = copy.deepcopy(template).eval()

    reference_parameter = next(model.parameters())

    state_tensor = torch.as_tensor(
        state,
        dtype=reference_parameter.dtype,
        device=reference_parameter.device,
    )

    with torch.no_grad():
        vector_to_parameters(
            state_tensor,
            model.parameters(),
        )

    return model


def ensemble_to_matrix(ensemble):
    """
    Convert an ensemble of models into a state matrix.

    Parameters
    ----------
    ensemble : sequence
        Ensemble of PyTorch models.

    Returns
    -------
    state : ndarray, shape (n_state, n_ensemble)
        Matrix whose columns contain the flattened parameters
        of the ensemble members.
    """

    return np.column_stack([
        model_to_state(model)
        for model in ensemble
    ])


# ============================================================
# Ensemble Kalman analysis
# ============================================================

def analysis_EnKF(state_f, state_hx, obs, R):
    """
    Perform a stochastic ensemble Kalman filter analysis.

    Parameters
    ----------
    state_f : ndarray, shape (n_state, n_ensemble)
        Forecast ensemble in state space.

    state_hx : ndarray, shape (n_obs, n_ensemble)
        Forecast ensemble mapped to observation space.

    obs : ndarray, shape (n_obs, n_ensemble)
        Perturbed observation ensemble.

    R : ndarray, shape (n_obs, n_obs)
        Observation-error covariance matrix.

    Returns
    -------
    state_a : ndarray, shape (n_state, n_ensemble)
        Analysis ensemble.
    """

    state_f = np.asarray(state_f)
    state_hx = np.asarray(state_hx)
    obs = np.asarray(obs)
    R = np.asarray(R)

    n_ensemble = state_f.shape[1]

    if n_ensemble < 2:
        raise ValueError(
            "At least two ensemble members are required."
        )

    if state_hx.shape[1] != n_ensemble:
        raise ValueError(
            "state_f and state_hx must have the same ensemble size."
        )

    if obs.shape != state_hx.shape:
        raise ValueError(
            "obs and state_hx must have the same shape."
        )

    n_obs = state_hx.shape[0]

    if R.shape != (n_obs, n_obs):
        raise ValueError(
            f"R must have shape ({n_obs}, {n_obs})."
        )

    # Ensemble anomalies
    state_prime = state_f - np.mean(
        state_f,
        axis=1,
        keepdims=True,
    )

    hx_prime = state_hx - np.mean(
        state_hx,
        axis=1,
        keepdims=True,
    )

    # Sample covariance matrices
    coeff = 1.0 / (n_ensemble - 1)

    p_xh = coeff * (
        state_prime @ hx_prime.T
    )

    p_hh = coeff * (
        hx_prime @ hx_prime.T
    )

    # Kalman gain
    innovation_cov = p_hh + R

    kalman_gain = np.linalg.solve(
        innovation_cov.T,
        p_xh.T,
    ).T

    # Analysis update
    state_a = (
        state_f
        + kalman_gain @ (obs - state_hx)
    )

    return state_a