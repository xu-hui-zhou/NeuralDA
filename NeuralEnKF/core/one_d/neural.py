import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .m2c import load_solution


# ============================================================
# Configuration
# ============================================================

EPS = 1.0e-10


# ============================================================
# Neural representation
# ============================================================

class Net(nn.Module):
    """
    Fully connected neural representation of a 1D flow state.

    The network maps the spatial coordinate x to the normalized
    flow variables (rho, u, p).
    """

    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()

        layers = [
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
        ]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        layers.append(
            nn.Linear(hidden_dim, 3)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)

        rho = F.softplus(raw[:, 0:1]) + EPS
        u = raw[:, 1:2]
        p = F.softplus(raw[:, 2:3]) + EPS

        return torch.cat(
            [rho, u, p],
            dim=1,
        )


# ============================================================
# Normalization
# ============================================================

def compute_normalization_stats(rho, u, p):
    """
    Compute normalization statistics from the forecast ensemble.

    The statistics are shared by all ensemble members within the
    current data-assimilation cycle.
    """

    return {
        "rho_max": float(np.max(rho)),
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
        "p_max": float(np.max(p)),
    }


def scale_solution(solution, norm_stats):
    """
    Normalize a physical flow state for neural-network training.

    Density and pressure are normalized by their ensemble-wide
    maximum values. Velocity is scaled to [0, 1] using the
    ensemble-wide minimum and maximum.
    """

    rho_max = max(
        norm_stats["rho_max"],
        EPS,
    )

    p_max = max(
        norm_stats["p_max"],
        EPS,
    )

    u_range = max(
        norm_stats["u_max"] - norm_stats["u_min"],
        EPS,
    )

    rho = (
        solution["rho"]
        / rho_max
    )

    u = (
        solution["u"] - norm_stats["u_min"]
    ) / u_range

    p = (
        solution["p"]
        / p_max
    )

    return rho, u, p


def inverse_scale_solution(rho, u, p, norm_stats):
    """
    Transform normalized flow variables back to physical space.
    """

    rho_max = max(
        norm_stats["rho_max"],
        EPS,
    )

    p_max = max(
        norm_stats["p_max"],
        EPS,
    )

    u_range = max(
        norm_stats["u_max"] - norm_stats["u_min"],
        EPS,
    )

    rho = rho * rho_max

    u = (
        u * u_range
        + norm_stats["u_min"]
    )

    p = p * p_max

    return rho, u, p


# ============================================================
# Train one ensemble member
# ============================================================

def train_one(
    model,
    vtr_path,
    input_st_path,
    norm_stats,
    lr=1.0e-3,
    epochs=20000,
    print_every=1000,
    target_loss=1.0e-6,
):
    """
    Train one neural representation from an M2C solution.

    Returns
    -------
    model : nn.Module
        Trained neural model.

    training_time : float
        Wall-clock training time in seconds.
    """

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Load and normalize training data
    # --------------------------------------------------------

    solution = load_solution(
        vtr_path,
        input_st_path,
    )

    x_train = torch.from_numpy(
        solution["x"].astype(np.float32)
    ).view(-1, 1)

    rho, u, p = scale_solution(
        solution,
        norm_stats,
    )

    y_train = torch.from_numpy(
        np.column_stack([
            rho,
            u,
            p,
        ]).astype(np.float32)
    )

    # --------------------------------------------------------
    # Training setup
    # --------------------------------------------------------

    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    final_loss = np.inf

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        prediction = model(
            x_train
        )

        loss = F.mse_loss(
            prediction,
            y_train,
        )

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

        if epoch % print_every == 0:
            print(
                f"        Ep {epoch:5d} | "
                f"Loss: {final_loss:.4e}"
            )

        if final_loss < target_loss:
            print(
                f"        Converged at Ep {epoch} | "
                f"Final Loss: {final_loss:.4e}"
            )
            break

    if final_loss >= target_loss:
        print(
            "        Max epochs reached | "
            f"Final Loss: {final_loss:.4e}"
        )

    training_time = (
        time.perf_counter()
        - start_time
    )

    return model, training_time


# ============================================================
# Chain-based sequential training
# ============================================================

def train_chain(
    t_idx,
    da_interval,
    base_dir,
    order,
    parent,
    norm_stats,
    case_name,
    hidden_dim=64,
    num_layers=4,
    lr=1.0e-3,
    first_lr=1.0e-2,
    epochs=20000,
    print_every=1000,
    target_loss=1.0e-6,
):
    """
    Train the ensemble sequentially along a prescribed chain.

    Each member is warm-started from its parent model when a parent
    is available. The first member is trained from a newly initialized
    network.

    Returns
    -------
    trained : dict
        Dictionary mapping ensemble-member indices to trained models.
    """

    base_dir = Path(
        base_dir
    )

    if len(order) != len(parent):
        raise ValueError(
            "order and parent must have the same length."
        )

    trained = {}

    chain_start_time = (
        time.perf_counter()
    )

    print(
        f"\n>>> Starting chain-based training "
        f"(DA step {t_idx})"
    )

    for k, idx in enumerate(order):

        par = parent[k]

        model = Net(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Warm start
        # ----------------------------------------------------

        if (
            par is not None
            and par >= 0
            and par in trained
        ):
            model.load_state_dict(
                copy.deepcopy(
                    trained[par].state_dict()
                )
            )

            lr_eff = lr
            parent_name = (
                f"{case_name}_{par:03d}"
            )

        else:
            lr_eff = first_lr
            parent_name = "None"

        member_name = (
            f"{case_name}_{idx:03d}"
        )

        print(
            f"\n    [Member {k + 1:02d}/{len(order):02d}] "
            f"{member_name} | "
            f"Parent: {parent_name}"
        )

        # ----------------------------------------------------
        # Member paths
        # ----------------------------------------------------

        member_dir = (
            base_dir
            / member_name
        )

        vtr_path = (
            member_dir
            / "results"
            / f"solution_{t_idx:04d}_{da_interval:04d}.vtr"
        )

        input_st_path = (
            member_dir
            / "input.st"
        )

        # ----------------------------------------------------
        # Train member
        # ----------------------------------------------------

        model, training_time = train_one(
            model=model,
            vtr_path=vtr_path,
            input_st_path=input_st_path,
            norm_stats=norm_stats,
            lr=lr_eff,
            epochs=epochs,
            print_every=print_every,
            target_loss=target_loss,
        )

        trained[idx] = model

        print(
            f"        Training time: "
            f"{training_time:.2f} s"
        )

    total_training_time = (
        time.perf_counter()
        - chain_start_time
    )

    print(
        f"\n>>> Chain training complete | "
        f"Total time: {total_training_time:.2f} s"
    )

    return trained