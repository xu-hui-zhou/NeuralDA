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

EPS = 1.0e-12

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Neural representation
# ============================================================

class Net(nn.Module):
    """
    Fully connected neural representation of a 2D flow state.

    The network maps the spatial coordinates (x, y) to the normalized
    flow variables (rho, u, v, p).
    """

    def __init__(
        self,
        hidden_dim=100,
        num_layers=6,
    ):
        super().__init__()

        layers = [
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
        ]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ])

        layers.append(
            nn.Linear(hidden_dim, 4)
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        raw = self.net(x)

        rho = F.softplus(
            raw[:, 0:1]
        )

        u = raw[:, 1:2]

        v = raw[:, 2:3]

        p = F.softplus(
            raw[:, 3:4]
        )

        return torch.cat(
            [
                rho,
                u,
                v,
                p,
            ],
            dim=1,
        )


# ============================================================
# Normalization
# ============================================================

def compute_normalization_stats(
    rho,
    u,
    v,
    p,
):
    """
    Compute normalization statistics from the forecast ensemble.

    The statistics are shared by all ensemble members within the
    current data-assimilation cycle.
    """

    return {
        "rho_max": float(np.max(rho)),
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
        "v_min": float(np.min(v)),
        "v_max": float(np.max(v)),
        "p_max": float(np.max(p)),
    }


def scale_solution(
    solution,
    norm_stats,
):
    """
    Normalize a physical 2D flow state for neural-network training.

    Density and pressure are normalized by their ensemble-wide
    maximum values. The x- and y-velocity components are scaled
    using their ensemble-wide minimum and maximum values.
    """

    rho = (
        solution["rho"]
        / (norm_stats["rho_max"] + EPS)
    )

    u_den = (
        norm_stats["u_max"]
        - norm_stats["u_min"]
    )

    u = (
        solution["u"]
        - norm_stats["u_min"]
    ) / (
        u_den
        if u_den > EPS
        else 1.0
    )

    v_den = (
        norm_stats["v_max"]
        - norm_stats["v_min"]
    )

    v = (
        solution["v"]
        - norm_stats["v_min"]
    ) / (
        v_den
        if v_den > EPS
        else 1.0
    )

    p = (
        solution["p"]
        / (norm_stats["p_max"] + EPS)
    )

    return rho, u, v, p


def inverse_scale_solution(
    rho,
    u,
    v,
    p,
    norm_stats,
):
    """
    Transform normalized 2D flow variables back to physical space.
    """

    rho = (
        rho
        * (norm_stats["rho_max"] + EPS)
    )

    rho = np.clip(
        rho,
        EPS,
        None,
    )

    u = (
        u
        * (
            norm_stats["u_max"]
            - norm_stats["u_min"]
            + EPS
        )
        + norm_stats["u_min"]
    )

    v = (
        v
        * (
            norm_stats["v_max"]
            - norm_stats["v_min"]
            + EPS
        )
        + norm_stats["v_min"]
    )

    p = (
        p
        * (norm_stats["p_max"] + EPS)
    )

    p = np.clip(
        p,
        EPS,
        None,
    )

    return rho, u, v, p


# ============================================================
# Train one ensemble member
# ============================================================

def train_one(
    model,
    vtr_path,
    norm_stats,
    lr=1.0e-3,
    epochs=100000,
    print_every=500,
    target_loss=1.0e-5,
):
    """
    Train one neural representation from a 2D M2C solution.

    Training is performed on the available CUDA device when possible.
    The trained model is moved back to CPU before being returned.

    Returns
    -------
    model : nn.Module
        Trained neural model stored on CPU.

    training_time : float
        Wall-clock training time in seconds.
    """

    start_time = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Load and normalize training data
    # --------------------------------------------------------

    solution = load_solution(
        vtr_path
    )

    xy_train = torch.from_numpy(
        np.column_stack([
            solution["x"],
            solution["y"],
        ]).astype(np.float32)
    )

    rho, u, v, p = scale_solution(
        solution,
        norm_stats,
    )

    y_train = torch.from_numpy(
        np.column_stack([
            rho,
            u,
            v,
            p,
        ]).astype(np.float32)
    )

    # --------------------------------------------------------
    # Move model and training data to device
    # --------------------------------------------------------

    model = model.to(
        device
    )

    xy_train = xy_train.to(
        device
    )

    y_train = y_train.to(
        device
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

    for epoch in range(
        1,
        epochs + 1,
    ):

        optimizer.zero_grad()

        prediction = model(
            xy_train
        )

        loss = F.mse_loss(
            prediction,
            y_train,
        )

        loss.backward()

        optimizer.step()

        final_loss = (
            loss.item()
        )

        if epoch % print_every == 0:
            print(
                f"        Ep {epoch:6d} | "
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

    # --------------------------------------------------------
    # Move trained model back to CPU
    # --------------------------------------------------------

    model = model.cpu()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    outdir=None,
    hidden_dim=100,
    num_layers=6,
    lr=1.0e-3,
    first_lr=1.0e-2,
    epochs=100000,
    print_every=500,
    target_loss=1.0e-5,
):
    """
    Train the ensemble sequentially along a prescribed chain.

    Each member is warm-started from its parent model when a parent
    is available. The first member is trained from a newly initialized
    network.

    Parameters
    ----------
    t_idx : int
        Current data-assimilation step.

    da_interval : int
        Number of M2C time steps between data-assimilation cycles.

    base_dir : str or Path
        Directory containing the ensemble-member case directories.

    order : sequence of int
        Training order of the ensemble members.

    parent : sequence of int
        Parent member associated with each member in ``order``.

    norm_stats : dict
        Ensemble-wide normalization statistics.

    case_name : str
        Prefix used for ensemble-member directories, for example ``"bw"``.

    outdir : str or Path, optional
        Directory in which trained forecast-model state dictionaries
        are saved. If None, models are not written to disk.

    Returns
    -------
    trained : dict
        Dictionary mapping ensemble-member indices to trained models.
        All returned models are stored on CPU.
    """

    base_dir = Path(
        base_dir
    )

    if outdir is not None:

        outdir = Path(
            outdir
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True,
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
        f"(DA step {t_idx}, device={device})"
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
        # Member path
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

        # ----------------------------------------------------
        # Train member
        # ----------------------------------------------------

        model, training_time = train_one(
            model=model,
            vtr_path=vtr_path,
            norm_stats=norm_stats,
            lr=lr_eff,
            epochs=epochs,
            print_every=print_every,
            target_loss=target_loss,
        )

        trained[idx] = model

        # ----------------------------------------------------
        # Save model if requested
        # ----------------------------------------------------

        if outdir is not None:

            torch.save(
                model.state_dict(),
                outdir
                / f"{member_name}_combined.pt",
            )

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