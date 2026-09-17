import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Configuration
# ============================================================

HIDDEN_DIM = 16
NUM_LAYERS = 3

FIRST_LR = 1.0e-2
LR = 1.0e-3

EPOCHS = 10000
PRINT_EVERY = 1000
TARGET_LOSS = 2.0e-6


# ============================================================
# Neural representation
# ============================================================

class Net(nn.Module):
    """
    Fully connected neural representation of a scalar field h(x).
    """

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        nonnegative=False,
    ):
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
            nn.Linear(hidden_dim, 1)
        )

        if nonnegative:
            layers.append(
                nn.Softplus()
            )

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Utilities
# ============================================================

def to_tensor(array):
    """Convert a one-dimensional array to a float tensor of shape (N, 1)."""

    return torch.as_tensor(
        array,
        dtype=torch.float32,
    ).reshape(-1, 1)


def count_params(model):
    """Return the number of trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Train one ensemble member
# ============================================================

def train_one(
    model,
    x_train,
    h_train,
    lr=LR,
    epochs=EPOCHS,
    print_every=PRINT_EVERY,
    target_loss=TARGET_LOSS,
):
    """
    Train one scalar neural representation using mean-square error.

    Returns
    -------
    model : nn.Module
        Trained neural model.

    training_time : float
        Wall-clock training time in seconds.
    """

    start_time = time.perf_counter()

    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.MSELoss()

    final_loss = np.inf

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        prediction = model(
            x_train
        )

        loss = criterion(
            prediction,
            h_train,
        )

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

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

    training_time = (
        time.perf_counter()
        - start_time
    )

    return model, training_time


# ============================================================
# Chain-based sequential training
# ============================================================

def train_chain(
    x,
    samples,
    order,
    parent,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    nonnegative=False,
    lr=LR,
    first_lr=FIRST_LR,
    epochs=EPOCHS,
    print_every=PRINT_EVERY,
    target_loss=TARGET_LOSS,
):
    """
    Train ensemble members sequentially along the prescribed chain.

    The first member is trained from a random initialization.
    Each subsequent member is initialized from its assigned parent
    model and fine-tuned on the corresponding ensemble sample.

    Parameters
    ----------
    x : ndarray, shape (n_points,)
        Spatial coordinates.

    samples : ndarray, shape (n_points, n_ensemble)
        Forecast ensemble stored column-wise.

    order : sequence of int
        Sequential training order.

    parent : sequence of int
        Warm-start parent associated with each entry in ``order``.
        The first member has parent ``-1``.

    Returns
    -------
    trained : dict
        Dictionary mapping ensemble-member indices to trained models.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    samples = np.asarray(
        samples,
        dtype=float,
    )

    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape "
            "(n_points, n_ensemble)."
        )

    if samples.shape[0] != len(x):
        raise ValueError(
            "The first dimension of samples must match len(x)."
        )

    if len(order) != len(parent):
        raise ValueError(
            "order and parent must have the same length."
        )

    x_train = to_tensor(
        x
    )

    trained = {}

    chain_start_time = (
        time.perf_counter()
    )

    print(
        "\n>>> Starting chain-based training"
    )

    for k, idx in enumerate(order):

        par = parent[k]

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        if (
            par is not None
            and par >= 0
            and par in trained
        ):
            model = Net(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                nonnegative=nonnegative,
            )

            model.load_state_dict(
                copy.deepcopy(
                    trained[par].state_dict()
                )
            )

            lr_eff = lr
            parent_name = f"{par:03d}"

        else:
            model = Net(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                nonnegative=nonnegative,
            )

            lr_eff = first_lr
            parent_name = "None"

        h_train = to_tensor(
            samples[:, idx]
        )

        print(
            f"\n    [Member {k + 1:02d}/{len(order):02d}] "
            f"{idx:03d} | Parent: {parent_name}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model, training_time = train_one(
            model=model,
            x_train=x_train,
            h_train=h_train,
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