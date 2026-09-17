from pathlib import Path

import numpy as np


# ============================================================
# Distance metric
# ============================================================

def rmse(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute the root-mean-square distance between two arrays."""

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}"
        )

    return float(
        np.sqrt(
            np.mean(
                (a - b) ** 2
            )
        )
    )


# ============================================================
# Chain construction
# ============================================================

def build_chain(
    samples: np.ndarray,
    jitter: bool = False,
    seed: int = 42,
):
    """
    Construct a successive nearest-neighbor chain.

    Parameters
    ----------
    samples : ndarray, shape (n_state, n_ensemble)
        Ensemble samples stored column-wise.

    jitter : bool
        Randomly break near-equal distance ties.

    seed : int
        Random seed used when ``jitter=True``.

    Returns
    -------
    order : list[int]
        Training order.

    parent : list[int]
        Parent member for each entry in ``order``.
        The first member has parent ``-1``.
    """

    samples = np.asarray(samples)

    if samples.ndim != 2:
        raise ValueError(
            "samples must have shape "
            "(n_state, n_ensemble)."
        )

    n_ensemble = samples.shape[1]

    if n_ensemble == 0:
        return [], []

    # --------------------------------------------------------
    # Pairwise distances
    # --------------------------------------------------------

    distance = np.zeros(
        (n_ensemble, n_ensemble),
        dtype=float,
    )

    for i in range(n_ensemble):
        for j in range(i + 1, n_ensemble):

            d = rmse(
                samples[:, i],
                samples[:, j],
            )

            distance[i, j] = d
            distance[j, i] = d

    # --------------------------------------------------------
    # Central starting member
    # --------------------------------------------------------

    start = int(
        np.argmin(
            distance.sum(axis=1)
        )
    )

    selected = [start]
    order = [start]
    parent = [-1]

    remaining = list(
        range(n_ensemble)
    )

    remaining.remove(start)

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Successive nearest-neighbor selection
    # --------------------------------------------------------

    while remaining:

        nearest_distance = []
        nearest_parent = []

        for member in remaining:

            distances = distance[
                member,
                selected,
            ]

            nearest_idx = int(
                np.argmin(distances)
            )

            nearest_distance.append(
                float(
                    distances[nearest_idx]
                )
            )

            nearest_parent.append(
                int(
                    selected[nearest_idx]
                )
            )

        nearest_distance = np.asarray(
            nearest_distance
        )

        minimum = float(
            nearest_distance.min()
        )

        candidates = [
            i
            for i, value in enumerate(nearest_distance)
            if np.isclose(value, minimum)
        ]

        if jitter and len(candidates) > 1:
            pick = int(
                rng.choice(candidates)
            )
        else:
            pick = candidates[0]

        next_member = int(
            remaining[pick]
        )

        next_parent = int(
            nearest_parent[pick]
        )

        selected.append(
            next_member
        )

        order.append(
            next_member
        )

        parent.append(
            next_parent
        )

        remaining.remove(
            next_member
        )

    return order, parent


# ============================================================
# Chain output
# ============================================================

def save_chain(
    path,
    order,
    parent,
) -> None:
    """
    Save the training chain as two rows.

    Row 0 contains the training order.
    Row 1 contains the parent indices.
    """

    if len(order) != len(parent):
        raise ValueError(
            "order and parent must have the same length."
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array = np.vstack([
        np.asarray(
            order,
            dtype=int,
        ),
        np.asarray(
            parent,
            dtype=int,
        ),
    ])

    np.savetxt(
        path,
        array,
        fmt="%d",
        header=(
            "chain\n"
            "row0 = order, row1 = parent"
        ),
    )