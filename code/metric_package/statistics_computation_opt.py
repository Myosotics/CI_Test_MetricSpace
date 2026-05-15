from metric_package.geometry_opt import (
    DataBundle,
    broadcast_pair_array,
    _main_right, _main_left,
    compute_distance,
)
from typing import Callable, Dict, Optional, Union, Literal
import numpy as np
import torch
import time


ArrayLike = Union[np.ndarray, torch.Tensor]


# ============================================================
# Data generation utilities
# ------------------------------------------------------------
# This section provides vectorized data-generating mechanisms for
# conditional independence simulations on metric spaces.
#
# Supported sample spaces:
#   - Euclidean space;
#   - the unit sphere;
#   - the SPD manifold.
#
# The implementation supports both NumPy and PyTorch backends:
#   - device="numpy" returns NumPy arrays;
#   - device="cpu" or device="cuda" returns torch tensors.
#
# Shape convention:
#   - Euclidean / sphere: (n, d)
#   - SPD:                (n, p, p)
# ============================================================


def generate_data(
    n: int,
    space_type: Literal["euclidean", "sphere", "spd"] = "euclidean",
    size: int = 2,
    rho: float = 0.0,
    seed: int | None = None,
    sigma_perm: float = 2.0,
    device: Literal["numpy", "cpu", "cuda"] | str | torch.device = "numpy",
    dtype: np.dtype | torch.dtype | None = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    r"""
    Generate simulated samples ``(X, Y, Z)`` for metric-space
    conditional independence experiments.

    Parameters
    ----------
    n : int
        Sample size.

    space_type : {"euclidean", "sphere", "spd"}, default="euclidean"
        Metric space type.

    size : int, default=2
        Dimension parameter.

        - Euclidean / sphere:
            ambient dimension ``d``.
        - SPD:
            matrix dimension ``p``.

    rho : float, default=0.0
        Dependence parameter satisfying ``-1 <= rho <= 1``.

    seed : int or None, default=None
        Random seed.

    sigma_perm : float, default=2.0
        Noise scale for Euclidean and spherical settings.

    device : {"numpy", "cpu", "cuda"} or torch.device, default="numpy"
        Backend/device used for generation.

    dtype : numpy dtype, torch.dtype, or None, default=None
        Floating-point dtype.

    Returns
    -------
    X, Y, Z : tuple[ArrayLike, ArrayLike, ArrayLike]

        Output shapes:

        - Euclidean / sphere:
            ``(n, d)``
        - SPD:
            ``(n, p, p)``

        Backend follows ``device``:

        - ``device="numpy"``:
            returns ``np.ndarray``.
        - otherwise:
            returns ``torch.Tensor``.

    Raises
    ------
    ValueError
        If inputs are invalid.

    RuntimeError
        If CUDA is requested but unavailable.
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must satisfy -1 <= rho <= 1.")

    rho_comp = max(0.0, 1.0 - rho**2) ** 0.5

    # ============================================================
    # NumPy backend
    # ============================================================
    if device == "numpy":
        if dtype is None:
            dtype = np.float64

        rng = np.random.default_rng(seed)

        if space_type == "euclidean":
            d = size
            sigma = sigma_perm

            Z = rng.normal(size=(n, d)).astype(dtype)
            U = rng.normal(size=(n, d)).astype(dtype)
            V = rng.normal(size=(n, d)).astype(dtype)

            X = Z + sigma * U
            Y = Z + sigma * (rho * U + rho_comp * V)

            return X.astype(dtype), Y.astype(dtype), Z.astype(dtype)

        elif space_type == "sphere":
            d = size
            sigma = sigma_perm

            if d < 2:
                raise ValueError("For spherical data, size must be at least 2.")

            Z = rng.normal(size=(n, d)).astype(dtype)
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)

            U = rng.normal(size=(n, d)).astype(dtype)
            U = U - np.sum(U * Z, axis=1, keepdims=True) * Z
            U_norm = np.linalg.norm(U, axis=1, keepdims=True)

            V = rng.normal(size=(n, d)).astype(dtype)
            V = V - np.sum(V * Z, axis=1, keepdims=True) * Z
            V_norm = np.linalg.norm(V, axis=1, keepdims=True)

            bad_u = U_norm[:, 0] < 1e-14
            while np.any(bad_u):
                bad_idx = np.where(bad_u)[0]

                U_new = rng.normal(size=(bad_idx.size, d)).astype(dtype)
                Z_bad = Z[bad_idx]

                U_new = U_new - np.sum(U_new * Z_bad, axis=1, keepdims=True) * Z_bad
                U_new_norm = np.linalg.norm(U_new, axis=1, keepdims=True)

                good_new = U_new_norm[:, 0] >= 1e-14
                good_idx = bad_idx[good_new]

                U[good_idx] = U_new[good_new]
                U_norm[good_idx] = U_new_norm[good_new]

                bad_u[good_idx] = False

            bad_v = V_norm[:, 0] < 1e-14
            while np.any(bad_v):
                bad_idx = np.where(bad_v)[0]

                V_new = rng.normal(size=(bad_idx.size, d)).astype(dtype)
                Z_bad = Z[bad_idx]

                V_new = V_new - np.sum(V_new * Z_bad, axis=1, keepdims=True) * Z_bad
                V_new_norm = np.linalg.norm(V_new, axis=1, keepdims=True)

                good_new = V_new_norm[:, 0] >= 1e-14
                good_idx = bad_idx[good_new]

                V[good_idx] = V_new[good_new]
                V_norm[good_idx] = V_new_norm[good_new]

                bad_v[good_idx] = False

            U /= U_norm
            V /= V_norm

            eps1 = rng.normal(size=(n, 1)).astype(dtype)
            eps2 = rng.normal(size=(n, 1)).astype(dtype)

            X = Z + sigma * eps1 * U
            Y = Z + sigma * (rho * eps1 + rho_comp * eps2) * V

            X /= np.linalg.norm(X, axis=1, keepdims=True)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)

            return X.astype(dtype), Y.astype(dtype), Z.astype(dtype)

        elif space_type == "spd":
            p = size
            nu = p + 6

            A = rng.normal(size=(n, nu, p)).astype(dtype)
            Z = np.matmul(np.transpose(A, (0, 2, 1)), A) / nu

            U = rng.normal(size=(n, nu, p)).astype(dtype)
            V = rng.normal(size=(n, nu, p)).astype(dtype)
            W = rho * U + rho_comp * V

            Sx = np.matmul(np.transpose(U, (0, 2, 1)), U) / nu
            Sy = np.matmul(np.transpose(W, (0, 2, 1)), W) / nu

            Z_half = np.linalg.cholesky(Z)

            X = Z_half @ Sx @ np.transpose(Z_half, (0, 2, 1))
            Y = Z_half @ Sy @ np.transpose(Z_half, (0, 2, 1))

            return X.astype(dtype), Y.astype(dtype), Z.astype(dtype)

        else:
            raise ValueError("space_type must be one of {'euclidean', 'sphere', 'spd'}.")

    # ============================================================
    # Torch backend: CPU / CUDA
    # ============================================================
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if dtype is None:
        dtype = torch.float64

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    if space_type == "euclidean":
        d = size
        sigma = sigma_perm

        Z = torch.randn(n, d, device=device, dtype=dtype)
        U = torch.randn(n, d, device=device, dtype=dtype)
        V = torch.randn(n, d, device=device, dtype=dtype)

        X = Z + sigma * U
        Y = Z + sigma * (rho * U + rho_comp * V)

        return X, Y, Z

    elif space_type == "sphere":
        d = size
        sigma = sigma_perm

        if d < 2:
            raise ValueError("For spherical data, size must be at least 2.")

        Z = torch.randn(n, d, device=device, dtype=dtype)
        Z = Z / torch.linalg.norm(Z, dim=1, keepdim=True)

        U = torch.randn(n, d, device=device, dtype=dtype)
        U = U - torch.sum(U * Z, dim=1, keepdim=True) * Z
        U_norm = torch.linalg.norm(U, dim=1, keepdim=True)

        V = torch.randn(n, d, device=device, dtype=dtype)
        V = V - torch.sum(V * Z, dim=1, keepdim=True) * Z
        V_norm = torch.linalg.norm(V, dim=1, keepdim=True)

        bad_u = U_norm[:, 0] < 1e-14
        while torch.any(bad_u):
            bad_idx = torch.where(bad_u)[0]

            U_new = torch.randn(bad_idx.numel(), d, device=device, dtype=dtype)
            Z_bad = Z[bad_idx]

            U_new = U_new - torch.sum(U_new * Z_bad, dim=1, keepdim=True) * Z_bad
            U_new_norm = torch.linalg.norm(U_new, dim=1, keepdim=True)

            good_new = U_new_norm[:, 0] >= 1e-14
            good_idx = bad_idx[good_new]

            U[good_idx] = U_new[good_new]
            U_norm[good_idx] = U_new_norm[good_new]

            bad_u[good_idx] = False

        bad_v = V_norm[:, 0] < 1e-14
        while torch.any(bad_v):
            bad_idx = torch.where(bad_v)[0]

            V_new = torch.randn(bad_idx.numel(), d, device=device, dtype=dtype)
            Z_bad = Z[bad_idx]

            V_new = V_new - torch.sum(V_new * Z_bad, dim=1, keepdim=True) * Z_bad
            V_new_norm = torch.linalg.norm(V_new, dim=1, keepdim=True)

            good_new = V_new_norm[:, 0] >= 1e-14
            good_idx = bad_idx[good_new]

            V[good_idx] = V_new[good_new]
            V_norm[good_idx] = V_new_norm[good_new]

            bad_v[good_idx] = False

        U = U / U_norm
        V = V / V_norm

        eps1 = torch.randn(n, 1, device=device, dtype=dtype)
        eps2 = torch.randn(n, 1, device=device, dtype=dtype)

        X = Z + sigma * eps1 * U
        Y = Z + sigma * (rho * eps1 + rho_comp * eps2) * V

        X = X / torch.linalg.norm(X, dim=1, keepdim=True)
        Y = Y / torch.linalg.norm(Y, dim=1, keepdim=True)

        return X, Y, Z

    elif space_type == "spd":
        p = size
        nu = p + 6

        A = torch.randn(n, nu, p, device=device, dtype=dtype)
        Z = torch.matmul(A.transpose(-1, -2), A) / nu

        U = torch.randn(n, nu, p, device=device, dtype=dtype)
        V = torch.randn(n, nu, p, device=device, dtype=dtype)
        W = rho * U + rho_comp * V

        Sx = torch.matmul(U.transpose(-1, -2), U) / nu
        Sy = torch.matmul(W.transpose(-1, -2), W) / nu

        Z_half = torch.linalg.cholesky(Z)

        X = Z_half @ Sx @ Z_half.transpose(-1, -2)
        Y = Z_half @ Sy @ Z_half.transpose(-1, -2)

        return X, Y, Z

    else:
        raise ValueError("space_type must be one of {'euclidean', 'sphere', 'spd'}.")


# ============================================================
# Test statistic
# ------------------------------------------------------------
# This section computes the conditional independence test statistic
# using observed samples and generated conditional samples.
#
# Shape convention:
#   - Bundle_X.data, Bundle_Y.data, Bundle_Z.data:
#       (n, d) for Euclidean / sphere
#       (n, p, p) for SPD
#   - Bundle_X_gen.data, Bundle_Y_gen.data:
#       (n, M, d) or (n, M, p, p)
#
# Backend convention:
#   - NumPy inputs use NumPy computation.
#   - Torch inputs use torch computation on their existing device.
# ============================================================


def statistics(
    Bundle_X: DataBundle,
    Bundle_Y: DataBundle,
    Bundle_Z: DataBundle,
    Bundle_X_gen: DataBundle,
    Bundle_Y_gen: DataBundle,
    atol: float = 1e-12,
    batch_size: int = 1024,
) -> float:
    r"""
    Compute the test statistic from observed and generated samples.
    Parameters
    ----------
    Bundle_X, Bundle_Y, Bundle_Z : DataBundle
        Observed samples with first dimension ``n``.
    Bundle_X_gen, Bundle_Y_gen : DataBundle
        Generated conditional samples with shape starting ``(n, M, ...)``.
    atol : float, default=1e-12
        Numerical tolerance used in distance comparisons.
    batch_size : int, default=1024
        Number of ordered pairs ``(i, j)`` processed per batch.
    Returns
    -------
    float
        Test statistic value.
    """

    # ============================================================
    # Basic backend information
    # ============================================================
    X_data = Bundle_X.data
    n = X_data.shape[0]
    M = Bundle_X_gen.data.shape[1]

    if isinstance(X_data, torch.Tensor):
        backend_info = {
            "backend": "torch",
            "device": X_data.device,
            "dtype": X_data.dtype,
        }
    elif isinstance(X_data, np.ndarray):
        backend_info = {
            "backend": "numpy",
            "dtype": X_data.dtype,
        }
    else:
        raise TypeError("Bundle_X.data must be np.ndarray or torch.Tensor.")

    # ============================================================
    # Precompute distances
    # ============================================================
    distance_dict = {
        "d_x": compute_distance(
            broadcast_pair_array(_main_left(Bundle_X), mode="ij"),
            broadcast_pair_array(_main_right(Bundle_X), mode="ji"),
            space_type=Bundle_X.space_type,
        ),

        "d_y": compute_distance(
            broadcast_pair_array(_main_left(Bundle_Y), mode="ij"),
            broadcast_pair_array(_main_right(Bundle_Y), mode="ji"),
            space_type=Bundle_Y.space_type,
        ),

        "d_z": compute_distance(
            broadcast_pair_array(_main_left(Bundle_Z), mode="ij"),
            broadcast_pair_array(_main_right(Bundle_Z), mode="ji"),
            space_type=Bundle_Z.space_type,
        ),

        "d_x_sim": compute_distance(
            broadcast_pair_array(_main_left(Bundle_X), mode="ijm", rep=M),
            broadcast_pair_array(_main_right(Bundle_X_gen), mode="jim"),
            space_type=Bundle_X.space_type,
        ),

        "d_y_sim": compute_distance(
            broadcast_pair_array(_main_left(Bundle_Y), mode="ijm", rep=M),
            broadcast_pair_array(_main_right(Bundle_Y_gen), mode="jim"),
            space_type=Bundle_Y.space_type,
        ),
    }

    total_pairs = n * (n - 1)
    total = 0.0

    # ============================================================
    # Torch backend
    # ============================================================
    if backend_info["backend"] == "torch":

        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)

            k = torch.arange(start, end, device=backend_info["device"], dtype=torch.long)
            ib = torch.div(k, n - 1, rounding_mode="floor")
            r = torch.remainder(k, n - 1)
            jb = r + (r >= ib).to(torch.long)

            dx_ij = distance_dict["d_x"][ib, jb]
            dy_ij = distance_dict["d_y"][ib, jb]
            dz_ij = distance_dict["d_z"][ib, jb]

            emdf_P = (
                (distance_dict["d_x"][ib, :] <= dx_ij[:, None] + atol)
                & (distance_dict["d_y"][ib, :] <= dy_ij[:, None] + atol)
                & (distance_dict["d_z"][ib, :] <= dz_ij[:, None] + atol)
            ).to(backend_info["dtype"]).mean(dim=1)

            tx = (
                distance_dict["d_x_sim"][ib, :, :] <= dx_ij[:, None, None] + atol
            ).to(backend_info["dtype"]).mean(dim=2)

            ty = (
                distance_dict["d_y_sim"][ib, :, :] <= dy_ij[:, None, None] + atol
            ).to(backend_info["dtype"]).mean(dim=2)

            dz = (
                distance_dict["d_z"][ib, :] <= dz_ij[:, None] + atol
            ).to(backend_info["dtype"])

            emdf_I = (tx * ty * dz).mean(dim=1)

            diff = emdf_P - emdf_I
            total += float((diff * diff).sum().item())

        return total / total_pairs

    # ============================================================
    # NumPy backend
    # ============================================================
    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)

        k = np.arange(start, end, dtype=np.int64)
        ib = k // (n - 1)
        r = k % (n - 1)
        jb = r + (r >= ib)

        dx_ij = distance_dict["d_x"][ib, jb]
        dy_ij = distance_dict["d_y"][ib, jb]
        dz_ij = distance_dict["d_z"][ib, jb]

        emdf_P = (
            (distance_dict["d_x"][ib, :] <= dx_ij[:, None] + atol)
            & (distance_dict["d_y"][ib, :] <= dy_ij[:, None] + atol)
            & (distance_dict["d_z"][ib, :] <= dz_ij[:, None] + atol)
        ).astype(backend_info["dtype"]).mean(axis=1)

        tx = (
            distance_dict["d_x_sim"][ib, :, :] <= dx_ij[:, None, None] + atol
        ).astype(backend_info["dtype"]).mean(axis=2)

        ty = (
            distance_dict["d_y_sim"][ib, :, :] <= dy_ij[:, None, None] + atol
        ).astype(backend_info["dtype"]).mean(axis=2)

        dz = (
            distance_dict["d_z"][ib, :] <= dz_ij[:, None] + atol
        ).astype(backend_info["dtype"])

        emdf_I = (tx * ty * dz).mean(axis=1)

        diff = emdf_P - emdf_I
        total += float(np.sum(diff * diff))

    return total / total_pairs


# ============================================================
# Local permutation utilities
# ------------------------------------------------------------
# This section implements the local permutation procedure used
# for conditional independence testing on metric-space data.
#
# Main components:
#   1. compute_pairwise_distances:
#      compute the pairwise distance matrix of conditioning samples.
#
#   2. build_knn_indices:
#      construct local neighborhoods from the pairwise distance matrix.
#
#   3. local_permute_y:
#      generate locally permuted indices for Y using the KNN sets.
#
#   4. local_permutation_test:
#      run the full permutation test using repeated local permutations.
#
# Backend convention:
#   - NumPy input bundles use NumPy arrays and np.random.Generator.
#   - Torch input bundles use torch tensors and torch.Generator.
# ============================================================


def compute_pairwise_distances(
    Bundle_Z: DataBundle,
    batch_size: int | None = None,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Compute the pairwise distance matrix for conditioning samples.

    Parameters
    ----------
    Bundle_Z : DataBundle
        Conditioning sample bundle.

        Shape of ``Bundle_Z.data``:
        - Euclidean / sphere: ``(n, d)``
        - SPD: ``(n, p, p)``

    batch_size : int or None, default=None
        Number of upper-triangular pairs processed per batch.
        If ``None``, all pairs are processed at once.

    Returns
    -------
    np.ndarray or torch.Tensor
        Pairwise distance matrix of shape ``(n, n)``.
    """

    Z_data = Bundle_Z.data
    
    n = Z_data.shape[0]

    if isinstance(Z_data, torch.Tensor):
        backend = "torch"
        device = Z_data.device
        dtype = Z_data.dtype

    elif isinstance(Z_data, np.ndarray):
        backend = "numpy"
        dtype = Z_data.dtype
    else:
        raise TypeError("Bundle_X.data must be np.ndarray or torch.Tensor.")

    n_total = n * (n - 1) // 2

    # ============================================================
    # Torch backend
    # ============================================================
    if backend == "torch":
        D = torch.zeros((n, n), device=device, dtype=dtype)
        if n_total == 0:
            return D
        if batch_size is None:
            batch_size = n_total
        row_idx, col_idx = torch.triu_indices(
            n,
            n,
            offset=1,
            device=device,
        )
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)

            i_idx = row_idx[start:end]
            j_idx = col_idx[start:end]

            d_ij = compute_distance(
                _main_left(Bundle_Z.slice(i_idx)),
                _main_right(Bundle_Z.slice(j_idx)),
                space_type=Bundle_Z.space_type,
            )

            D[i_idx, j_idx] = d_ij
            D[j_idx, i_idx] = d_ij

        return D

    # ============================================================
    # NumPy backend
    # ============================================================
    if backend == "numpy":
        D = np.zeros((n, n), dtype=dtype)
        if n_total == 0:
            return D
        if batch_size is None:
            batch_size = n_total
        row_idx, col_idx = np.triu_indices(
            n,
            k=1,
        )

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)

            i_idx = row_idx[start:end]
            j_idx = col_idx[start:end]

            d_ij = compute_distance(
                _main_left(Bundle_Z.slice(i_idx)),
                _main_right(Bundle_Z.slice(j_idx)),
                space_type=Bundle_Z.space_type,
            )

            D[i_idx, j_idx] = d_ij
            D[j_idx, i_idx] = d_ij

        return D


def build_knn_indices(
    Bundle_Z: DataBundle,
    k: int,
    include_self: bool = True,
    batch_size: int | None = None,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Build k-nearest-neighbor indices from conditioning samples.

    Parameters
    ----------
    Bundle_Z : DataBundle
        Conditioning sample bundle with first dimension ``n``.

    k : int
        Number of neighbors.

    include_self : bool, default=True
        Whether each point may include itself as a neighbor.

    batch_size : int or None, default=None
        Batch size used when computing pairwise distances.

    Returns
    -------
    np.ndarray or torch.Tensor
        Neighbor indices of shape ``(n, k)``.
    """
    if not isinstance(Bundle_Z, DataBundle):
        raise TypeError("Bundle_Z must be a DataBundle.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    D = compute_pairwise_distances(
        Bundle_Z=Bundle_Z,
        batch_size=batch_size,
    )

    n = D.shape[0]

    if include_self:
        if k > n:
            raise ValueError(
                f"When include_self=True, k={k} cannot exceed sample size n={n}."
            )
    else:
        if k >= n:
            raise ValueError(
                f"When include_self=False, k={k} must be smaller than n={n}."
            )

    # ============================================================
    # Torch backend
    # ============================================================
    if isinstance(D, torch.Tensor):

        if not include_self:
            diag_idx = torch.arange(n, device=D.device)
            D[diag_idx, diag_idx] = torch.inf

        cand = torch.topk(
            D,
            k=k,
            dim=1,
            largest=False,
        ).indices

        cand_dist = torch.gather(
            D,
            dim=1,
            index=cand,
        )

        order = torch.argsort(
            cand_dist,
            dim=1,
        )

        return torch.gather(
            cand,
            dim=1,
            index=order,
        )

    # ============================================================
    # NumPy backend
    # ============================================================
    if isinstance(D, np.ndarray):

        if not include_self:
            diag_idx = np.arange(n)
            D[diag_idx, diag_idx] = np.inf

        cand = np.argpartition(
            D,
            kth=k - 1,
            axis=1,
        )[:, :k]

        cand_dist = np.take_along_axis(
            D,
            cand,
            axis=1,
        )

        order = np.argsort(
            cand_dist,
            axis=1,
        )

        return np.take_along_axis(
            cand,
            order,
            axis=1,
        ).astype(np.int64, copy=False)

    raise TypeError("Distance matrix must be np.ndarray or torch.Tensor.")


def local_permute_y(
    Bundle_Y: DataBundle,
    knn_indices: Union[np.ndarray, torch.Tensor],
    generator: np.random.Generator | torch.Generator | None = None,
    strategy: str = "resample",
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Generate locally permuted indices for ``Y``.

    Parameters
    ----------
    Bundle_Y : DataBundle
        Data bundle for ``Y`` with first dimension ``n``.

    knn_indices : np.ndarray or torch.Tensor
        Neighbor index array of shape ``(n, k)``.

    generator : np.random.Generator, torch.Generator, or None, default=None
        Random number generator matching the backend.

    strategy : {"resample", "shuffle_once"}, default="resample"
        Local permutation strategy.

    Returns
    -------
    np.ndarray or torch.Tensor
        Chosen permutation indices of shape ``(n,)``.
    """

    if not isinstance(Bundle_Y, DataBundle):
        raise TypeError("Bundle_Y must be a DataBundle.")

    Y = Bundle_Y.data
    n = Y.shape[0]
        
    if not isinstance(knn_indices, (np.ndarray, torch.Tensor)):
        raise TypeError("knn_indices must be np.ndarray or torch.Tensor.")

    if knn_indices.ndim != 2:
        raise ValueError(
            f"knn_indices must have shape (n, k), got {tuple(knn_indices.shape)}."
        )

    if knn_indices.shape[0] != n:
        raise ValueError("knn_indices and Bundle_Y must have the same first dimension.")

    k = knn_indices.shape[1]

    if k == 0:
        raise ValueError("Each row of knn_indices must contain at least one neighbor.")

    # ============================================================
    # Torch backend
    # ============================================================
    if isinstance(Y, torch.Tensor):

        if not isinstance(knn_indices, torch.Tensor):
            raise TypeError("For torch Bundle_Y, knn_indices must be torch.Tensor.")

        if knn_indices.device != Y.device:
            raise ValueError("knn_indices must be on the same device as Bundle_Y.data.")

        if knn_indices.dtype != torch.long:
            raise ValueError("knn_indices must have dtype torch.long.")

        device = Y.device

        if generator is None:
            generator = torch.Generator(device=device)
        elif not isinstance(generator, torch.Generator):
            raise TypeError("For torch backend, generator must be torch.Generator or None.")

        if strategy == "resample":
            rand_col = torch.randint(
                low=0,
                high=k,
                size=(n,),
                device=device,
                generator=generator,
            )

            row_idx = torch.arange(n, device=device)
            chosen = knn_indices[row_idx, rand_col]

            return chosen
        
        elif strategy == "shuffle_once":
            perm = torch.randperm(n, device=device, generator=generator)

            rank = torch.empty(n, device=device, dtype=torch.long)
            rank[perm] = torch.arange(n, device=device, dtype=torch.long)

            available = torch.ones(n, device=device, dtype=torch.bool)
            chosen = torch.empty(n, device=device, dtype=torch.long)

            last_selected: int | None = None

            for i in range(n):
                neigh = knn_indices[i]
                feasible = neigh[available[neigh]]

                if feasible.numel() > 0:
                    selected = feasible[torch.argmin(rank[feasible])]
                    chosen[i] = selected
                    available[selected] = False
                    last_selected = int(selected.item())
                else:
                    if last_selected is not None:
                        chosen[i] = last_selected
                    else:
                        rand_pos = torch.randint(
                            low=0,
                            high=neigh.numel(),
                            size=(1,),
                            device=device,
                            generator=generator,
                        )
                        chosen[i] = neigh[rand_pos.item()]

            return chosen
        
        raise ValueError("strategy must be one of {'resample', 'shuffle_once'}.")

    # ============================================================
    # NumPy backend
    # ============================================================
    elif isinstance(Y, np.ndarray):

        if not isinstance(knn_indices, np.ndarray):
            raise TypeError("For NumPy Bundle_Y, knn_indices must be np.ndarray.")

        if not np.issubdtype(knn_indices.dtype, np.integer):
            raise ValueError("knn_indices must have integer dtype.")

        if generator is None:
            generator = np.random.default_rng()
        elif not isinstance(generator, np.random.Generator):
            raise TypeError("For NumPy backend, generator must be np.random.Generator or None.")

        if strategy == "resample":
            rand_col = generator.integers(
                low=0,
                high=k,
                size=n,
            )

            row_idx = np.arange(n)
            chosen = knn_indices[row_idx, rand_col]

            return chosen
        
        elif strategy == "shuffle_once":

            perm = generator.permutation(n)

            rank = np.empty(n, dtype=np.int64)
            rank[perm] = np.arange(n, dtype=np.int64)

            available = np.ones(n, dtype=bool)
            chosen = np.empty(n, dtype=np.int64)

            last_selected: int | None = None

            for i in range(n):
                neigh = knn_indices[i]
                feasible = neigh[available[neigh]]

                if feasible.size > 0:
                    selected = feasible[np.argmin(rank[feasible])]
                    chosen[i] = selected
                    available[selected] = False
                    last_selected = int(selected)
                else:
                    if last_selected is not None:
                        chosen[i] = last_selected
                    else:
                        rand_pos = generator.integers(
                            low=0,
                            high=neigh.size,
                        )
                        chosen[i] = neigh[rand_pos]

            return chosen
        
        raise ValueError("strategy must be one of {'resample', 'shuffle_once'}.")

    raise TypeError("Bundle_Y.data must be np.ndarray or torch.Tensor.")


def local_permutation_test(
    Bundle_X: DataBundle,
    Bundle_Y: DataBundle,
    Bundle_Z: DataBundle,
    stat_fn: Callable[..., float],
    stat_kwargs: Optional[Dict] = None,
    B: int = 500,
    k: Optional[int] = None,
    permutation_strategy: str = "resample",
    alpha: float = 0.05,
    generator: np.random.Generator | torch.Generator | None = None,
    show_progress: bool = False,
    knn_batch_size: int | None = None,
) -> Dict[str, object]:
    r"""
    Perform a local permutation test.

    Parameters
    ----------
    Bundle_X, Bundle_Y, Bundle_Z : DataBundle
        Observed data bundles with shared first dimension ``n``.

    stat_fn : callable
        Test statistic function.

    stat_kwargs : dict or None, default=None
        Additional arguments passed to ``stat_fn``.

    B : int, default=500
        Number of permutation replicates.

    k : int or None, default=None
        Neighborhood size. If ``None``, uses ``max(10, ceil(sqrt(n)))``.

    permutation_strategy : {"resample", "shuffle_once"}, default="resample"
        Strategy used by ``local_permute_y``.

    alpha : float, default=0.05
        Significance level.

    generator : np.random.Generator, torch.Generator, or None, default=None
        Random number generator matching the backend.

    show_progress : bool, default=False
        Whether to show a progress bar.

    knn_batch_size : int or None, default=None
        Batch size for KNN distance computation.

    Returns
    -------
    dict
        Test result containing ``T_obs``, ``T_perm``, ``p_value``,
        ``reject``, ``k``, and ``knn_indices``.
    """

    if stat_kwargs is None:
        stat_kwargs = {}

    if not isinstance(Bundle_X, DataBundle):
        raise TypeError("Bundle_X must be a DataBundle.")
    if not isinstance(Bundle_Y, DataBundle):
        raise TypeError("Bundle_Y must be a DataBundle.")
    if not isinstance(Bundle_Z, DataBundle):
        raise TypeError("Bundle_Z must be a DataBundle.")

    if Bundle_X.data.shape[0] != Bundle_Y.data.shape[0] or Bundle_X.data.shape[0] != Bundle_Z.data.shape[0]:
        raise ValueError("Bundle_X, Bundle_Y, and Bundle_Z must have the same sample size.")

    if Bundle_X.space_type != Bundle_Y.space_type or Bundle_X.space_type != Bundle_Z.space_type:
        raise ValueError("Bundle_X, Bundle_Y, and Bundle_Z must have the same space_type.")

    Z_data = Bundle_Z.data
    n = Z_data.shape[0]

    if n < 2:
        raise ValueError("At least two observations are required.")

    if B <= 0:
        raise ValueError("B must be positive.")

    if k is None:
        k = max(10, int(np.ceil(np.sqrt(float(n)))))

    if k >= n:
        k = n - 1

    if k < 1:
        raise ValueError("k must be at least 1 after adjustment.")

    # ============================================================
    # Backend setup
    # ============================================================

    if isinstance(Z_data, torch.Tensor):

        device = Z_data.device
        dtype = Z_data.dtype

        if generator is None:
            generator = torch.Generator(device=device)
        elif not isinstance(generator, torch.Generator):
            raise TypeError("For torch backend, generator must be torch.Generator or None.")

        T_perm = torch.empty((B,), device=device, dtype=dtype)

    elif isinstance(Z_data, np.ndarray):

        dtype = Z_data.dtype

        if generator is None:
            generator = np.random.default_rng()
        elif not isinstance(generator, np.random.Generator):
            raise TypeError("For NumPy backend, generator must be np.random.Generator or None.")

        T_perm = np.empty((B,), dtype=dtype)

    else:
        raise TypeError("Bundle_Z.data must be np.ndarray or torch.Tensor.")

    # ============================================================
    # KNN construction
    # ============================================================

    knn_indices = build_knn_indices(
        Bundle_Z=Bundle_Z,
        k=k,
        include_self=False,
        batch_size=knn_batch_size,
    )

    # ============================================================
    # Observed statistic
    # ============================================================

    T_obs = stat_fn(
        Bundle_X,
        Bundle_Y,
        Bundle_Z,
        **stat_kwargs,
    )

    T_obs_float = float(T_obs.item()) if isinstance(T_obs, torch.Tensor) else float(T_obs)

    # ============================================================
    # Permutation statistics
    # ============================================================

    iterator = range(B)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Permutation", leave=False)

    for b in iterator:
        perm_index = local_permute_y(
            Bundle_Y=Bundle_Y,
            knn_indices=knn_indices,
            generator=generator,
            strategy=permutation_strategy,
        )

        Bundle_Y_star = Bundle_Y.slice(perm_index)

        Tb = stat_fn(
            Bundle_X,
            Bundle_Y_star,
            Bundle_Z,
            **stat_kwargs,
        )

        if isinstance(T_perm, torch.Tensor):
            T_perm[b] = Tb if isinstance(Tb, torch.Tensor) else float(Tb)
        else:
            T_perm[b] = float(Tb.item()) if isinstance(Tb, torch.Tensor) else float(Tb)

    # ============================================================
    # P-value
    # ============================================================

    if isinstance(T_perm, torch.Tensor):
        p_value_tensor = (
            1.0 + torch.sum(T_perm >= T_obs_float).to(dtype=dtype)
        ) / (B + 1.0)

        p_value = float(p_value_tensor.item())

    else:
        p_value = float(
            (1.0 + np.sum(T_perm >= T_obs_float)) / (B + 1.0)
        )

    return {
        "T_obs": T_obs_float,
        "T_perm": T_perm,
        "p_value": p_value,
        "reject": bool(p_value <= alpha),
        "k": int(k),
        "knn_indices": knn_indices,
    }


# ============================================================
# Optimized local permutation statistic
# ------------------------------------------------------------
# This section implements an optimized local permutation test.
#
# The observed statistic is computed once by ``statistics_obs``.
# During this step, distance matrices and reusable components of
# the independence empirical distribution are cached.
#
# Each permutation replicate then calls ``statistics_perm``, which
# reuses the cached quantities and only updates the parts affected
# by the local permutation of Y.
#
# Shape convention:
#   - Observed bundles:
#       (n, d) for Euclidean / sphere
#       (n, p, p) for SPD
#   - Generated bundles:
#       (n, M, d) or (n, M, p, p)
#
# Backend convention:
#   - NumPy bundles use NumPy arrays.
#   - Torch bundles use torch tensors on their existing device.
# ============================================================


def statistics_obs(
    Bundle_X: DataBundle,
    Bundle_Y: DataBundle,
    Bundle_Z: DataBundle,
    Bundle_X_gen: DataBundle,
    Bundle_Y_gen: DataBundle,
    atol: float = 1e-12,
    batch_size: int = 1024,
):
    r"""
    Compute the observed statistic and cache reusable quantities.

    Parameters
    ----------
    Bundle_X, Bundle_Y, Bundle_Z : DataBundle
        Observed samples with first dimension ``n``.

    Bundle_X_gen, Bundle_Y_gen : DataBundle
        Generated conditional samples with shape starting ``(n, M, ...)``.

    atol : float, default=1e-12
        Numerical tolerance used in distance comparisons.

    batch_size : int, default=1024
        Number of ordered pairs ``(i, j)`` processed per batch.

    Returns
    -------
    T_obs : float
        Observed statistic.

    backend_info : dict
        Backend metadata, including backend, dtype, and batch size.

    distance_dict : dict
        Precomputed distance arrays/tensors.

    emdf_I_info : dict
        Cached reusable terms for permutation statistics.
    """

    X_data = Bundle_X.data
    n = X_data.shape[0]
    M = Bundle_X_gen.data.shape[1]

    if isinstance(X_data, torch.Tensor):
        backend_info = {
            "backend": "torch",
            "device": X_data.device,
            "dtype": X_data.dtype,
            "batch_size": batch_size,
        }
    elif isinstance(X_data, np.ndarray):
        backend_info = {
            "backend": "numpy",
            "dtype": X_data.dtype,
            "batch_size": batch_size,
        }
    else:
        raise TypeError("Bundle_X.data must be np.ndarray or torch.Tensor.")

    distance_dict = {
        "d_x": compute_distance(
            broadcast_pair_array(_main_left(Bundle_X), mode="ij"),
            broadcast_pair_array(_main_right(Bundle_X), mode="ji"),
            space_type=Bundle_X.space_type,
        ),
        "d_y": compute_distance(
            broadcast_pair_array(_main_left(Bundle_Y), mode="ij"),
            broadcast_pair_array(_main_right(Bundle_Y), mode="ji"),
            space_type=Bundle_Y.space_type,
        ),
        "d_z": compute_distance(
            broadcast_pair_array(_main_left(Bundle_Z), mode="ij"),
            broadcast_pair_array(_main_right(Bundle_Z), mode="ji"),
            space_type=Bundle_Z.space_type,
        ),
        "d_x_sim": compute_distance(
            broadcast_pair_array(_main_left(Bundle_X), mode="ijm", rep=M),
            broadcast_pair_array(_main_right(Bundle_X_gen), mode="jim"),
            space_type=Bundle_X.space_type,
        ),
        "d_y_sim": compute_distance(
            broadcast_pair_array(_main_left(Bundle_Y), mode="ijm", rep=M),
            broadcast_pair_array(_main_right(Bundle_Y_gen), mode="jim"),
            space_type=Bundle_Y.space_type,
        ),
    }

    total_pairs = n * (n - 1)
    total = 0.0

    if backend_info["backend"] == "torch":

        tx_cache = torch.empty(
            (total_pairs, n),
            device=backend_info["device"],
            dtype=backend_info["dtype"],
        )
        dz_cache = torch.empty(
            (total_pairs, n),
            device=backend_info["device"],
            dtype=backend_info["dtype"],
        )

        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)

            k = torch.arange(
                start,
                end,
                device=backend_info["device"],
                dtype=torch.long,
            )
            ib = torch.div(k, n - 1, rounding_mode="floor")
            r = torch.remainder(k, n - 1)
            jb = r + (r >= ib).to(torch.long)

            dx_ij = distance_dict["d_x"][ib, jb]
            dy_ij = distance_dict["d_y"][ib, jb]
            dz_ij = distance_dict["d_z"][ib, jb]

            emdf_P = (
                (distance_dict["d_x"][ib, :] <= dx_ij[:, None] + atol)
                & (distance_dict["d_y"][ib, :] <= dy_ij[:, None] + atol)
                & (distance_dict["d_z"][ib, :] <= dz_ij[:, None] + atol)
            ).to(backend_info["dtype"]).mean(dim=1)

            tx = (
                distance_dict["d_x_sim"][ib, :, :]
                <= dx_ij[:, None, None] + atol
            ).to(backend_info["dtype"]).mean(dim=2)

            ty = (
                distance_dict["d_y_sim"][ib, :, :]
                <= dy_ij[:, None, None] + atol
            ).to(backend_info["dtype"]).mean(dim=2)

            dz = (
                distance_dict["d_z"][ib, :]
                <= dz_ij[:, None] + atol
            ).to(backend_info["dtype"])

            tx_cache[start:end, :] = tx
            dz_cache[start:end, :] = dz

            emdf_I = (tx * ty * dz).mean(dim=1)

            diff = emdf_P - emdf_I
            total += float((diff * diff).sum().item())

    else:

        tx_cache = np.empty(
            (total_pairs, n),
            dtype=backend_info["dtype"],
        )
        dz_cache = np.empty(
            (total_pairs, n),
            dtype=backend_info["dtype"],
        )

        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)

            k = np.arange(start, end, dtype=np.int64)
            ib = k // (n - 1)
            r = k % (n - 1)
            jb = r + (r >= ib)

            dx_ij = distance_dict["d_x"][ib, jb]
            dy_ij = distance_dict["d_y"][ib, jb]
            dz_ij = distance_dict["d_z"][ib, jb]

            emdf_P = (
                (distance_dict["d_x"][ib, :] <= dx_ij[:, None] + atol)
                & (distance_dict["d_y"][ib, :] <= dy_ij[:, None] + atol)
                & (distance_dict["d_z"][ib, :] <= dz_ij[:, None] + atol)
            ).astype(backend_info["dtype"]).mean(axis=1)

            tx = (
                distance_dict["d_x_sim"][ib, :, :]
                <= dx_ij[:, None, None] + atol
            ).astype(backend_info["dtype"]).mean(axis=2)

            ty = (
                distance_dict["d_y_sim"][ib, :, :]
                <= dy_ij[:, None, None] + atol
            ).astype(backend_info["dtype"]).mean(axis=2)

            dz = (
                distance_dict["d_z"][ib, :]
                <= dz_ij[:, None] + atol
            ).astype(backend_info["dtype"])

            tx_cache[start:end, :] = tx
            dz_cache[start:end, :] = dz

            emdf_I = (tx * ty * dz).mean(axis=1)

            diff = emdf_P - emdf_I
            total += float(np.sum(diff * diff))

    emdf_I_info = {
        "tx_cache": tx_cache,
        "dz_cache": dz_cache,
    }

    return total / total_pairs, backend_info, distance_dict, emdf_I_info


def statistics_perm(
    backend_info: dict,
    distance_dict: dict,
    emdf_I_info: dict,
    perm_index,
    atol: float = 1e-12,
) -> float:
    r"""
    Compute the statistic for one local permutation of Y.

    Parameters
    ----------
    backend_info : dict
        Backend metadata returned by ``statistics_obs``.

    distance_dict : dict
        Precomputed distance arrays/tensors returned by ``statistics_obs``.

    emdf_I_info : dict
        Cached reusable terms returned by ``statistics_obs``.

    perm_index : np.ndarray or torch.Tensor
        Local permutation indices of shape ``(n,)``.

    atol : float, default=1e-12
        Numerical tolerance used in distance comparisons.

    Returns
    -------
    float
        Permutation statistic.
    """

    d_x = distance_dict["d_x"]
    d_y = distance_dict["d_y"]
    d_z = distance_dict["d_z"]
    d_y_sim = distance_dict["d_y_sim"]

    tx_cache = emdf_I_info["tx_cache"]
    dz_cache = emdf_I_info["dz_cache"]

    batch_size = backend_info["batch_size"]

    n = d_x.shape[0]
    total_pairs = n * (n - 1)
    total = 0.0

    if backend_info["backend"] == "torch":

        device = backend_info["device"]
        dtype = backend_info["dtype"]

        if not isinstance(perm_index, torch.Tensor):
            perm_index = torch.as_tensor(
                perm_index,
                device=device,
                dtype=torch.long,
            )
        else:
            perm_index = perm_index.to(device=device, dtype=torch.long)

        d_y_perm = d_y[perm_index][:, perm_index]
        d_y_sim_perm = d_y_sim[perm_index, :, :]

        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)

            k = torch.arange(start, end, device=device, dtype=torch.long)
            ib = torch.div(k, n - 1, rounding_mode="floor")
            r = torch.remainder(k, n - 1)
            jb = r + (r >= ib).to(torch.long)

            dx_ij = d_x[ib, jb]
            dy_ij = d_y_perm[ib, jb]
            dz_ij = d_z[ib, jb]

            emdf_P = (
                (d_x[ib, :] <= dx_ij[:, None] + atol)
                & (d_y_perm[ib, :] <= dy_ij[:, None] + atol)
                & (d_z[ib, :] <= dz_ij[:, None] + atol)
            ).to(dtype).mean(dim=1)

            tx = tx_cache[start:end, :]
            dz = dz_cache[start:end, :]

            ty = (
                d_y_sim_perm[ib, :, :]
                <= dy_ij[:, None, None] + atol
            ).to(dtype).mean(dim=2)

            emdf_I = (tx * ty * dz).mean(dim=1)

            diff = emdf_P - emdf_I
            total += float((diff * diff).sum().item())

        return total / total_pairs

    dtype = backend_info["dtype"]
    perm_index = np.asarray(perm_index, dtype=np.int64)

    d_y_perm = d_y[np.ix_(perm_index, perm_index)]
    d_y_sim_perm = d_y_sim[perm_index, :, :]

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)

        k = np.arange(start, end, dtype=np.int64)
        ib = k // (n - 1)
        r = k % (n - 1)
        jb = r + (r >= ib)

        dx_ij = d_x[ib, jb]
        dy_ij = d_y_perm[ib, jb]
        dz_ij = d_z[ib, jb]

        emdf_P = (
            (d_x[ib, :] <= dx_ij[:, None] + atol)
            & (d_y_perm[ib, :] <= dy_ij[:, None] + atol)
            & (d_z[ib, :] <= dz_ij[:, None] + atol)
        ).astype(dtype).mean(axis=1)

        tx = tx_cache[start:end, :]
        dz = dz_cache[start:end, :]

        ty = (
            d_y_sim_perm[ib, :, :]
            <= dy_ij[:, None, None] + atol
        ).astype(dtype).mean(axis=2)

        emdf_I = (tx * ty * dz).mean(axis=1)

        diff = emdf_P - emdf_I
        total += float(np.sum(diff * diff))

    return total / total_pairs


def local_permutation_test_opt(
    Bundle_X: DataBundle,
    Bundle_Y: DataBundle,
    Bundle_Z: DataBundle,
    stat_kwargs: Optional[Dict] = None,
    B: int = 500,
    k: Optional[int] = None,
    permutation_strategy: str = "resample",
    alpha: float = 0.05,
    generator: np.random.Generator | torch.Generator | None = None,
    show_progress: bool = False,
    knn_batch_size: int | None = None,
) -> Dict[str, object]:
    r"""
    Perform the optimized local permutation test.

    Parameters
    ----------
    Bundle_X, Bundle_Y, Bundle_Z : DataBundle
        Observed data bundles with shared first dimension ``n``.

    stat_kwargs : dict or None, default=None
        Keyword arguments passed to ``statistics_obs``.
        Should include generated bundles such as ``Bundle_X_gen`` and
        ``Bundle_Y_gen``.

    B : int, default=500
        Number of permutation replicates.

    k : int or None, default=None
        Neighborhood size. If ``None``, uses ``max(10, ceil(sqrt(n)))``.

    permutation_strategy : {"resample", "shuffle_once"}, default="resample"
        Strategy used to generate local permutation indices.

    alpha : float, default=0.05
        Significance level.

    generator : np.random.Generator, torch.Generator, or None, default=None
        Random number generator matching the backend.

    show_progress : bool, default=False
        Whether to show a progress bar.

    knn_batch_size : int or None, default=None
        Batch size used when computing KNN distances.

    Returns
    -------
    dict
        Test result containing ``T_obs``, ``T_perm``, ``p_value``,
        ``reject``, ``k``, and ``knn_indices``.
    """

    if stat_kwargs is None:
        stat_kwargs = {}

    # ============================================================
    # Basic checks
    # ============================================================

    if not isinstance(Bundle_X, DataBundle):
        raise TypeError("Bundle_X must be a DataBundle.")
    if not isinstance(Bundle_Y, DataBundle):
        raise TypeError("Bundle_Y must be a DataBundle.")
    if not isinstance(Bundle_Z, DataBundle):
        raise TypeError("Bundle_Z must be a DataBundle.")

    n = Bundle_Z.data.shape[0]

    if Bundle_X.data.shape[0] != n or Bundle_Y.data.shape[0] != n:
        raise ValueError(
            "Bundle_X, Bundle_Y, and Bundle_Z must have the same sample size."
        )

    if n < 2:
        raise ValueError("At least two observations are required.")

    if B <= 0:
        raise ValueError("B must be positive.")

    if k is None:
        k = max(10, int(np.ceil(np.sqrt(float(n)))))

    if k >= n:
        k = n - 1

    if k < 1:
        raise ValueError("k must be at least 1 after adjustment.")

    Z_data = Bundle_Z.data

    # ============================================================
    # Backend setup
    # ============================================================

    if isinstance(Z_data, torch.Tensor):

        device = Z_data.device
        dtype = Z_data.dtype

        if generator is None:
            generator = torch.Generator(device=device)
        elif not isinstance(generator, torch.Generator):
            raise TypeError(
                "For torch backend, generator must be torch.Generator or None."
            )

        T_perm = torch.empty(
            (B,),
            device=device,
            dtype=dtype,
        )

    elif isinstance(Z_data, np.ndarray):

        dtype = Z_data.dtype

        if generator is None:
            generator = np.random.default_rng()
        elif not isinstance(generator, np.random.Generator):
            raise TypeError(
                "For NumPy backend, generator must be np.random.Generator or None."
            )

        T_perm = np.empty(
            (B,),
            dtype=dtype,
        )

    else:
        raise TypeError("Bundle_Z.data must be np.ndarray or torch.Tensor.")

    # ============================================================
    # KNN construction
    # ============================================================

    knn_indices = build_knn_indices(
        Bundle_Z=Bundle_Z,
        k=k,
        include_self=False,
        batch_size=knn_batch_size,
    )

    # ============================================================
    # Observed statistic and reusable cache
    # ============================================================

    T_obs, backend_info, distance_dict, emdf_I_info = statistics_obs(
        Bundle_X=Bundle_X,
        Bundle_Y=Bundle_Y,
        Bundle_Z=Bundle_Z,
        **stat_kwargs,
    )

    T_obs_float = (
        float(T_obs.item())
        if isinstance(T_obs, torch.Tensor)
        else float(T_obs)
    )

    # ============================================================
    # Permutation statistics
    # ============================================================

    iterator = range(B)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Permutation", leave=False)

    for b in iterator:

        perm_index = local_permute_y(
            Bundle_Y=Bundle_Y,
            knn_indices=knn_indices,
            generator=generator,
            strategy=permutation_strategy,
        )

        Tb = statistics_perm(
            backend_info=backend_info,
            distance_dict=distance_dict,
            emdf_I_info=emdf_I_info,
            perm_index=perm_index,
        )

        if isinstance(T_perm, torch.Tensor):
            T_perm[b] = float(Tb)
        else:
            T_perm[b] = float(Tb)

    # ============================================================
    # P-value
    # ============================================================

    if isinstance(T_perm, torch.Tensor):

        p_value_tensor = (
            1.0 + torch.sum(T_perm >= T_obs_float).to(dtype=dtype)
        ) / (B + 1.0)

        p_value = float(p_value_tensor.item())

    else:

        p_value = float(
            (1.0 + np.sum(T_perm >= T_obs_float)) / (B + 1.0)
        )

    return {
        "T_obs": T_obs_float,
        "T_perm": T_perm,
        "p_value": p_value,
        "reject": bool(p_value <= alpha),
        "k": int(k),
        "knn_indices": knn_indices,
    }