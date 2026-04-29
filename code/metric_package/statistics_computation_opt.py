from scipy.linalg import expm
from scipy.stats import wishart
from metric_package.geometry_opt import (
    build_component_groups,
    stack_product_samples,
    emdf_product_pair_batch,
    compute_distance,
)
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal
import numpy as np
import torch
import time


ArrayLike = Union[np.ndarray, List[np.ndarray]]


# ============================================================
# Data generation
# ------------------------------------------------------------
# This section implements vectorized data-generating mechanisms
# for the simulation settings considered in the paper.
#
# The routines are kept in NumPy, since sample generation is not
# the primary computational bottleneck relative to the repeated
# evaluation of the test statistic and the resampling procedure.
# ============================================================


def generate_data(
    n: int,
    space_type: str = "euclidean",
    size: int = 2,
    rho: float = 0.0,
    seed: int | None = None,
    sigma_perm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Generate simulated samples ``(X, Y, Z)`` for conditional independence
    experiments on metric spaces.

    The data-generating mechanism depends on ``space_type`` and is constructed
    so that the dependence between ``X`` and ``Y`` conditional on ``Z`` is
    controlled by the parameter ``rho``. Specifically:

    - For ``space_type="euclidean"``, the variables are generated in
      Euclidean space with additive Gaussian perturbations.
    - For ``space_type="sphere"``, the variables are generated on the unit
      sphere by perturbing a base point ``Z`` along random tangent directions
      and projecting back to the sphere.
    - For ``space_type="spd"``, the variables are generated in the space of
      symmetric positive definite matrices through Wishart-type constructions.

    Parameters
    ----------
    n : int
        Sample size.

    space_type : {"euclidean", "sphere", "spd"}, default="euclidean"
        Type of metric space on which the data are generated.

    size : int, default=2
        Dimension parameter of the ambient space.

        - If ``space_type`` is ``"euclidean"`` or ``"sphere"``, then
          ``size`` is the ambient dimension ``d``.
        - If ``space_type`` is ``"spd"``, then ``size`` is the matrix size
          ``p``, so that the observations lie in the space of ``p x p``
          symmetric positive definite matrices.

    rho : float, default=0.0
        Dependence parameter controlling the conditional association between
        ``X`` and ``Y`` given ``Z``. The value should satisfy ``-1 <= rho <= 1``.

    seed : int or None, default=None
        Seed for the random number generator.

    sigma_perm : float, default=2.0
        Scale parameter for the perturbation magnitude in the Euclidean and
        spherical cases.

    Returns
    -------
    X, Y, Z : tuple of np.ndarray
        Generated samples.

        - If ``space_type`` is ``"euclidean"`` or ``"sphere"``, each returned
          array has shape ``(n, d)``.
        - If ``space_type`` is ``"spd"``, each returned array has shape
          ``(n, p, p)``.

    Raises
    ------
    ValueError
        If ``space_type`` is invalid, or if ``size < 2`` in the spherical case.

    Notes
    -----
    The implementation is vectorized over the sample index for computational
    efficiency. In the SPD case, Wishart-distributed matrices are generated
    through Gaussian matrix factors rather than repeated calls to a scalar
    Wishart sampler.
    """
    rng = np.random.default_rng(seed)
    rho_comp = np.sqrt(max(0.0, 1.0 - rho**2))

    # ------------------------------------------------------------
    # Case 1: Euclidean space
    # ------------------------------------------------------------
    # The latent variable Z is generated from a standard Gaussian law.
    # Conditional on Z, the variables X and Y are formed by adding Gaussian
    # perturbations whose correlation is governed by rho.
    # ------------------------------------------------------------
    if space_type == "euclidean":
        d = size
        sigma = sigma_perm

        Z = rng.normal(size=(n, d))
        U = rng.normal(size=(n, d))
        V = rng.normal(size=(n, d))

        eps_x = sigma * U
        eps_y = sigma * (rho * U + rho_comp * V)

        X = Z + eps_x
        Y = Z + eps_y

        return (
            np.asarray(X, dtype=float),
            np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float),
        )

    # ------------------------------------------------------------
    # Case 2: Unit sphere
    # ------------------------------------------------------------
    # A base point Z is first generated uniformly on the unit sphere by
    # normalizing a Gaussian vector. Tangent directions are then sampled by
    # projecting Gaussian vectors onto the tangent space at Z and normalizing.
    # Finally, X and Y are obtained by tangent perturbations followed by
    # renormalization onto the sphere.
    # ------------------------------------------------------------
    elif space_type == "sphere":
        d = size
        sigma = sigma_perm

        if d < 2:
            raise ValueError("For spherical data, size must be at least 2.")

        # Generate base points on the unit sphere.
        Z = rng.normal(size=(n, d))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)

        # Generate tangent directions for X.
        U = rng.normal(size=(n, d))
        U = U - np.sum(U * Z, axis=1, keepdims=True) * Z
        U_norm = np.linalg.norm(U, axis=1, keepdims=True)

        # Generate tangent directions for Y.
        V = rng.normal(size=(n, d))
        V = V - np.sum(V * Z, axis=1, keepdims=True) * Z
        V_norm = np.linalg.norm(V, axis=1, keepdims=True)

        # In rare degenerate cases, the projected vector may be numerically
        # close to zero. Such entries are resampled until a valid tangent
        # direction is obtained.
        bad_u = U_norm[:, 0] < 1e-14
        bad_v = V_norm[:, 0] < 1e-14

        while np.any(bad_u):
            U_new = rng.normal(size=(bad_u.sum(), d))
            Z_bad = Z[bad_u]
            U_new = U_new - np.sum(U_new * Z_bad, axis=1, keepdims=True) * Z_bad
            U[bad_u] = U_new
            U_norm = np.linalg.norm(U, axis=1, keepdims=True)
            bad_u = U_norm[:, 0] < 1e-14

        while np.any(bad_v):
            V_new = rng.normal(size=(bad_v.sum(), d))
            Z_bad = Z[bad_v]
            V_new = V_new - np.sum(V_new * Z_bad, axis=1, keepdims=True) * Z_bad
            V[bad_v] = V_new
            V_norm = np.linalg.norm(V, axis=1, keepdims=True)
            bad_v = V_norm[:, 0] < 1e-14

        U /= U_norm
        V /= V_norm

        # Generate correlated scalar perturbation magnitudes.
        eps1 = rng.normal(size=(n, 1))
        eps2 = rng.normal(size=(n, 1))

        xi_x = sigma * eps1
        xi_y = sigma * (rho * eps1 + rho_comp * eps2)

        # Perturb along tangent directions and project back to the sphere.
        X = Z + xi_x * U
        Y = Z + xi_y * V

        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)

        return (
            np.asarray(X, dtype=float),
            np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float),
        )

    # ------------------------------------------------------------
    # Case 3: SPD space
    # ------------------------------------------------------------
    # The base matrix Z is generated from a scaled Wishart law through a
    # Gaussian factor representation. Conditional perturbations are then
    # constructed via correlated Gaussian matrix factors, yielding SPD-valued
    # variables X and Y.
    # ------------------------------------------------------------
    elif space_type == "spd":
        p = size
        nu = p + 6

        # Generate Z ~ Wishart(nu, I_p) / nu via Gaussian factors.
        A = rng.normal(size=(n, nu, p))
        Z = np.matmul(np.transpose(A, (0, 2, 1)), A) / nu

        # Generate correlated Gaussian matrix factors.
        U = rng.normal(size=(n, nu, p))
        V = rng.normal(size=(n, nu, p))
        W = rho * U + rho_comp * V

        Sx = np.matmul(np.transpose(U, (0, 2, 1)), U) / nu
        Sy = np.matmul(np.transpose(W, (0, 2, 1)), W) / nu

        # Form X and Y by congruence transformation using the Cholesky factor
        # of Z, which preserves positive definiteness.
        Z_half = np.linalg.cholesky(Z)

        X = np.matmul(np.matmul(Z_half, Sx), np.transpose(Z_half, (0, 2, 1)))
        Y = np.matmul(np.matmul(Z_half, Sy), np.transpose(Z_half, (0, 2, 1)))

        return (
            np.asarray(X, dtype=float),
            np.asarray(Y, dtype=float),
            np.asarray(Z, dtype=float),
        )

    else:
        raise ValueError("space_type must be one of {'euclidean', 'sphere', 'spd'}.")


# ============================================================
# Conditional generators
# ------------------------------------------------------------
# This module provides conditional sample generators for
# metric-space models. The interface is designed to support both
# oracle generators and learned black-box models (e.g., neural
# conditional samplers).
#
# Each generator may implement either:
#   (i) a single-sample interface via ``__call__``, or
#   (ii) a batched interface via ``generate_batch``.
#
# This abstraction allows the downstream simulation and testing
# routines to remain agnostic to the underlying generation
# mechanism.
# ============================================================


def generate_conditional_samples(
    Z: np.ndarray,
    M: int,
    generators,
    space_type: str = "euclidean",
    seed: int | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Generate conditional samples for a batch of conditioning values.

    For each observed conditioning value ``Z_i``, this function generates
    ``M`` conditional sample pairs from the user-supplied generator
    ``generators``. The generator may either be a single-sample sampler
    accepting one conditioning value at a time, or a batched sampler
    implementing a ``generate_batch`` method.

    Parameters
    ----------
    Z : np.ndarray
        Batch of conditioning values.

        If the conditioning variable takes values in a Euclidean or spherical
        space, then ``Z`` typically has shape ``(n, d)``. If it takes values
        in an SPD space, then ``Z`` typically has shape ``(n, p, p)``.

    M : int
        Number of conditional replications generated for each observation.

    generators : callable or object
        Conditional generator. Two interfaces are supported:

        1. A callable with signature
           ``generators(z, space_type=..., rng=..., **kwargs)``,
           returning one sample pair ``(x, y)`` for a single conditioning
           value ``z``;

        2. An object implementing a method
           ``generate_batch(Z, M=..., space_type=..., rng=..., **kwargs)``,
           returning batched conditional samples directly.

        This design allows the generator to represent either an oracle sampler
        or a fitted black-box model, such as a neural conditional generator.

    space_type : {"euclidean", "sphere", "spd"}, default="euclidean"
        Metric-space type.

    seed : int or None, default=None
        Seed for the random number generator used in conditional sampling.

    **kwargs
        Additional keyword arguments passed to the generator.

    Returns
    -------
    X_all, Y_all : tuple of np.ndarray
        Arrays containing the generated conditional samples.

        If each generated sample has shape ``s``, then each returned array has
        shape ``(n, M) + s``.

    Notes
    -----
    This routine does not assume that the conditional generator admits a
    closed-form or vectorized representation. When a batched implementation is
    available through ``generate_batch``, it is used directly; otherwise the
    samples are generated by repeated calls to the single-sample generator.
    """
    rng = np.random.default_rng(seed)

    if hasattr(generators, "generate_batch") and callable(generators.generate_batch):
        return generators.generate_batch(
            Z,
            M=M,
            space_type=space_type,
            rng=rng,
            **kwargs,
        )

    X_all = []
    Y_all = []

    for z in Z:
        Xi = []
        Yi = []

        for _ in range(M):
            x, y = generators(z, space_type=space_type, rng=rng, **kwargs)
            Xi.append(x)
            Yi.append(y)

        X_all.append(np.stack(Xi, axis=0))
        Y_all.append(np.stack(Yi, axis=0))

    return np.stack(X_all, axis=0), np.stack(Y_all, axis=0)


# ============================================================
# Oracle product-form estimator
# ------------------------------------------------------------
# This section implements batched evaluations of the oracle
# product-form estimator ``\hat F_n^{M,\perp}``, which is one of
# the main computational components of the statistic.
#
# The implementation is designed so that the dominant batched
# distance and indicator calculations can be carried out on GPU
# when PyTorch/CUDA is used.
# ============================================================


def f_perp_gen_pair_batch(
    u_batch: Union[np.ndarray, torch.Tensor],
    v_batch: Union[np.ndarray, torch.Tensor],
    X_orc_array: Union[np.ndarray, torch.Tensor],
    Y_orc_array: Union[np.ndarray, torch.Tensor],
    Z_array: Union[np.ndarray, torch.Tensor],
    space_types: Tuple[
        Literal["euclidean", "sphere", "spd"],
        Literal["euclidean", "sphere", "spd"],
        Literal["euclidean", "sphere", "spd"],
    ] = ("euclidean", "euclidean", "euclidean"),
    atol: float = 1e-12,
    GPU: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Compute batched values of the oracle product-form estimator
    ``\hat F_n^{M,\perp}(u, v)``.

    For each query pair ``(u_b, v_b)``, this function evaluates
    \[
    \hat F_n^{M,\perp}(u_b, v_b)
    =
    \frac{1}{n}
    \sum_{l=1}^n
    \left[
    \frac{1}{M}\sum_{m=1}^M
    \delta_X(u_{b,x}, v_{b,x}, \tilde X_l^{(m)})
    \right]
    \left[
    \frac{1}{M}\sum_{m=1}^M
    \delta_Y(u_{b,y}, v_{b,y}, \tilde Y_l^{(m)})
    \right]
    \delta_Z(u_{b,z}, v_{b,z}, Z_l),
    \]
    where ``\tilde X_l^{(m)}`` and ``\tilde Y_l^{(m)}`` are oracle conditional
    samples generated at ``Z_l``.

    Parameters
    ----------
    u_batch, v_batch : np.ndarray or torch.Tensor
        Batched query points of shape ``(B, 3, ...)``. The inputs are assumed
        to be already consistent with the selected backend.

    X_orc_array, Y_orc_array : np.ndarray or torch.Tensor
        Oracle conditional samples of shape ``(n, M, ...)``.

    Z_array : np.ndarray or torch.Tensor
        Conditioning sample of shape ``(n, ...)``.

    space_types : tuple of 3 strings, default=("euclidean", "euclidean", "euclidean")
        Metric-space types for the ``X``, ``Y``, and ``Z`` components.

    atol : float, default=1e-12
        Absolute tolerance used in the indicator comparison.

    GPU : bool, default=False
        Whether the computation is carried out with PyTorch on CUDA.

    Returns
    -------
    np.ndarray or torch.Tensor
        A one-dimensional array/tensor of length ``B`` containing the
        oracle product-form estimates.

    Notes
    -----
    This routine assumes that all inputs have already been converted to the
    appropriate backend representation prior to the function call.
    """
    if u_batch.shape != v_batch.shape:
        raise ValueError(
            f"u_batch and v_batch must have the same shape, but got "
            f"{tuple(u_batch.shape)} and {tuple(v_batch.shape)}."
        )

    if u_batch.ndim < 3:
        raise ValueError(
            f"Expected u_batch and v_batch to have shape (B, 3, ...), "
            f"but got {tuple(u_batch.shape)}."
        )

    B = u_batch.shape[0]
    n = Z_array.shape[0]
    M = X_orc_array.shape[1]

    ux = u_batch[:, 0, ...]
    uy = u_batch[:, 1, ...]
    uz = u_batch[:, 2, ...]

    vx = v_batch[:, 0, ...]
    vy = v_batch[:, 1, ...]
    vz = v_batch[:, 2, ...]

    x_shape = tuple(ux.shape[1:])
    y_shape = tuple(uy.shape[1:])
    z_shape = tuple(uz.shape[1:])

    # ------------------------------------------------------------
    # X component
    # ------------------------------------------------------------
    if GPU:
        ux_exp = ux[:, None, None, ...].expand(B, n, M, *x_shape)
        vx_exp = vx[:, None, None, ...].expand(B, n, M, *x_shape)
        X_exp = X_orc_array[None, :, :, ...].expand(B, n, M, *x_shape)
    else:
        x_batch_shape = (B, n, M) + x_shape
        ux_exp = np.broadcast_to(ux[:, None, None, ...], x_batch_shape)
        vx_exp = np.broadcast_to(vx[:, None, None, ...], x_batch_shape)
        X_exp = np.broadcast_to(X_orc_array[None, :, :, ...], x_batch_shape)

    dux = compute_distance(
        ux_exp.reshape(B * n * M, *x_shape),
        X_exp.reshape(B * n * M, *x_shape),
        space_types[0],
        GPU=GPU,
    )
    duv_x = compute_distance(
        ux_exp.reshape(B * n * M, *x_shape),
        vx_exp.reshape(B * n * M, *x_shape),
        space_types[0],
        GPU=GPU,
    )

    # ------------------------------------------------------------
    # Y component
    # ------------------------------------------------------------
    if GPU:
        uy_exp = uy[:, None, None, ...].expand(B, n, M, *y_shape)
        vy_exp = vy[:, None, None, ...].expand(B, n, M, *y_shape)
        Y_exp = Y_orc_array[None, :, :, ...].expand(B, n, M, *y_shape)
    else:
        y_batch_shape = (B, n, M) + y_shape
        uy_exp = np.broadcast_to(uy[:, None, None, ...], y_batch_shape)
        vy_exp = np.broadcast_to(vy[:, None, None, ...], y_batch_shape)
        Y_exp = np.broadcast_to(Y_orc_array[None, :, :, ...], y_batch_shape)

    duy = compute_distance(
        uy_exp.reshape(B * n * M, *y_shape),
        Y_exp.reshape(B * n * M, *y_shape),
        space_types[1],
        GPU=GPU,
    )
    duv_y = compute_distance(
        uy_exp.reshape(B * n * M, *y_shape),
        vy_exp.reshape(B * n * M, *y_shape),
        space_types[1],
        GPU=GPU,
    )

    # ------------------------------------------------------------
    # Z component
    # ------------------------------------------------------------
    if GPU:
        uz_exp = uz[:, None, ...].expand(B, n, *z_shape)
        vz_exp = vz[:, None, ...].expand(B, n, *z_shape)
        Z_exp = Z_array[None, :, ...].expand(B, n, *z_shape)
    else:
        z_batch_shape = (B, n) + z_shape
        uz_exp = np.broadcast_to(uz[:, None, ...], z_batch_shape)
        vz_exp = np.broadcast_to(vz[:, None, ...], z_batch_shape)
        Z_exp = np.broadcast_to(Z_array[None, :, ...], z_batch_shape)

    duz = compute_distance(
        uz_exp.reshape(B * n, *z_shape),
        Z_exp.reshape(B * n, *z_shape),
        space_types[2],
        GPU=GPU,
    )
    duv_z = compute_distance(
        uz_exp.reshape(B * n, *z_shape),
        vz_exp.reshape(B * n, *z_shape),
        space_types[2],
        GPU=GPU,
    )

    # ------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------
    if GPU:
        dx = (dux <= duv_x + atol).reshape(B, n, M).to(torch.float32)
        dy = (duy <= duv_y + atol).reshape(B, n, M).to(torch.float32)
        dz = (duz <= duv_z + atol).reshape(B, n).to(torch.float32)

        tx = dx.mean(dim=2)
        ty = dy.mean(dim=2)
        return (tx * ty * dz).mean(dim=1)

    dx = (dux <= duv_x + atol).reshape(B, n, M).astype(float)
    dy = (duy <= duv_y + atol).reshape(B, n, M).astype(float)
    dz = (duz <= duv_z + atol).reshape(B, n).astype(float)

    tx = dx.mean(axis=2)
    ty = dy.mean(axis=2)
    return (tx * ty * dz).mean(axis=1)


# ============================================================
# Test statistic
# ------------------------------------------------------------
# This section implements the main test statistic used in the
# conditional independence procedure. The computation is carried
# out over all ordered pairs ``(i, j)`` with ``i != j``, processed
# in batches for memory efficiency.
#
# The statistic routine also serves as the entry point to the
# GPU computation path: when GPU acceleration is requested,
# NumPy inputs are converted internally to CUDA tensors, and
# the dominant batched evaluations of ``\hat F_n^{\mathcal M}``
# and ``\hat F_n^{M,\perp}`` are then carried out through PyTorch.
# ============================================================


def statistics(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    X_orc_array: np.ndarray,
    Y_orc_array: np.ndarray,
    space_types: Tuple[str, str, str],
    GPU: bool = False,
    batch_size: int = 1024,
) -> float:
    r"""
    Compute the statistic
    \[
    T_n^{\mathrm{gen}}
    =
    \frac{1}{n(n-1)}
    \sum_{i \ne j}
    \left[
    \hat F_n^{\mathcal M}(u_i, u_j)
    -
    \hat F_n^{M,\perp}(u_i, u_j)
    \right]^2.
    \]

    Parameters
    ----------
    X, Y, Z : np.ndarray
        Observed samples in the three component spaces.

    X_orc_array, Y_orc_array : np.ndarray
        Oracle conditional samples of shape ``(n, M, ...)``, where ``M`` denotes
        the number of generated samples per conditioning value.

    space_types : tuple[str, str, str]
        Metric-space types for the three components ``(X, Y, Z)``. These determine
        the geometry used in the evaluation of both
        ``\hat F_n^{\mathcal M}`` and ``\hat F_n^{M,\perp}``.

    GPU : bool, default=False
        If ``True``, all inputs are converted to CUDA tensors at the beginning of
        the routine, and subsequent batched computations are carried out using
        PyTorch on the GPU. Otherwise, computations are performed using NumPy.

    batch_size : int, default=1024
        Number of ordered pairs ``(i, j)`` processed in each batch.

    Returns
    -------
    float
        The value of the test statistic.

    Notes
    -----
    This routine serves as the main computational entry point of the test statistic.
    All inputs are expected to be NumPy arrays; if GPU acceleration is requested,
    they are converted internally to CUDA tensors. The computation over all ordered
    pairs ``(i, j)`` with ``i \ne j`` is carried out in batches for memory efficiency.

    The grouping structure required for evaluating
    ``\hat F_n^{\mathcal M}`` is constructed internally from ``space_types``.
    """
    n = len(X)
    if n < 2:
        raise ValueError("At least two observations are required.")
    
    if GPU:
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32)
        Y = torch.as_tensor(Y, device="cuda", dtype=torch.float32)
        Z = torch.as_tensor(Z, device="cuda", dtype=torch.float32)
        X_orc_array = torch.as_tensor(X_orc_array, device="cuda", dtype=torch.float32)
        Y_orc_array = torch.as_tensor(Y_orc_array, device="cuda", dtype=torch.float32)

    component_groups = build_component_groups(space_types)

    S = stack_product_samples(X, Y, Z, GPU=GPU)
    total_pairs = n * (n - 1)
    total = 0.0

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)
        bsz = end - start

        if GPU:
            k = torch.arange(start, end, device=S.device, dtype=torch.long)
            ib = torch.div(k, n - 1, rounding_mode="floor")
            r = torch.remainder(k, n - 1)
            jb = r + (r >= ib).to(torch.long)

            u_batch = S[ib]
            v_batch = S[jb]
        else:
            k = np.arange(start, end, dtype=np.int64)
            ib = k // (n - 1)
            r = k % (n - 1)
            jb = r + (r >= ib).astype(np.int64)

            u_batch = S[ib]
            v_batch = S[jb]

        emdf_P = emdf_product_pair_batch(
            xyz_samples=S,
            u_batch=u_batch,
            v_batch=v_batch,
            component_groups=component_groups,
            atol=1e-12,
            GPU=GPU,
        )

        emdf_I = f_perp_gen_pair_batch(
            u_batch=u_batch,
            v_batch=v_batch,
            X_orc_array=X_orc_array,
            Y_orc_array=Y_orc_array,
            Z_array=Z,
            space_types=space_types,
            atol=1e-12,
            GPU=GPU,
        )

        if GPU:
            diff = emdf_P - emdf_I
            total += float((diff * diff).sum().item())
        else:
            diff = emdf_P - emdf_I
            total += float(np.sum(diff * diff))

    return total / (n * (n - 1))


# ============================================================
# Local permutation
# ------------------------------------------------------------
# This part is kept mostly on CPU.
# Neighbor search and permutation logic are not the main bottleneck.
# ============================================================


def compute_pairwise_distances(
    Z: np.ndarray,
    z_space_type: Literal["euclidean", "sphere", "spd"],
) -> np.ndarray:
    r"""
    Compute the pairwise distance matrix on a supported metric space.

    Parameters
    ----------
    Z : np.ndarray
        Input sample.

        - For Euclidean and spherical spaces: shape ``(n, d)``;
        - For SPD space: shape ``(n, p, p)``.

    z_space_type : {"euclidean", "sphere", "spd"}
        Metric-space type of the sample.

    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape ``(n, n)``.

    Raises
    ------
    ValueError
        If the input shape is incompatible with ``z_space_type``, or if the
        sample is empty.

    Notes
    -----
    The implementation uses fully vectorized formulas for Euclidean and
    spherical spaces. For the SPD case equipped with the Cholesky distance,
    the computation is reduced to a Euclidean distance matrix after mapping
    each SPD matrix to its Cholesky factor.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]

    if n == 0:
        raise ValueError("Z must be non-empty.")

    # ------------------------------------------------------------
    # Case 1: Euclidean space
    # ------------------------------------------------------------
    if z_space_type == "euclidean":
        if Z.ndim != 2:
            raise ValueError(
                f"For Euclidean space, expected Z.shape = (n, d), but got {Z.shape}."
            )

        sq_norms = np.sum(Z * Z, axis=1, keepdims=True)
        D2 = sq_norms + sq_norms.T - 2.0 * (Z @ Z.T)
        D2 = np.maximum(D2, 0.0)

        return np.sqrt(D2)

    # ------------------------------------------------------------
    # Case 2: Spherical space
    # ------------------------------------------------------------
    if z_space_type == "sphere":
        if Z.ndim != 2:
            raise ValueError(
                f"For sphere space, expected Z.shape = (n, d), but got {Z.shape}."
            )

        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("Sphere inputs must have nonzero norm.")

        Z_unit = Z / norms
        cos_theta = Z_unit @ Z_unit.T
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.arccos(cos_theta)

    # ------------------------------------------------------------
    # Case 3: SPD space
    # ------------------------------------------------------------
    if z_space_type == "spd":
        if Z.ndim != 3 or Z.shape[1] != Z.shape[2]:
            raise ValueError(
                f"For SPD space, expected Z.shape = (n, p, p), but got {Z.shape}."
            )

        # Cholesky factors: shape (n, p, p)
        L = np.linalg.cholesky(Z)

        # Flatten each factor into a vector in R^(p^2)
        L_flat = L.reshape(n, -1)

        # Frobenius distances between Cholesky factors
        sq_norms = np.sum(L_flat * L_flat, axis=1, keepdims=True)
        D2 = sq_norms + sq_norms.T - 2.0 * (L_flat @ L_flat.T)
        D2 = np.maximum(D2, 0.0)

        return np.sqrt(D2)

    raise ValueError("z_space_type must be one of {'euclidean', 'sphere', 'spd'}.")


def build_knn_indices(
    Z: Union[np.ndarray, list[np.ndarray]],
    k: int,
    z_space_type: Literal["euclidean", "sphere", "spd"],
    include_self: bool = True,
) -> np.ndarray:
    r"""
    Construct k-nearest-neighbor index sets in the Z-space.

    Parameters
    ----------
    Z : np.ndarray or list[np.ndarray]
        Sample in the conditioning space.

    k : int
        Number of nearest neighbors to retain.

    z_space_type : {"euclidean", "sphere", "spd"}
        Metric-space type of the conditioning variable.

    include_self : bool, default=True
        Whether each point is allowed to include itself among its neighbors.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, k)`` whose ``i``-th row contains the indices of
        the ``k`` nearest neighbors of the ``i``-th point.

    Notes
    -----
    The pairwise distance matrix is first computed in full. Neighbor selection
    is then carried out row-wise using partial sorting, which is more efficient
    than a full sort when only the first ``k`` nearest neighbors are needed.
    """
    D = compute_pairwise_distances(Z, z_space_type)
    n = D.shape[0]

    if k < 1:
        raise ValueError("k must be at least 1.")

    if include_self:
        if k > n:
            raise ValueError(
                f"When include_self=True, k={k} cannot exceed sample size n={n}."
            )
    else:
        if k >= n:
            raise ValueError(
                f"When include_self=False, k={k} must be smaller than n={n}, "
                "since each point has at most n-1 non-self neighbors."
            )

    if include_self:
        # Partial sort: extract the k smallest entries in each row.
        cand = np.argpartition(D, kth=k - 1, axis=1)[:, :k]

        # Sort these k candidates by their actual distances.
        row_idx = np.arange(n)[:, None]
        cand_dist = D[row_idx, cand]
        order = np.argsort(cand_dist, axis=1)

        return cand[row_idx, order]

    # Exclude self by setting diagonal to +inf.
    D_work = D.copy()
    np.fill_diagonal(D_work, np.inf)

    cand = np.argpartition(D_work, kth=k - 1, axis=1)[:, :k]

    row_idx = np.arange(n)[:, None]
    cand_dist = D_work[row_idx, cand]
    order = np.argsort(cand_dist, axis=1)

    return cand[row_idx, order]


# def local_permute_y(
#     Y: np.ndarray,
#     knn_indices: np.ndarray,
#     rng: Optional[np.random.Generator] = None,
#     strategy: str = "resample",
# ) -> np.ndarray:
#     r"""
#     Construct a locally permuted version of Y.

#     Parameters
#     ----------
#     Y : np.ndarray
#         Sample to be locally permuted. The first dimension is interpreted as
#         the sample index.

#     knn_indices : np.ndarray
#         Array of shape (n, k) whose i-th row contains the admissible
#         neighbor indices associated with the i-th observation.

#     rng : np.random.Generator or None, default=None
#         Random number generator. If None, a default generator is used.

#     strategy : {"resample", "shuffle_once"}, default="resample"
#         Local permutation strategy.

#     Returns
#     -------
#     np.ndarray
#         Locally permuted version of Y with the same shape as Y.
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     n = Y.shape[0]

#     if knn_indices.ndim != 2:
#         raise ValueError(
#             f"knn_indices must have shape (n, k), but got {knn_indices.shape}."
#         )

#     if knn_indices.shape[0] != n:
#         raise ValueError("knn_indices and Y must have the same first dimension.")

#     if knn_indices.shape[1] == 0:
#         raise ValueError("Each row of knn_indices must contain at least one neighbor.")

#     if strategy not in {"resample", "shuffle_once"}:
#         raise ValueError("strategy must be one of {'resample', 'shuffle_once'}.")

#     # ------------------------------------------------------------
#     # Strategy 1: independent local resampling
#     # ------------------------------------------------------------
#     if strategy == "resample":
#         k = knn_indices.shape[1]
#         chosen = knn_indices[np.arange(n), rng.integers(0, k, size=n)]
#         return Y[chosen].copy()

#     # ------------------------------------------------------------
#     # Strategy 2: one-pass constrained shuffle with fallback
#     # ------------------------------------------------------------
#     perm = rng.permutation(n)

#     # rank[j] = position of index j in perm
#     rank = np.empty(n, dtype=int)
#     rank[perm] = np.arange(n)

#     available = np.ones(n, dtype=bool)
#     chosen = np.empty(n, dtype=int)

#     for i in range(n):
#         neigh = knn_indices[i]
#         feasible = neigh[available[neigh]]

#         if feasible.size > 0:
#             selected = feasible[np.argmin(rank[feasible])]
#             chosen[i] = selected
#             available[selected] = False
#         else:
#             chosen[i] = neigh[rng.integers(0, len(neigh))]

#     return Y[chosen].copy()


def local_permute_y(
    Y: np.ndarray,
    knn_indices: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "resample",
) -> np.ndarray:
    r"""
    Construct a locally permuted version of ``Y``.

    Parameters
    ----------
    Y : np.ndarray
        Sample to be locally permuted. The first dimension is interpreted as
        the sample index.

    knn_indices : np.ndarray
        Array of shape ``(n, k)`` whose ``i``-th row contains the admissible
        neighbor indices associated with the ``i``-th observation.

    rng : np.random.Generator or None, default=None
        Random number generator. If ``None``, a default generator is used.

    strategy : {"resample", "shuffle_once"}, default="resample"
        Local permutation strategy.

    Returns
    -------
    np.ndarray
        Locally permuted version of ``Y`` with the same shape as ``Y``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = Y.shape[0]

    if knn_indices.ndim != 2:
        raise ValueError(
            f"knn_indices must have shape (n, k), but got {knn_indices.shape}."
        )

    if knn_indices.shape[0] != n:
        raise ValueError("knn_indices and Y must have the same first dimension.")

    if knn_indices.shape[1] == 0:
        raise ValueError("Each row of knn_indices must contain at least one neighbor.")

    if strategy not in {"resample", "shuffle_once"}:
        raise ValueError("strategy must be one of {'resample', 'shuffle_once'}.")

    # ------------------------------------------------------------
    # Strategy 1: independent local resampling
    # ------------------------------------------------------------
    if strategy == "resample":
        k = knn_indices.shape[1]
        chosen = knn_indices[np.arange(n), rng.integers(0, k, size=n)]
        return Y[chosen].copy()

    # ------------------------------------------------------------
    # Strategy 2: one-pass constrained shuffle with fallback
    # ------------------------------------------------------------
    perm = rng.permutation(n)

    rank = np.empty(n, dtype=int)
    rank[perm] = np.arange(n)

    available = np.ones(n, dtype=bool)
    chosen = np.empty(n, dtype=int)

    last_selected: Optional[int] = None

    for i in range(n):
        neigh = knn_indices[i]
        feasible = neigh[available[neigh]]

        if feasible.size > 0:
            selected = feasible[np.argmin(rank[feasible])]
            chosen[i] = selected
            available[selected] = False
            last_selected = selected
        else:
            if last_selected is not None:
                chosen[i] = last_selected
            else:
                # 极端情形：第一次就无可用邻居时，退回本地随机选择
                chosen[i] = neigh[rng.integers(0, len(neigh))]

    return Y[chosen].copy()


def local_permutation_test(
    X,
    Y,
    Z,
    stat_fn: Callable[..., float],
    stat_kwargs: Optional[Dict] = None,
    B: int = 500,
    k: Optional[int] = None,
    z_space_type: str = "euclidean",
    permutation_strategy: str = "resample",
    alpha: float = 0.05,
    seed: Optional[int] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """
    Generic local permutation test core.

    Parameters
    ----------
    X, Y, Z : array-like
        Observed samples.

    stat_fn : callable
        Function that computes the test statistic from X, Y, Z.
        Expected signature:
            stat_fn(X, Y, Z, **stat_kwargs) -> float

    stat_kwargs : dict, optional
        Extra keyword arguments passed to stat_fn.

    B : int, default=500
        Number of local permutation replicates.

    k : int, optional
        Neighborhood size. If None, use max(10, ceil(sqrt(n))).

    z_space_type : str, default="euclidean"
        One of {"euclidean", "sphere", "spd"}.

    permutation_strategy : str, default="resample"
        One of {"resample", "shuffle_once"}.

    alpha : float, default=0.05
        Significance level.

    seed : int, optional
        Random seed.

    show_progress : bool, default=False
        Whether to display a tqdm progress bar for the permutation loop.

    Returns
    -------
    result : dict
        {
            "T_obs": float,
            "T_perm": np.ndarray of shape (B,),
            "p_value": float,
            "reject": bool,
            "k": int,
            "knn_indices": np.ndarray
        }
    """
    if stat_kwargs is None:
        stat_kwargs = {}

    rng = np.random.default_rng(seed)
    n = len(Z)

    if k is None:
        k = max(10, int(np.ceil(np.sqrt(n))))

    # 与 build_knn_indices 的约束保持一致
    if k >= n:
        k = n - 1

    knn_indices = build_knn_indices(
        Z=Z,
        k=k,
        z_space_type=z_space_type,
        include_self=False,
    )

    T_obs = stat_fn(X, Y, Z, **stat_kwargs)

    T_perm = np.zeros(B, dtype=float)

    iterator = range(B)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Permutation", leave=False)

    for b in iterator:
        Y_star = local_permute_y(
            Y=Y,
            knn_indices=knn_indices,
            rng=rng,
            strategy=permutation_strategy,
        )
        T_perm[b] = stat_fn(X, Y_star, Z, **stat_kwargs)

    p_value = (1.0 + np.sum(T_perm >= T_obs)) / (B + 1.0)

    return {
        "T_obs": float(T_obs),
        "T_perm": T_perm,
        "p_value": float(p_value),
        "reject": bool(p_value <= alpha),
        "k": int(k),
        "knn_indices": knn_indices,
    }


def local_permutation_test_time(
    X,
    Y,
    Z,
    stat_fn: Callable[..., float],
    stat_kwargs: Optional[Dict] = None,
    B: int = 500,
    k: Optional[int] = None,
    z_space_type: str = "euclidean",
    permutation_strategy: str = "resample",
    alpha: float = 0.05,
    seed: Optional[int] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """
    Generic local permutation test core.

    Parameters
    ----------
    X, Y, Z : array-like
        Observed samples.

    stat_fn : callable
        Function that computes the test statistic from X, Y, Z.
        Expected signature:
            stat_fn(X, Y, Z, **stat_kwargs) -> float

    stat_kwargs : dict, optional
        Extra keyword arguments passed to stat_fn.

    B : int, default=500
        Number of local permutation replicates.

    k : int, optional
        Neighborhood size. If None, use max(10, ceil(sqrt(n))).

    z_space_type : str, default="euclidean"
        One of {"euclidean", "sphere", "spd"}.

    permutation_strategy : str, default="resample"
        One of {"resample", "shuffle_once"}.

    alpha : float, default=0.05
        Significance level.

    seed : int, optional
        Random seed.

    show_progress : bool, default=False
        Whether to display a tqdm progress bar for the permutation loop.

    Returns
    -------
    result : dict
        {
            "T_obs": float,
            "T_perm": np.ndarray of shape (B,),
            "p_value": float,
            "reject": bool,
            "k": int,
            "knn_indices": np.ndarray
        }
    """

    start = time.time()

    if stat_kwargs is None:
        stat_kwargs = {}

    rng = np.random.default_rng(seed)
    n = len(Z)

    if k is None:
        k = max(10, int(np.ceil(np.sqrt(n))))

    # 与 build_knn_indices 的约束保持一致
    if k >= n:
        k = n - 1

    knn_indices = build_knn_indices(
        Z=Z,
        k=k,
        z_space_type=z_space_type,
        include_self=False,
    )

    T_start = time.time()
    T_obs = stat_fn(X, Y, Z, **stat_kwargs)
    T_end = time.time()
    T_elapsed = T_end - T_start
    permutation_elapsed = 0.0

    T_perm = np.zeros(B, dtype=float)

    iterator = range(B)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Permutation", leave=False)

    for b in iterator:
        permutation_start = time.time()
        Y_star = local_permute_y(
            Y=Y,
            knn_indices=knn_indices,
            rng=rng,
            strategy=permutation_strategy,
        )
        permutation_end = time.time()
        permutation_elapsed = permutation_end - permutation_start
        T_start = time.time()
        T_perm[b] = stat_fn(X, Y_star, Z, **stat_kwargs)
        T_end = time.time()
        T_elapsed += T_end - T_start
        permutation_elapsed += permutation_end - permutation_start

    p_value = (1.0 + np.sum(T_perm >= T_obs)) / (B + 1.0)

    end = time.time()
    elapsed = end - start
    print(f"Local permutation test completed in {elapsed:.2f} seconds.")
    print(f"Total time for statistic computation: {T_elapsed:.2f} seconds.")
    print(f"Total time for permutation: {permutation_elapsed:.2f} seconds.")

    return {
        "T_obs": float(T_obs),
        "T_perm": T_perm,
        "p_value": float(p_value),
        "reject": bool(p_value <= alpha),
        "k": int(k),
        "knn_indices": knn_indices,
    }

