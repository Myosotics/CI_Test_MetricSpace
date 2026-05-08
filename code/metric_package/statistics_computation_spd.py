import torch
from dataclasses import dataclass
import torch
from typing import Callable, Optional, Dict


# ============================================================
# Data Bundle
# ------------------------------------------------------------
# This section implements vectorized data-generating mechanisms
# for the simulation settings considered in the paper.
#
# The routines are kept in NumPy, since sample generation is not
# the primary computational bottleneck relative to the repeated
# evaluation of the test statistic and the resampling procedure.
# ============================================================


@dataclass
class SPDDataBundle:
    r"""
    Precomputed representation for SPD matrices with arbitrary batch shape.

    Attributes
    ----------
    matrix : torch.Tensor
        Original SPD matrices.

        Shape:
        - ``(*batch_shape, p, p)``

    inv_half : torch.Tensor
        Inverse square roots of the SPD matrices.

        Shape:
        - ``(*batch_shape, p, p)``

    eigvals : torch.Tensor
        Eigenvalues of the SPD matrices.

        Shape:
        - ``(*batch_shape, p)``

    eigvecs : torch.Tensor
        Eigenvectors of the SPD matrices.

        Shape:
        - ``(*batch_shape, p, p)``

    cholesky : torch.Tensor
        Lower-triangular Cholesky factors of the SPD matrices.

        Shape:
        - ``(*batch_shape, p, p)``
    """
    matrix: torch.Tensor
    inv_half: torch.Tensor
    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    cholesky: torch.Tensor


def Bundle_spd(
    X: torch.Tensor,
    atol: float = 1e-12,
) -> SPDDataBundle:
    r"""
    Prepare a precomputed bundle for SPD matrices on GPU.

    This function accepts SPD matrices with arbitrary batch shape and returns
    a structured object containing the matrices together with several
    quantities frequently used in geometric computations.

    Parameters
    ----------
    X : torch.Tensor
        Input SPD matrices on GPU.

        Shape:
        - ``(p, p)``, or
        - ``(*batch_shape, p, p)``

        If the input has shape (p, p), it is promoted to (1, p, p).

    atol : float, default=1e-12
        Lower bound applied to eigenvalues for numerical stability.

    Returns
    -------
    SPDBundle
        Structured bundle containing:

        - ``matrix`` : ``(*batch_shape, p, p)``
        - ``inv_half`` : ``(*batch_shape, p, p)``
        - ``eigvals`` : ``(*batch_shape, p)``
        - ``eigvecs`` : ``(*batch_shape, p, p)``
        - ``cholesky`` : ``(*batch_shape, p, p)``
    """
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor.")

    if not X.is_cuda:
        raise RuntimeError("X must be a CUDA tensor.")

    if X.ndim < 2:
        raise ValueError("X must have shape (p, p) or (*batch_shape, p, p).")

    if X.shape[-1] != X.shape[-2]:
        raise ValueError(
            f"Each SPD matrix must be square, but got shape {tuple(X.shape)}."
        )

    # eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(X)
    eigvals = torch.clamp(eigvals, min=atol)

    # inverse square root
    inv_sqrt = 1.0 / torch.sqrt(eigvals)
    inv_half = torch.matmul(
        eigvecs * inv_sqrt[..., None, :],
        eigvecs.transpose(-1, -2),
    )

    # Cholesky factor
    cholesky = torch.linalg.cholesky(X)

    return SPDDataBundle(
        matrix=X,
        inv_half=inv_half,
        eigvals=eigvals,
        eigvecs=eigvecs,
        cholesky=cholesky,
    )


def slice_spd_bundle(
    bundle: SPDDataBundle,
    idx: torch.Tensor,
) -> SPDDataBundle:
    r"""
    Slice an SPDDataBundle along the first dimension.

    Parameters
    ----------
    bundle : SPDDataBundle
        Input bundle with leading dimension ``(N, ...)``.

    idx : torch.Tensor
        Index tensor of shape ``(B,)`` with dtype ``torch.long``.

    Returns
    -------
    SPDDataBundle
        Sliced bundle with leading dimension ``(B, ...)``.

    Notes
    -----
    This function performs advanced indexing on each field of the bundle.
    No explicit data copying is performed beyond what PyTorch indexing requires.
    All returned tensors share the same device and dtype as the input bundle.
    """
    if not isinstance(bundle, SPDDataBundle):
        raise TypeError("bundle must be an SPDDataBundle.")

    if not isinstance(idx, torch.Tensor):
        raise TypeError("idx must be a torch.Tensor.")

    if idx.dtype != torch.long:
        raise ValueError("idx must have dtype torch.long.")

    return SPDDataBundle(
        matrix=bundle.matrix[idx],
        inv_half=bundle.inv_half[idx],
        eigvals=bundle.eigvals[idx],
        eigvecs=bundle.eigvecs[idx],
        cholesky=bundle.cholesky[idx],
    )


def stack_spd_triplet(
    Bundle_X: SPDDataBundle,
    Bundle_Y: SPDDataBundle,
    Bundle_Z: SPDDataBundle,
) -> SPDDataBundle:
    r"""
    Stack three SPD bundles into a single 3-component product-space bundle.

    Parameters
    ----------
    Bundle_X, Bundle_Y, Bundle_Z : SPDDataBundle
        Input bundles with matrix shape ``(n, p, p)``.

    Returns
    -------
    SPDDataBundle
        Combined bundle with matrix shape ``(n, 3, p, p)``.
    """
    if Bundle_X.matrix.shape != Bundle_Y.matrix.shape or Bundle_X.matrix.shape != Bundle_Z.matrix.shape:
        raise ValueError("Bundle_X, Bundle_Y, and Bundle_Z must have identical shapes.")

    return SPDDataBundle(
        matrix=torch.stack([Bundle_X.matrix, Bundle_Y.matrix, Bundle_Z.matrix], dim=1),
        inv_half=torch.stack([Bundle_X.inv_half, Bundle_Y.inv_half, Bundle_Z.inv_half], dim=1),
        eigvals=torch.stack([Bundle_X.eigvals, Bundle_Y.eigvals, Bundle_Z.eigvals], dim=1),
        eigvecs=torch.stack([Bundle_X.eigvecs, Bundle_Y.eigvecs, Bundle_Z.eigvecs], dim=1),
        cholesky=torch.stack([Bundle_X.cholesky, Bundle_Y.cholesky, Bundle_Z.cholesky], dim=1),
    )


# ============================================================
# Distance Computation
# ------------------------------------------------------------
# This section implements vectorized data-generating mechanisms
# for the simulation settings considered in the paper.
#
# The routines are kept in NumPy, since sample generation is not
# the primary computational bottleneck relative to the repeated
# evaluation of the test statistic and the resampling procedure.
# ============================================================


def spd_affine_invariant_distance_from_inv_half(
    P1_inv_half: torch.Tensor,
    P2_matrix: torch.Tensor,
    atol: float = 1e-12,
) -> torch.Tensor:
    r"""
    Compute affine-invariant SPD distances from precomputed inverse square roots.

    Parameters
    ----------
    P1_inv_half : torch.Tensor
        Inverse square roots of the first SPD argument.

        Shape:
        - ``(p, p)``, or
        - ``(*batch_shape, p, p)``

    P2_matrix : torch.Tensor
        SPD matrices of the second argument.

        Must have the same shape as ``P1_inv_half``.

    atol : float, default=1e-12
        Lower bound applied to eigenvalues for numerical stability.

    Returns
    -------
    torch.Tensor
        Affine-invariant distances.

        Shape:
        - ``(1,)`` if inputs are ``(p, p)``
        - ``batch_shape`` if inputs are ``(*batch_shape, p, p)``
    """
    if P1_inv_half.shape != P2_matrix.shape:
        raise ValueError(
            f"Shape mismatch: P1_inv_half has shape {tuple(P1_inv_half.shape)}, "
            f"but P2_matrix has shape {tuple(P2_matrix.shape)}."
        )

    if P1_inv_half.ndim == 2:
        P1_inv_half = P1_inv_half.unsqueeze(0)
        P2_matrix = P2_matrix.unsqueeze(0)

    p = P1_inv_half.shape[-1]

    A = P1_inv_half
    B = P2_matrix

    G = A @ B @ A
    G = 0.5 * (G + G.transpose(-1, -2))

    if p == 2:
        a = G[..., 0, 0]
        b = G[..., 0, 1]
        c = G[..., 1, 1]

        tr_half = 0.5 * (a + c)
        rad = torch.sqrt(
            torch.clamp((0.5 * (a - c)) ** 2 + b ** 2, min=0.0)
        )

        lam1 = torch.clamp(tr_half - rad, min=atol)
        lam2 = torch.clamp(tr_half + rad, min=atol)

        return torch.sqrt(torch.log(lam1) ** 2 + torch.log(lam2) ** 2)

    lam = torch.linalg.eigvalsh(G)
    lam = torch.clamp(lam, min=atol)

    return torch.sqrt(torch.sum(torch.log(lam) ** 2, dim=-1))


def spd_affine_invariant_distance(
    Bundle_P1: SPDDataBundle,
    Bundle_P2: SPDDataBundle,
    atol: float = 1e-12,
) -> torch.Tensor:
    return spd_affine_invariant_distance_from_inv_half(
        P1_inv_half=Bundle_P1.inv_half,
        P2_matrix=Bundle_P2.matrix,
        atol=atol,
    )


# ============================================================
# 工具函数
# ============================================================


def broadcast_spd_pairs(
    X: torch.Tensor,
    mode: str,
    rep: int | None = None,
) -> torch.Tensor:
    r"""
    Broadcast batched matrices/tensors to a repeated pairwise-like structure
    without materializing copies.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape:

        - ``(n, a, b)``, or
        - ``(*batch_shape, n, a, b)``

    mode : {"ij", "ji"}
        Broadcasting pattern.

        - ``"ij"``:
          Each sample ``X_i`` is expanded along a new axis after ``n``.
          The output satisfies:
              out[..., i, j] = X[..., i]
          Shape:
              ``(*batch_shape, n, rep, a, b)``

        - ``"ji"``:
          The full batch is expanded along a new axis before ``n``.
          The output satisfies:
              out[..., i, j] = X[..., j]
          Shape:
              ``(*batch_shape, rep, n, a, b)``

    rep : int or None, default=None
        Number of repetitions along the broadcasted dimension.
        If ``None``, defaults to ``n``.

    Returns
    -------
    torch.Tensor
        Broadcasted tensor:

        - if ``mode="ij"``:
            shape ``(*batch_shape, n, rep, a, b)``
        - if ``mode="ji"``:
            shape ``(*batch_shape, rep, n, a, b)``

        The returned tensor is a view created via ``expand`` and
        does not allocate new memory.
    """

    if X.ndim < 3:
        raise ValueError("X must have shape (n, a, b) or (*batch_shape, n, a, b).")

    *batch_shape, n, a, b = X.shape

    if rep is None:
        rep = n

    if rep <= 0:
        raise ValueError("rep must be a positive integer.")

    if mode == "ij":
        # (..., n, a, b) -> (..., n, 1, a, b) -> (..., n, rep, a, b)
        return X.unsqueeze(-3).expand(*batch_shape, n, rep, a, b)

    if mode == "ji":
        # (..., n, a, b) -> (..., 1, n, a, b) -> (..., rep, n, a, b)
        return X.unsqueeze(-4).expand(*batch_shape, rep, n, a, b)

    raise ValueError("mode must be 'ij' or 'ji'")


def delta_product(
    u_inv_half: torch.Tensor,
    v_matrix: torch.Tensor,
    xyz_matrix: torch.Tensor,
    atol: float = 1e-12,
) -> torch.Tensor:
    r"""
    Compute the SPD product-space ball indicator.

    Parameters
    ----------
    u_inv_half : torch.Tensor
        Inverse square roots of query points.

        Shape:
        - ``(M, K, p, p)``

    v_matrix : torch.Tensor
        Second query points.

        Shape:
        - ``(M, K, p, p)``

    xyz_matrix : torch.Tensor
        Sample points.

        Shape:
        - ``(M, K, p, p)``

    Returns
    -------
    torch.Tensor
        Indicator values of shape ``(M,)``.
    """
    if u_inv_half.shape != v_matrix.shape or u_inv_half.shape != xyz_matrix.shape:
        raise ValueError("u_inv_half, v_matrix, and xyz_matrix must have the same shape.")

    if u_inv_half.ndim != 4:
        raise ValueError("Inputs must have shape (M, K, p, p).")

    M, K, p, _ = v_matrix.shape

    u_flat = u_inv_half.reshape(M * K, p, p)
    v_flat = v_matrix.reshape(M * K, p, p)
    x_flat = xyz_matrix.reshape(M * K, p, p)

    dux = spd_affine_invariant_distance_from_inv_half(
        P1_inv_half=u_flat,
        P2_matrix=x_flat,
        atol=atol,
    )

    duv = spd_affine_invariant_distance_from_inv_half(
        P1_inv_half=u_flat,
        P2_matrix=v_flat,
        atol=atol,
    )

    indicators = (dux <= duv + atol).to(torch.int8).reshape(M, K)

    return torch.prod(indicators, dim=1)


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


def generate_data_spd(
    n: int,
    size: int = 2,
    rho: float = 0.0,
    seed: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Generate simulated SPD-valued samples ``(X, Y, Z)`` using PyTorch.

    This function only supports the SPD case. The outputs are returned on
    the specified device. If ``device`` is None, CUDA is used when available;
    otherwise CPU is used.

    Parameters
    ----------
    n : int
        Sample size.

    size : int, default=2
        Matrix size ``p``. The observations lie in the space of ``p x p``
        symmetric positive definite matrices.

    rho : float, default=0.0
        Dependence parameter controlling the conditional association between
        ``X`` and ``Y`` given ``Z``. The value should satisfy ``-1 <= rho <= 1``.

    seed : int or None, default=None
        Random seed.

    device : torch.device or None, default=None
        Device on which the samples are generated and returned. If None,
        ``torch.device("cuda")`` is used when CUDA is available; otherwise
        ``torch.device("cpu")`` is used.

    dtype : torch.dtype, default=torch.float64
        Floating-point dtype of the generated tensors.

    Returns
    -------
    X, Y, Z : tuple of torch.Tensor
        Tensors of shape ``(n, p, p)`` stored on ``device`` with dtype ``dtype``.

    Notes
    -----
    The base matrix ``Z`` is generated from a Wishart-type construction:
    ``Z = A^T A / nu``.
    Then conditional perturbations are constructed through correlated Gaussian
    matrix factors, and ``X, Y`` are formed by congruence transformations
    using the Cholesky factor of ``Z``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must satisfy -1 <= rho <= 1.")

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    p = size
    nu = p + 6
    rho_comp = max(0.0, 1.0 - rho**2) ** 0.5

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


# ============================================================
# SPD product-space empirical MDF (batched)
# ------------------------------------------------------------
# This section implements the empirical metric distribution
# function (MDF) for SPD-valued product spaces under batched
# query evaluation.
#
# Given observed SPD product-space samples
#     X_1, ..., X_n ∈ (SPD)^K,
# the empirical MDF at a query pair (u, v) is defined as
#
#     F_n(u, v) = (1/n) ∑_{i=1}^n δ(u, v, X_i),
#
# where δ is the product-space indicator constructed from the
# affine-invariant Riemannian distance on each SPD component.
#
# This implementation:
#   - assumes all components lie in SPD spaces,
#   - uses precomputed SPDDataBundle representations,
#   - runs entirely on GPU with PyTorch,
#   - vectorizes over both sample index (n) and query batch (B)
#     by forming a matched batch of size B × n.
#
# The function ``emdf_product_pair_batch`` evaluates the empirical
# MDF simultaneously for a batch of query pairs.
#
# Shape convention:
#   - Samples:
#         (n, K, p, p)
#   - Query batch:
#         (B, K, p, p)
#   - Output:
#         (B,)
# ============================================================


def emdf_product_pair_batch(
    Bundle_xyz_samples: SPDDataBundle,
    Bundle_u_batch: SPDDataBundle,
    Bundle_v_batch: SPDDataBundle,
    atol: float = 1e-12,
) -> torch.Tensor:
    r"""
    Compute empirical metric distribution function values for a batch of
    SPD product-space query pairs.

    For each query pair ``(u_b, v_b)``, this function evaluates

    .. math::
        \hat F_n(u_b, v_b)
        =
        \frac{1}{n}
        \sum_{i=1}^n
        \delta(u_b, v_b, x_i),

    where ``x_i`` is an observed product-space sample and ``\delta`` is the
    product-space indicator

    .. math::
        \delta(u_b, v_b, x_i)
        =
        \prod_{k=1}^K
        1\{
            d(u_{b,k}, x_{i,k})
            \le
            d(u_{b,k}, v_{b,k}) + \mathrm{atol}
        \}.

    Here ``d`` is the affine-invariant Riemannian distance on the SPD
    manifold. All computations are performed on GPU using precomputed
    ``SPDDataBundle`` quantities.

    Parameters
    ----------
    Bundle_xyz_samples : SPDDataBundle
        Observed SPD product-space samples.

        Required field:
        - ``matrix`` : shape ``(n, K, p, p)``

    Bundle_u_batch : SPDDataBundle
        First batch of query points.

        Required field:
        - ``inv_half`` : shape ``(B, K, p, p)``

    Bundle_v_batch : SPDDataBundle
        Second batch of query points.

        Required field:
        - ``matrix`` : shape ``(B, K, p, p)``

    atol : float, default=1e-12
        Absolute tolerance used in the indicator comparison.

    Returns
    -------
    torch.Tensor
        Empirical MDF values of shape ``(B,)``.

    Notes
    -----
    The computation forms the matched batch of pairs ``(u_b, v_b, x_i)``
    over all ``b = 1, ..., B`` and ``i = 1, ..., n``. The resulting
    indicators are averaged over the sample index ``i``.
    """
    X_matrix = Bundle_xyz_samples.matrix      # (n, K, p, p)
    U_inv_half = Bundle_u_batch.inv_half      # (B, K, p, p)
    V_matrix = Bundle_v_batch.matrix          # (B, K, p, p)

    n = X_matrix.shape[0]
    B = U_inv_half.shape[0]

    K, p, q = X_matrix.shape[1:]
    # if p != q:
    #     raise ValueError(f"Each SPD matrix must be square, but got {(p, q)}.")

    # if U_inv_half.shape[1:] != (K, p, p) or V_matrix.shape[1:] != (K, p, p):
    #     raise ValueError("Incompatible product-space shapes.")

    # Expand to matched query-sample pairs and flatten (B, n) -> B*n.
    Bn_shape = (B * n, K, p, p)

    U_exp_inv_half = (
        U_inv_half[:, None, ...]
        .expand(B, n, K, p, p)
        .reshape(Bn_shape)
    )

    V_exp_matrix = (
        V_matrix[:, None, ...]
        .expand(B, n, K, p, p)
        .reshape(Bn_shape)
    )

    X_exp_matrix = (
        X_matrix[None, :, ...]
        .expand(B, n, K, p, p)
        .reshape(Bn_shape)
    )

    vals = delta_product(
        u_inv_half=U_exp_inv_half,
        v_matrix=V_exp_matrix,
        xyz_matrix=X_exp_matrix,
        atol=atol,
    )  # (B*n,)

    # Average over the empirical sample index.
    return vals.reshape(B, n).to(dtype=X_matrix.dtype).mean(dim=1)



# ============================================================
# Batched product-form estimator in the SPD setting
# ------------------------------------------------------------
# This section implements the batched evaluation of the estimator
#
#     \hat F_n^{M,\perp}(u, v),
#
# for three-component product-space query pairs in which each
# component takes values in an SPD manifold.
#
# For each query pair (u_b, v_b), the estimator combines:
#   - the empirical conditional indicator average over generated
#     X-samples,
#   - the empirical conditional indicator average over generated
#     Y-samples,
#   - the indicator evaluated on the observed Z-sample,
#
# and then averages the resulting product over the observed sample
# index l = 1, ..., n.
#
# All distances are computed using the affine-invariant Riemannian
# metric on SPD matrices. Inputs are represented as SPDDataBundle
# objects so that precomputed geometric quantities can be reused.
#
# The implementation is fully vectorized over:
#   - the query batch dimension B,
#   - the observed sample dimension n,
#   - and, for the X and Y components, the conditional replication
#     dimension M.
#
# Shape convention:
#   - Query bundles:
#         (B, 3, p, p)
#     where the second dimension corresponds to the
#     (X, Y, Z) components.
#
#   - Conditional generated bundles:
#         (n, M, p, p)
#
#   - Observed conditioning bundle:
#         (n, p, p)
#
#   - Output:
#         (B,)
# ============================================================


def _select_matrix(bundle: SPDDataBundle, comp: int) -> torch.Tensor:
    return bundle.matrix[:, comp, ...]


def _select_inv_half(bundle: SPDDataBundle, comp: int) -> torch.Tensor:
    return bundle.inv_half[:, comp, ...]


def _expand_bn_matrix_from_b(x: torch.Tensor, n_expand: int) -> torch.Tensor:
    # (B, p, p) -> (B, n, p, p) -> (B*n, p, p)
    B, p, _ = x.shape
    return x[:, None, ...].expand(B, n_expand, p, p).reshape(B * n_expand, p, p)


def _expand_bnm_matrix_from_b(x: torch.Tensor, n_expand: int, m_expand: int) -> torch.Tensor:
    # (B, p, p) -> (B, n, M, p, p) -> (B*n*M, p, p)
    B, p, _ = x.shape
    return (
        x[:, None, None, ...]
        .expand(B, n_expand, m_expand, p, p)
        .reshape(B * n_expand * m_expand, p, p)
    )


def _broadcast_nm_matrix(x: torch.Tensor, B_expand: int) -> torch.Tensor:
    # (n, M, p, p) -> (B, n, M, p, p) -> (B*n*M, p, p)
    n, M, p, _ = x.shape
    return (
        x[None, ...]
        .expand(B_expand, n, M, p, p)
        .reshape(B_expand * n * M, p, p)
    )


def _broadcast_n_matrix(x: torch.Tensor, B_expand: int) -> torch.Tensor:
    # (n, p, p) -> (B, n, p, p) -> (B*n, p, p)
    n, p, _ = x.shape
    return (
        x[None, ...]
        .expand(B_expand, n, p, p)
        .reshape(B_expand * n, p, p)
    )


def f_perp_gen_pair_batch(
    Bundle_u_batch: SPDDataBundle,
    Bundle_v_batch: SPDDataBundle,
    Bundle_X_orc: SPDDataBundle,
    Bundle_Y_orc: SPDDataBundle,
    Bundle_Z: SPDDataBundle,
    atol: float = 1e-12,
    chunk_size_xy: int|None = None,
    chunk_size_z: int|None = None,
) -> torch.Tensor:
    r"""
    Compute batched values of the product-form estimator
    ``\hat F_n^{M,\perp}(u, v)`` in the SPD setting.

    Parameters
    ----------
    Bundle_u_batch : SPDDataBundle
        Query bundle with matrix shape ``(B, 3, p, p)``.

    Bundle_v_batch : SPDDataBundle
        Query bundle with matrix shape ``(B, 3, p, p)``.

    Bundle_X_orc : SPDDataBundle
        Conditional generated samples for the X component, with matrix shape
        ``(n, M, p, p)``.

    Bundle_Y_orc : SPDDataBundle
        Conditional generated samples for the Y component, with matrix shape
        ``(n, M, p, p)``.

    Bundle_Z : SPDDataBundle
        Observed conditioning sample for the Z component, with matrix shape
        ``(n, p, p)``.

    atol : float, default=1e-12
        Absolute tolerance used in the indicator comparison.

    chunk_size_xy : int, default=8192
        Chunk size used for the X and Y component expansions of size ``B*n*M``.

    chunk_size_z : int, default=8192
        Chunk size used for the Z component expansions of size ``B*n``.

    Returns
    -------
    torch.Tensor
        One-dimensional tensor of shape ``(B,)`` containing the product-form
        estimates.
    """
    # if Bundle_u_batch.matrix.shape != Bundle_v_batch.matrix.shape:
    #     raise ValueError(
    #         "Bundle_u_batch.matrix and Bundle_v_batch.matrix must have the same shape."
    #     )

    # if Bundle_u_batch.matrix.ndim != 4:
    #     raise ValueError(
    #         "Bundle_u_batch.matrix must have shape (B, 3, p, p)."
    #     )

    # if Bundle_u_batch.matrix.shape[1] != 3:
    #     raise ValueError(
    #         "The second dimension of Bundle_u_batch.matrix must be 3 "
    #         "for the (X, Y, Z) components."
    #     )

    B, _, p, q = Bundle_u_batch.matrix.shape
    # if p != q:
    #     raise ValueError("Each SPD query matrix must be square.")

    # if Bundle_X_orc.matrix.ndim != 4 or Bundle_Y_orc.matrix.ndim != 4:
    #     raise ValueError(
    #         "Bundle_X_orc.matrix and Bundle_Y_orc.matrix must have shape (n, M, p, p)."
    #     )

    # if Bundle_Z.matrix.ndim != 3:
    #     raise ValueError("Bundle_Z.matrix must have shape (n, p, p).")

    n = Bundle_Z.matrix.shape[0]
    M = Bundle_X_orc.matrix.shape[1]

    # if Bundle_Y_orc.matrix.shape[:2] != (n, M):
    #     raise ValueError(
    #         "Bundle_X_orc and Bundle_Y_orc must have matching leading shapes (n, M)."
    #     )

    # if Bundle_X_orc.matrix.shape[-2:] != (p, p):
    #     raise ValueError("Bundle_X_orc matrices must have shape (p, p) matching queries.")
    # if Bundle_Y_orc.matrix.shape[-2:] != (p, p):
    #     raise ValueError("Bundle_Y_orc matrices must have shape (p, p) matching queries.")
    # if Bundle_Z.matrix.shape[-2:] != (p, p):
    #     raise ValueError("Bundle_Z matrices must have shape (p, p) matching queries.")

    dtype = Bundle_Z.matrix.dtype
    device = Bundle_Z.matrix.device

    # ------------------------------------------------------------
    # Split X / Y / Z query components
    # ------------------------------------------------------------
    ux_inv_half = Bundle_u_batch.inv_half[:, 0, ...]   # (B, p, p)
    uy_inv_half = Bundle_u_batch.inv_half[:, 1, ...]
    uz_inv_half = Bundle_u_batch.inv_half[:, 2, ...]

    vx_matrix = Bundle_v_batch.matrix[:, 0, ...]       # (B, p, p)
    vy_matrix = Bundle_v_batch.matrix[:, 1, ...]
    vz_matrix = Bundle_v_batch.matrix[:, 2, ...]

    # ============================================================
    # X / Y components: total size = B * n * M
    # ============================================================
    n_total_xy = B * n * M

    dux = torch.empty((n_total_xy,), device=device, dtype=dtype)
    duv_x = torch.empty((n_total_xy,), device=device, dtype=dtype)
    duy = torch.empty((n_total_xy,), device=device, dtype=dtype)
    duv_y = torch.empty((n_total_xy,), device=device, dtype=dtype)

    if chunk_size_xy is None:
        chunk_size_xy = n_total_xy

    for start in range(0, n_total_xy, chunk_size_xy):
        end = min(start + chunk_size_xy, n_total_xy)

        flat_idx = torch.arange(start, end, device=device, dtype=torch.long)

        b_idx = torch.div(flat_idx, n * M, rounding_mode="floor")
        rem = torch.remainder(flat_idx, n * M)
        n_idx = torch.div(rem, M, rounding_mode="floor")
        m_idx = torch.remainder(rem, M)

        # X component
        ux_chunk = ux_inv_half[b_idx]              # (chunk, p, p)
        vx_chunk = vx_matrix[b_idx]                # (chunk, p, p)
        X_chunk = Bundle_X_orc.matrix[n_idx, m_idx]

        dux[start:end] = spd_affine_invariant_distance_from_inv_half(
            P1_inv_half=ux_chunk,
            P2_matrix=X_chunk,
            atol=atol,
        )
        duv_x[start:end] = spd_affine_invariant_distance_from_inv_half(
            P1_inv_half=ux_chunk,
            P2_matrix=vx_chunk,
            atol=atol,
        )

        # Y component
        uy_chunk = uy_inv_half[b_idx]
        vy_chunk = vy_matrix[b_idx]
        Y_chunk = Bundle_Y_orc.matrix[n_idx, m_idx]

        duy[start:end] = spd_affine_invariant_distance_from_inv_half(
            P1_inv_half=uy_chunk,
            P2_matrix=Y_chunk,
            atol=atol,
        )
        duv_y[start:end] = spd_affine_invariant_distance_from_inv_half(
            P1_inv_half=uy_chunk,
            P2_matrix=vy_chunk,
            atol=atol,
        )

    # ============================================================
    # Z component: total size = B * n
    # ============================================================
    n_total_z = B * n

    duz = torch.empty((n_total_z,), device=device, dtype=dtype)
    duv_z = torch.empty((n_total_z,), device=device, dtype=dtype)

    if chunk_size_z is None:
        chunk_size_z = n_total_z

    for start in range(0, n_total_z, chunk_size_z):
        end = min(start + chunk_size_z, n_total_z)

        flat_idx = torch.arange(start, end, device=device, dtype=torch.long)

        b_idx = torch.div(flat_idx, n, rounding_mode="floor")
        n_idx = torch.remainder(flat_idx, n)

        uz_chunk = uz_inv_half[b_idx]
        vz_chunk = vz_matrix[b_idx]
        Z_chunk = Bundle_Z.matrix[n_idx]

        duz[start:end] = spd_affine_invariant_distance_from_inv_half(
            P1_inv_half=uz_chunk,
            P2_matrix=Z_chunk,
            atol=atol,
        )
        duv_z[start:end] = spd_affine_invariant_distance_from_inv_half(
            P1_inv_half=uz_chunk,
            P2_matrix=vz_chunk,
            atol=atol,
        )

    # ------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------
    dx = (dux <= duv_x + atol).to(dtype=dtype).reshape(B, n, M)
    dy = (duy <= duv_y + atol).to(dtype=dtype).reshape(B, n, M)
    dz = (duz <= duv_z + atol).to(dtype=dtype).reshape(B, n)

    tx = dx.mean(dim=2)
    ty = dy.mean(dim=2)

    return (tx * ty * dz).mean(dim=1)


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
    Bundle_X: SPDDataBundle,
    Bundle_Y: SPDDataBundle,
    Bundle_Z: SPDDataBundle,
    Bundle_X_orc: SPDDataBundle,
    Bundle_Y_orc: SPDDataBundle,
    atol: float = 1e-12,
    batch_size: int = 1024,
    chunk_size_xy: int|None = None,
    chunk_size_z: int|None = None,
) -> float:
    r"""
    Compute the statistic

    .. math::
        T_n^{\mathrm{gen}}
        =
        \frac{1}{n(n-1)}
        \sum_{i \ne j}
        \left[
        \hat F_n^{\mathcal M}(u_i, u_j)
        -
        \hat F_n^{M,\perp}(u_i, u_j)
        \right]^2.

    Parameters
    ----------
    Bundle_X, Bundle_Y, Bundle_Z : SPDDataBundle
        Observed SPD samples for the three components.

        Shape of each ``.matrix``:
        - ``(n, p, p)``

    Bundle_X_orc, Bundle_Y_orc : SPDDataBundle
        Conditional generated samples for the ``X`` and ``Y`` components.

        Shape of each ``.matrix``:
        - ``(n, M, p, p)``

    atol : float, default=1e-12
        Absolute tolerance used in the indicator comparisons.

    batch_size : int, default=1024
        Number of ordered pairs ``(i, j)``, ``i \ne j``, processed in each batch.

    Returns
    -------
    float
        Value of the test statistic.

    Notes
    -----
    This routine assumes all inputs are represented as ``SPDDataBundle``
    objects on GPU. The computation over all ordered pairs ``(i, j)``
    with ``i \ne j`` is carried out in batches for memory efficiency.
    """

    if Bundle_X.matrix.shape != Bundle_Y.matrix.shape or Bundle_X.matrix.shape != Bundle_Z.matrix.shape:
        raise ValueError("Bundle_X, Bundle_Y, and Bundle_Z must have identical shapes.")

    n = Bundle_X.matrix.shape[0]
    if n < 2:
        raise ValueError("At least two observations are required.")

    if Bundle_X_orc.matrix.shape[0] != n or Bundle_Y_orc.matrix.shape[0] != n:
        raise ValueError("Bundle_X_orc and Bundle_Y_orc must have leading dimension n matching observed data.")

    Bundle_S = stack_spd_triplet(Bundle_X, Bundle_Y, Bundle_Z)

    total_pairs = n * (n - 1)
    total = 0.0
    device = Bundle_X.matrix.device

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)

        k = torch.arange(start, end, device=device, dtype=torch.long)
        ib = torch.div(k, n - 1, rounding_mode="floor")
        r = torch.remainder(k, n - 1)
        jb = r + (r >= ib).to(torch.long)

        Bundle_u_batch = slice_spd_bundle(Bundle_S, ib)
        Bundle_v_batch = slice_spd_bundle(Bundle_S, jb)

        emdf_P = emdf_product_pair_batch(
            Bundle_xyz_samples=Bundle_S,
            Bundle_u_batch=Bundle_u_batch,
            Bundle_v_batch=Bundle_v_batch,
            atol=atol,
        )

        emdf_I = f_perp_gen_pair_batch(
            Bundle_u_batch=Bundle_u_batch,
            Bundle_v_batch=Bundle_v_batch,
            Bundle_X_orc=Bundle_X_orc,
            Bundle_Y_orc=Bundle_Y_orc,
            Bundle_Z=Bundle_Z,
            atol=atol,
            chunk_size_xy=chunk_size_xy,
            chunk_size_z=chunk_size_z,
        )

        diff = emdf_P - emdf_I
        total += float((diff * diff).sum().item())

    return total / (n * (n - 1))


def statistics_fast(
    Bundle_X: SPDDataBundle,
    Bundle_Y: SPDDataBundle,
    Bundle_Z: SPDDataBundle,
    Bundle_X_orc: SPDDataBundle,
    Bundle_Y_orc: SPDDataBundle,
    atol: float = 1e-12,
    batch_size: int = 1024,
) -> float:
    n = Bundle_X.matrix.shape[0]
    M = Bundle_X_orc.matrix.shape[1]
    device = Bundle_X.matrix.device
    dtype = Bundle_X.matrix.dtype

    d_x = spd_affine_invariant_distance_from_inv_half(
        broadcast_spd_pairs(Bundle_X.inv_half, mode="ij"),
        broadcast_spd_pairs(Bundle_X.matrix, mode="ji"),
        atol=atol,
    )

    d_y = spd_affine_invariant_distance_from_inv_half(
        broadcast_spd_pairs(Bundle_Y.inv_half, mode="ij"),
        broadcast_spd_pairs(Bundle_Y.matrix, mode="ji"),
        atol=atol,
    )

    d_z = spd_affine_invariant_distance_from_inv_half(
        broadcast_spd_pairs(Bundle_Z.inv_half, mode="ij"),
        broadcast_spd_pairs(Bundle_Z.matrix, mode="ji"),
        atol=atol,
    )

    d_x_sim = spd_affine_invariant_distance_from_inv_half(
        Bundle_X.inv_half[:, None, None, :, :].expand(n, n, M, -1, -1),
        Bundle_X_orc.matrix[None, :, :, :, :].expand(n, n, M, -1, -1),
        atol=atol,
    )

    d_y_sim = spd_affine_invariant_distance_from_inv_half(
        Bundle_Y.inv_half[:, None, None, :, :].expand(n, n, M, -1, -1),
        Bundle_Y_orc.matrix[None, :, :, :, :].expand(n, n, M, -1, -1),
        atol=atol,
    )

    total_pairs = n * (n - 1)
    total = 0.0

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)

        k = torch.arange(start, end, device=device, dtype=torch.long)
        ib = torch.div(k, n - 1, rounding_mode="floor")
        r = torch.remainder(k, n - 1)
        jb = r + (r >= ib).to(torch.long)

        dx_ij = d_x[ib, jb]  # (B,)
        dy_ij = d_y[ib, jb]
        dz_ij = d_z[ib, jb]

        emdf_P = (
            (d_x[ib, :] <= dx_ij[:, None] + atol)
            & (d_y[ib, :] <= dy_ij[:, None] + atol)
            & (d_z[ib, :] <= dz_ij[:, None] + atol)
        ).to(dtype).mean(dim=1)

        tx = (
            d_x_sim[ib, :, :] <= dx_ij[:, None, None] + atol
        ).to(dtype).mean(dim=2)

        ty = (
            d_y_sim[ib, :, :] <= dy_ij[:, None, None] + atol
        ).to(dtype).mean(dim=2)

        dz = (
            d_z[ib, :] <= dz_ij[:, None] + atol
        ).to(dtype)

        emdf_I = (tx * ty * dz).mean(dim=1)

        diff = emdf_P - emdf_I
        total += float((diff * diff).sum().item())

    return total / total_pairs


# ============================================================
# Local permutation testing on SPD manifolds
# ------------------------------------------------------------
# This section implements a complete local permutation testing
# pipeline for conditional independence in SPD-valued settings.
#
# The workflow consists of three main components:
#
# 1. Pairwise distance computation:
#    - Computes affine-invariant Riemannian distances between
#      SPD matrices using GPU-accelerated batched operations.
#    - Only the upper-triangular entries are evaluated explicitly
#      and mirrored for efficiency.
#
# 2. K-nearest-neighbor (KNN) construction:
#    - Builds local neighborhoods in the conditioning space Z
#      based on the computed pairwise distance matrix.
#    - Supports inclusion/exclusion of self-neighbors.
#
# 3. Local permutation mechanism:
#    - Generates locally permuted versions of Y by sampling
#      from KNN neighborhoods.
#    - Two strategies are supported:
#        (a) independent local resampling
#        (b) constrained one-pass shuffle with fallback
#
# 4. Local permutation test:
#    - Repeatedly applies local permutations to Y while keeping
#      X and Z fixed.
#    - Evaluates a user-specified test statistic under both the
#      observed data and permuted samples.
#    - Computes a Monte Carlo p-value based on the permutation
#      distribution.
#
# Key features:
#    - Fully GPU-compatible (torch + CUDA)
#    - Memory-efficient via batching (distance + statistic)
#    - Modular design: stat_fn is user-defined
#    - Designed for integration with SPD-based conditional
#      independence tests
#
# Shape conventions:
#    - Observations:
#          (n, p, p)
#    - KNN indices:
#          (n, k)
#    - Permutation samples:
#          B replicates
#
# Output:
#    - Test statistic
#    - Permutation distribution
#    - p-value and rejection decision
# ============================================================


def compute_pairwise_distances(
    Bundle_Z: SPDDataBundle,
    atol: float = 1e-12,
    batch_size: int | None = None,
) -> torch.Tensor:
    r"""
    Compute the pairwise affine-invariant distance matrix for SPD samples.

    Given SPD-valued observations
        Z_1, ..., Z_n ∈ SPD(p),
    this function evaluates the pairwise distance matrix

    .. math::
        D_{i,j}
        =
        d(Z_i, Z_j),

    where ``d`` denotes the affine-invariant Riemannian distance

    .. math::
        d(P_1, P_2)
        =
        \left\|
        \log\!\left(P_1^{-1/2} P_2 P_1^{-1/2}\right)
        \right\|_F.

    Parameters
    ----------
    Bundle_Z : SPDDataBundle
        SPD sample bundle.

        Shape of ``Bundle_Z.matrix``:
        - ``(n, p, p)``

        Required fields:
        - ``matrix`` : SPD matrices
        - ``inv_half`` : inverse square roots of matrices

    atol : float, default=1e-12
        Lower bound applied to eigenvalues for numerical stability in the
        distance computation.

    batch_size : int or None, default=None
        Number of pairwise distances processed per batch.

        The total number of distinct pairs is

        .. math::
            n_{\text{pairs}} = \frac{n(n-1)}{2}.

        If ``None``, all pairs are processed in a single batch.

    Returns
    -------
    torch.Tensor
        Pairwise distance matrix of shape ``(n, n)``, where

        - ``D[i, j] = d(Z_i, Z_j)``
        - ``D[i, i] = 0``

    Notes
    -----
    1. Computation strategy:
       Only the upper-triangular entries (``i < j``) are computed explicitly.
       The lower-triangular part is obtained by symmetry.

    2. Vectorization:
       All pairs ``(i, j)``, ``i < j``, are flattened into a single index set
       of size ``n(n-1)/2`` and processed in batches.

    3. GPU execution:
       The function assumes all inputs are CUDA tensors and uses batched
       matrix operations for efficient evaluation.

    4. Memory efficiency:
       Chunking via ``batch_size`` prevents excessive memory usage when
       ``n`` is large.
    """

    Z = Bundle_Z.matrix
    Z_inv_half = Bundle_Z.inv_half

    n, p, _ = Z.shape

    device = Z.device
    dtype = Z.dtype

    D = torch.zeros((n, n), device=device, dtype=dtype)

    n_total = n * (n - 1) // 2
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

        d_ij = spd_affine_invariant_distance_from_inv_half(
            P1_inv_half=Z_inv_half[i_idx],
            P2_matrix=Z[j_idx],
            atol=atol,
        )

        D[i_idx, j_idx] = d_ij
        D[j_idx, i_idx] = d_ij

    return D


def build_knn_indices(
    Bundle_Z: SPDDataBundle,
    k: int,
    include_self: bool = True,
    batch_size: int | None = None,
    atol: float = 1e-12,
) -> torch.Tensor:
    r"""
    Construct k-nearest-neighbor index sets in the SPD conditioning space.

    Parameters
    ----------
    Bundle_Z : SPDDataBundle
        SPD sample bundle.

        Shape of ``Bundle_Z.matrix``:
        - ``(n, p, p)``

    k : int
        Number of nearest neighbors to retain.

    include_self : bool, default=True
        Whether each point is allowed to include itself among its neighbors.

    batch_size : int or None, default=None
        Number of pairwise distances processed per batch when computing the
        distance matrix.

    atol : float, default=1e-12
        Numerical tolerance passed to the SPD distance computation.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(n, k)`` whose ``i``-th row contains the
        indices of the ``k`` nearest neighbors of the ``i``-th point.
    """
    if not isinstance(Bundle_Z, SPDDataBundle):
        raise TypeError("Bundle_Z must be an SPDDataBundle.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    D = compute_pairwise_distances(
        Bundle_Z=Bundle_Z,
        atol=atol,
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

        # D is only used for neighbor selection, so inplace modification is safe.
        diag_idx = torch.arange(n, device=D.device)
        D[diag_idx, diag_idx] = torch.inf

    cand = torch.topk(D, k=k, dim=1, largest=False).indices

    # Sort the selected candidates by their actual distances.
    cand_dist = torch.gather(D, dim=1, index=cand)
    order = torch.argsort(cand_dist, dim=1)

    return torch.gather(cand, dim=1, index=order)


def local_permute_y(
    Bundle_Y: SPDDataBundle,
    knn_indices: torch.Tensor,
    generator: torch.Generator | None = None,
    strategy: str = "resample",
) -> SPDDataBundle:
    r"""
    Construct a locally permuted version of ``Y`` using neighbor indices.

    Parameters
    ----------
    Bundle_Y : SPDDataBundle
        Bundle of ``Y`` samples.

        Shape of ``Bundle_Y.matrix``:
        - ``(n, p, p)``

    knn_indices : torch.Tensor
        Long tensor of shape ``(n, k)`` whose ``i``-th row contains the
        admissible neighbor indices associated with the ``i``-th observation.

    generator : torch.Generator or None, default=None
        Random number generator used for local resampling or permutation.

    strategy : {"resample", "shuffle_once"}, default="resample"
        Local permutation strategy.

        - ``"resample"``:
          independently sample one neighbor from each row of ``knn_indices``.

        - ``"shuffle_once"``:
          perform a one-pass constrained shuffle with fallback.

    Returns
    -------
    SPDDataBundle
        Locally permuted bundle with the same shape as ``Bundle_Y``.
    """
    if not isinstance(Bundle_Y, SPDDataBundle):
        raise TypeError("Bundle_Y must be an SPDDataBundle.")

    Y = Bundle_Y.matrix
    n = Y.shape[0]

    if not isinstance(knn_indices, torch.Tensor):
        raise TypeError("knn_indices must be a torch.Tensor.")

    if knn_indices.device != Y.device:
        raise ValueError("knn_indices must be on the same device as Bundle_Y.")

    if knn_indices.dtype != torch.long:
        raise ValueError("knn_indices must have dtype torch.long.")

    if knn_indices.ndim != 2:
        raise ValueError(
            f"knn_indices must have shape (n, k), but got {tuple(knn_indices.shape)}."
        )

    if knn_indices.shape[0] != n:
        raise ValueError("knn_indices and Bundle_Y must have the same first dimension.")

    if knn_indices.shape[1] == 0:
        raise ValueError("Each row of knn_indices must contain at least one neighbor.")

    if strategy not in {"resample", "shuffle_once"}:
        raise ValueError("strategy must be one of {'resample', 'shuffle_once'}.")

    device = Y.device
    k = knn_indices.shape[1]

    if generator is None:
        generator = torch.Generator(device=device)

    # ------------------------------------------------------------
    # Strategy 1: independent local resampling
    # ------------------------------------------------------------
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

        return slice_spd_bundle(Bundle_Y, chosen)

    # ------------------------------------------------------------
    # Strategy 2: one-pass constrained shuffle with fallback
    # ------------------------------------------------------------
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

    return slice_spd_bundle(Bundle_Y, chosen)


def local_permutation_test(
    Bundle_X: SPDDataBundle,
    Bundle_Y: SPDDataBundle,
    Bundle_Z: SPDDataBundle,
    stat_fn: Callable[..., float],
    stat_kwargs: Optional[Dict] = None,
    B: int = 500,
    k: Optional[int] = None,
    permutation_strategy: str = "resample",
    alpha: float = 0.05,
    generator: torch.Generator | None = None,
    show_progress: bool = False,
    knn_batch_size: int | None = None,
    atol: float = 1e-12,
) -> Dict[str, object]:
    r"""
    Perform a local permutation test for SPD-valued data on GPU.

    Parameters
    ----------
    Bundle_X, Bundle_Y, Bundle_Z : SPDDataBundle
        Observed SPD samples.

        Shape of each ``.matrix``:
        - ``(n, p, p)``

    stat_fn : callable
        Statistic function with signature

        ``stat_fn(Bundle_X, Bundle_Y, Bundle_Z, **stat_kwargs)``.

    stat_kwargs : dict or None, default=None
        Additional keyword arguments passed to ``stat_fn``.
        For example, generated conditional samples such as
        ``Bundle_X_orc`` and ``Bundle_Y_orc`` should be passed here.

    B : int, default=500
        Number of local permutation replicates.

    k : int or None, default=None
        Neighborhood size. If None, use ``max(10, ceil(sqrt(n)))``.

    permutation_strategy : {"resample", "shuffle_once"}, default="resample"
        Local permutation strategy passed to ``local_permute_y``.

    alpha : float, default=0.05
        Significance level.

    generator : torch.Generator or None, default=None
        Random number generator controlling the local permutations.

    show_progress : bool, default=False
        Whether to display a tqdm progress bar.

    knn_batch_size : int or None, default=None
        Batch size used when computing pairwise SPD distances for KNN.

    atol : float, default=1e-12
        Numerical tolerance used in distance computations.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"T_obs"`` : observed statistic
        - ``"T_perm"`` : permutation statistics, tensor of shape ``(B,)``
        - ``"p_value"`` : permutation p-value
        - ``"reject"`` : whether to reject at level ``alpha``
        - ``"k"`` : neighborhood size
        - ``"knn_indices"`` : KNN index tensor of shape ``(n, k)``
    """
    if stat_kwargs is None:
        stat_kwargs = {}

    n = Bundle_Z.matrix.shape[0]
    device = Bundle_Z.matrix.device
    dtype = Bundle_Z.matrix.dtype

    if n < 2:
        raise ValueError("At least two observations are required.")

    if k is None:
        k = max(10, int(torch.ceil(torch.sqrt(torch.tensor(float(n)))).item()))

    if k >= n:
        k = n - 1

    if generator is None:
        generator = torch.Generator(device=device)

    knn_indices = build_knn_indices(
        Bundle_Z=Bundle_Z,
        k=k,
        include_self=False,
        batch_size=knn_batch_size,
        atol=atol,
    )

    T_obs = stat_fn(
        Bundle_X,
        Bundle_Y,
        Bundle_Z,
        atol=atol,
        **stat_kwargs,
    )

    T_perm = torch.empty((B,), device=device, dtype=dtype)

    iterator = range(B)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Permutation", leave=False)

    for b in iterator:
        Bundle_Y_star = local_permute_y(
            Bundle_Y=Bundle_Y,
            knn_indices=knn_indices,
            generator=generator,
            strategy=permutation_strategy,
        )

        T_perm[b] = stat_fn(
            Bundle_X,
            Bundle_Y_star,
            Bundle_Z,
            atol=atol,
            **stat_kwargs,
        )

    p_value_tensor = (1.0 + torch.sum(T_perm >= T_obs).to(dtype=dtype)) / (B + 1.0)

    return {
        "T_obs": float(T_obs),
        "T_perm": T_perm,
        "p_value": float(p_value_tensor.item()),
        "reject": bool(p_value_tensor.item() <= alpha),
        "k": int(k),
        "knn_indices": knn_indices,
    }

