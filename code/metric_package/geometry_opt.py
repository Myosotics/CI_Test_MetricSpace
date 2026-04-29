from typing import Literal, Sequence, Union, Dict, List
import numpy as np
import torch
from collections import defaultdict


# ============================================================
# Distance functions on a single metric space
# ------------------------------------------------------------
# This section implements distance functions for the metric spaces
# considered in the paper:
#   - Euclidean space;
#   - the unit sphere with geodesic distance;
#   - the space of symmetric positive definite matrices equipped
#     with the Cholesky distance.
#
# A unified dispatcher is also provided through ``compute_distance``.
#
# Backend convention:
#   * If GPU=False, computations are carried out with NumPy on CPU.
#   * If GPU=True, computations are carried out with PyTorch on CUDA.
#
# Throughout, strict shape matching is imposed and no broadcasting
# is performed.
# ============================================================


def euclidean_distance(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    GPU: bool = False,
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Compute Euclidean distances under strict shape matching.

    This function supports two input regimes:
    (i) a single-pair query, where ``x`` and ``y`` are one-dimensional
    arrays/tensors of the same shape, and
    (ii) a batched query, where ``x`` and ``y`` are two-dimensional
    arrays/tensors of the same shape and the distance is computed row-wise.

    No broadcasting is allowed. In particular, the shapes of ``x`` and ``y``
    must coincide exactly.

    Parameters
    ----------
    x, y : np.ndarray or torch.Tensor
        Input points or batches of points.

        - If ``GPU=False``, then ``x`` and ``y`` are assumed to be NumPy arrays.
        - If ``GPU=True``, then ``x`` and ``y`` are assumed to be CUDA torch
          tensors.

        Supported shapes are:

        - ``(d,)`` for a single query;
        - ``(M, d)`` for a batch of ``M`` matched queries.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        The Euclidean distance(s).

        - If ``x.shape == y.shape == (d,)``, returns a scalar ``float``.
        - If ``x.shape == y.shape == (M, d)``, returns a one-dimensional object
          of length ``M``:
            * a NumPy array when ``GPU=False``;
            * a torch tensor when ``GPU=True``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` do not have the same number of dimensions, do not
        have identical shapes, or are not one- or two-dimensional.

    Notes
    -----
    This routine assumes that the input types are already consistent with the
    selected backend. In particular, no implicit conversion between NumPy arrays
    and torch tensors is performed inside the function.
    """
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions.")

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have identical shapes, but got "
            f"{tuple(x.shape)} and {tuple(y.shape)}."
        )

    if not GPU:
        if x.ndim == 1:
            return float(np.linalg.norm(x - y))

        if x.ndim == 2:
            return np.linalg.norm(x - y, axis=1)

        raise ValueError("Inputs must be either one- or two-dimensional.")

    if x.ndim == 1:
        return float(torch.linalg.norm(x - y).item())

    if x.ndim == 2:
        return torch.linalg.norm(x - y, dim=1)

    raise ValueError("Inputs must be either one- or two-dimensional.")


def sphere_geodesic_distance(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    GPU: bool = False,
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Compute geodesic distances on the unit sphere under strict shape matching.

    This function supports two input regimes:
    (i) a single-pair query, where ``x`` and ``y`` are one-dimensional
    arrays/tensors of the same shape, and
    (ii) a batched query, where ``x`` and ``y`` are two-dimensional
    arrays/tensors of the same shape and the distance is computed row-wise.

    No broadcasting is allowed. In particular, the shapes of ``x`` and ``y``
    must coincide exactly.

    Parameters
    ----------
    x, y : np.ndarray or torch.Tensor
        Input points or batches of points on the unit sphere.

        - If ``GPU=False``, then ``x`` and ``y`` are assumed to be NumPy arrays.
        - If ``GPU=True``, then ``x`` and ``y`` are assumed to be CUDA torch
          tensors.

        Supported shapes are:

        - ``(d,)`` for a single query;
        - ``(M, d)`` for a batch of ``M`` matched queries.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        The spherical geodesic distance(s).

        - If ``x.shape == y.shape == (d,)``, returns a scalar ``float``.
        - If ``x.shape == y.shape == (M, d)``, returns a one-dimensional object
          of length ``M``:
            * a NumPy array when ``GPU=False``;
            * a torch tensor when ``GPU=True``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` do not have the same number of dimensions, do not
        have identical shapes, or are not one- or two-dimensional.

    Notes
    -----
    This routine assumes that the input types are already consistent with the
    selected backend. In particular, no implicit conversion between NumPy arrays
    and torch tensors is performed inside the function.

    To guard against numerical overflow in the inverse cosine evaluation, the
    inner product is truncated to the interval ``[-1, 1]`` before applying
    ``arccos``.
    """
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same number of dimensions.")

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have identical shapes, but got "
            f"{tuple(x.shape)} and {tuple(y.shape)}."
        )

    if not GPU:
        if x.ndim == 1:
            inner = np.dot(x, y)
            inner = np.clip(inner, -1.0, 1.0)
            return float(np.arccos(inner))

        if x.ndim == 2:
            inner = np.sum(x * y, axis=1)
            inner = np.clip(inner, -1.0, 1.0)
            return np.arccos(inner)

        raise ValueError("Inputs must be either one- or two-dimensional.")

    if x.ndim == 1:
        inner = torch.dot(x, y)
        inner = torch.clamp(inner, -1.0, 1.0)
        return float(torch.arccos(inner).item())

    if x.ndim == 2:
        inner = torch.sum(x * y, dim=1)
        inner = torch.clamp(inner, -1.0, 1.0)
        return torch.arccos(inner)

    raise ValueError("Inputs must be either one- or two-dimensional.")


def spd_cholesky_distance(
    P1: Union[np.ndarray, torch.Tensor],
    P2: Union[np.ndarray, torch.Tensor],
    GPU: bool = False,
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Compute the Cholesky distance for symmetric positive definite matrices
    under strict shape matching.

    This function supports two input regimes:
    (i) a single-pair query, where ``P1`` and ``P2`` are two-dimensional
    arrays/tensors of the same shape, and
    (ii) a batched query, where ``P1`` and ``P2`` are three-dimensional
    arrays/tensors of the same shape and the distance is computed entry-wise
    across the batch.

    No broadcasting is allowed. In particular, the shapes of ``P1`` and ``P2``
    must coincide exactly.

    Parameters
    ----------
    P1, P2 : np.ndarray or torch.Tensor
        Input SPD matrices or batches of SPD matrices.

        - If ``GPU=False``, then ``P1`` and ``P2`` are assumed to be NumPy arrays.
        - If ``GPU=True``, then ``P1`` and ``P2`` are assumed to be CUDA torch
          tensors.

        Supported shapes are:

        - ``(p, p)`` for a single query;
        - ``(M, p, p)`` for a batch of ``M`` matched queries.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        The Cholesky distance(s), defined as the Frobenius norm of the
        difference between the Cholesky factors.

        - If ``P1.shape == P2.shape == (p, p)``, returns a scalar ``float``.
        - If ``P1.shape == P2.shape == (M, p, p)``, returns a one-dimensional
          object of length ``M``:
            * a NumPy array when ``GPU=False``;
            * a torch tensor when ``GPU=True``.

    Raises
    ------
    ValueError
        If ``P1`` and ``P2`` do not have identical shapes, are not two- or
        three-dimensional, or do not represent square matrices.

    Notes
    -----
    This routine assumes that the input types are already consistent with the
    selected backend. In particular, no implicit conversion between NumPy arrays
    and torch tensors is performed inside the function.

    It is also assumed that all input matrices are symmetric positive definite,
    so that their Cholesky factors are well defined.
    """
    if P1.shape != P2.shape:
        raise ValueError(
            f"P1 and P2 must have identical shapes, but got "
            f"{tuple(P1.shape)} and {tuple(P2.shape)}."
        )

    if not GPU:
        if P1.ndim == 2:
            if P1.shape[0] != P1.shape[1]:
                raise ValueError(
                    f"Each SPD matrix must be square, but got shape {tuple(P1.shape)}."
                )

            L1 = np.linalg.cholesky(P1)
            L2 = np.linalg.cholesky(P2)
            return float(np.linalg.norm(L1 - L2, ord="fro"))

        if P1.ndim == 3:
            if P1.shape[1] != P1.shape[2]:
                raise ValueError(
                    f"Each SPD matrix must be square, but got shape {tuple(P1.shape[1:])}."
                )

            L1 = np.linalg.cholesky(P1)
            L2 = np.linalg.cholesky(P2)
            return np.linalg.norm(L1 - L2, axis=(1, 2))

        raise ValueError("Inputs must be either two- or three-dimensional.")

    if P1.ndim == 2:
        if P1.shape[0] != P1.shape[1]:
            raise ValueError(
                f"Each SPD matrix must be square, but got shape {tuple(P1.shape)}."
            )

        L1 = torch.linalg.cholesky(P1)
        L2 = torch.linalg.cholesky(P2)
        return float(torch.linalg.norm(L1 - L2, ord="fro").item())

    if P1.ndim == 3:
        if P1.shape[1] != P1.shape[2]:
            raise ValueError(
                f"Each SPD matrix must be square, but got shape {tuple(P1.shape[1:])}."
            )

        L1 = torch.linalg.cholesky(P1)
        L2 = torch.linalg.cholesky(P2)
        return torch.linalg.norm(L1 - L2, dim=(1, 2))

    raise ValueError("Inputs must be either two- or three-dimensional.")


def spd_affine_invariant_distance(
    P1: Union[np.ndarray, torch.Tensor],
    P2: Union[np.ndarray, torch.Tensor],
    atol: float = 1e-12,
    GPU: bool = False,
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Compute the affine-invariant Riemannian distance for symmetric positive
    definite (SPD) matrices under strict shape matching.

    This function supports two input regimes:
    (i) a single-pair query, where ``P1`` and ``P2`` are two-dimensional
    arrays/tensors of the same shape, and
    (ii) a batched query, where ``P1`` and ``P2`` are three-dimensional
    arrays/tensors of the same shape and the distance is computed entry-wise
    across the batch.

    No broadcasting is allowed. In particular, the shapes of ``P1`` and ``P2``
    must coincide exactly.

    The distance is defined by
    .. math::
        d(P_1, P_2)
        =
        \left\|
        \log\!\left(P_1^{-1/2} P_2 P_1^{-1/2}\right)
        \right\|_F.

    Equivalently, if :math:`\lambda_1, \dots, \lambda_p` are the eigenvalues of
    :math:`P_1^{-1/2} P_2 P_1^{-1/2}`, then
    .. math::
        d(P_1, P_2)
        =
        \left(\sum_{i=1}^p (\log \lambda_i)^2\right)^{1/2}.

    Parameters
    ----------
    P1, P2 : np.ndarray or torch.Tensor
        Input SPD matrices or batches of SPD matrices.

        - If ``GPU=False``, then ``P1`` and ``P2`` are assumed to be NumPy arrays.
        - If ``GPU=True``, then ``P1`` and ``P2`` are assumed to be CUDA torch
          tensors.

        Supported shapes are:

        - ``(p, p)`` for a single query;
        - ``(M, p, p)`` for a batch of ``M`` matched queries.

    atol : float, default=1e-12
        Numerical tolerance used to guard against non-positive eigenvalues
        caused by floating-point error.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        The affine-invariant distance(s).

        - If ``P1.shape == P2.shape == (p, p)``, returns a scalar ``float``.
        - If ``P1.shape == P2.shape == (M, p, p)``, returns a one-dimensional
          object of length ``M``:
            * a NumPy array when ``GPU=False``;
            * a torch tensor when ``GPU=True``.

    Raises
    ------
    ValueError
        If ``P1`` and ``P2`` do not have identical shapes, are not two- or
        three-dimensional, or do not represent square matrices.

    Notes
    -----
    This routine assumes that all input matrices are SPD. No explicit symmetry
    or positive-definiteness checks are performed beyond the numerical clipping
    safeguard on eigenvalues.
    """
    if P1.shape != P2.shape:
        raise ValueError(
            f"P1 and P2 must have identical shapes, but got "
            f"{tuple(P1.shape)} and {tuple(P2.shape)}."
        )

    if not GPU:
        if P1.ndim == 2:
            if P1.shape[0] != P1.shape[1]:
                raise ValueError(
                    f"Each SPD matrix must be square, but got shape {tuple(P1.shape)}."
                )

            vals, vecs = np.linalg.eigh(P1)
            vals = np.maximum(vals, atol)
            P1_inv_half = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

            M = P1_inv_half @ P2 @ P1_inv_half
            M = 0.5 * (M + M.T)

            lam = np.linalg.eigvalsh(M)
            lam = np.maximum(lam, atol)

            return float(np.sqrt(np.sum(np.log(lam) ** 2)))

        if P1.ndim == 3:
            if P1.shape[1] != P1.shape[2]:
                raise ValueError(
                    f"Each SPD matrix must be square, but got shape {tuple(P1.shape[1:])}."
                )

            vals, vecs = np.linalg.eigh(P1)                       # (M, p), (M, p, p)
            vals = np.maximum(vals, atol)

            inv_sqrt_diag = 1.0 / np.sqrt(vals)                  # (M, p)
            P1_inv_half = np.matmul(
                vecs * inv_sqrt_diag[:, None, :],
                np.transpose(vecs, (0, 2, 1))
            )                                                    # (M, p, p)

            M = np.matmul(np.matmul(P1_inv_half, P2), P1_inv_half)
            M = 0.5 * (M + np.transpose(M, (0, 2, 1)))

            lam = np.linalg.eigvalsh(M)                          # (M, p)
            lam = np.maximum(lam, atol)

            return np.sqrt(np.sum(np.log(lam) ** 2, axis=1))

        raise ValueError("Inputs must be either two- or three-dimensional.")

    if P1.ndim == 2:
        if P1.shape[0] != P1.shape[1]:
            raise ValueError(
                f"Each SPD matrix must be square, but got shape {tuple(P1.shape)}."
            )

        vals, vecs = torch.linalg.eigh(P1)
        vals = torch.clamp(vals, min=atol)
        P1_inv_half = vecs @ torch.diag(1.0 / torch.sqrt(vals)) @ vecs.transpose(-1, -2)

        M = P1_inv_half @ P2 @ P1_inv_half
        M = 0.5 * (M + M.transpose(-1, -2))

        lam = torch.linalg.eigvalsh(M)
        lam = torch.clamp(lam, min=atol)

        return float(torch.sqrt(torch.sum(torch.log(lam) ** 2)).item())

    if P1.ndim == 3:
        if P1.shape[1] != P1.shape[2]:
            raise ValueError(
                f"Each SPD matrix must be square, but got shape {tuple(P1.shape[1:])}."
            )

        vals, vecs = torch.linalg.eigh(P1)                       # (M, p), (M, p, p)
        vals = torch.clamp(vals, min=atol)

        inv_sqrt_diag = 1.0 / torch.sqrt(vals)                  # (M, p)
        P1_inv_half = torch.matmul(
            vecs * inv_sqrt_diag[:, None, :],
            vecs.transpose(-1, -2)
        )                                                       # (M, p, p)

        M = torch.matmul(torch.matmul(P1_inv_half, P2), P1_inv_half)
        M = 0.5 * (M + M.transpose(-1, -2))

        lam = torch.linalg.eigvalsh(M)                          # (M, p)
        lam = torch.clamp(lam, min=atol)

        return torch.sqrt(torch.sum(torch.log(lam) ** 2, dim=1))

    raise ValueError("Inputs must be either two- or three-dimensional.")


def compute_distance(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    space_type: Literal["euclidean", "sphere", "spd"],
    GPU: bool = False,
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Compute the distance between matched inputs in a supported metric space.

    This function provides a unified interface for the distance routines on the
    Euclidean space, the unit sphere, and the space of symmetric positive
    definite matrices equipped with the Cholesky distance.

    Parameters
    ----------
    a, b : np.ndarray or torch.Tensor
        Input objects whose distance is to be computed.

        - If ``GPU=False``, then ``a`` and ``b`` are assumed to be NumPy arrays.
        - If ``GPU=True``, then ``a`` and ``b`` are assumed to be CUDA torch
          tensors.

        The admissible shapes depend on ``space_type``:

        - ``space_type="euclidean"``:
          ``(d,)`` or ``(M, d)``;
        - ``space_type="sphere"``:
          ``(d,)`` or ``(M, d)``;
        - ``space_type="spd"``:
          ``(p, p)`` or ``(M, p, p)``.

        In all cases, strict shape matching is imposed: no broadcasting is
        performed, and ``a`` and ``b`` must have identical shapes.

    space_type : {"euclidean", "sphere", "spd"}
        The metric space in which the distance is computed.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        The computed distance(s).

        - For a single matched pair, returns a scalar ``float``.
        - For a matched batch, returns a one-dimensional object:
            * a NumPy array when ``GPU=False``;
            * a torch tensor when ``GPU=True``.

    Raises
    ------
    ValueError
        If ``space_type`` is not one of ``"euclidean"``, ``"sphere"``, or
        ``"spd"``.

    Notes
    -----
    This function is a dispatcher only. All shape validation and metric-specific
    computations are delegated to the corresponding backend routine.
    """
    if space_type == "euclidean":
        return euclidean_distance(a, b, GPU=GPU)

    if space_type == "sphere":
        return sphere_geodesic_distance(a, b, GPU=GPU)

    if space_type == "spd":
        return spd_affine_invariant_distance(a, b, GPU=GPU)

    raise ValueError(
        "space_type must be one of {'euclidean', 'sphere', 'spd'}."
    )


# ============================================================
# Single-space ball indicator and empirical MDF
# ------------------------------------------------------------
# This section implements the ball indicator and the empirical
# metric distribution function on a single metric space.
#
# The indicator is defined by
#   delta(u, v, x) = 1{ d(u, x) <= d(u, v) + atol },
# where d denotes the metric associated with the specified
# space type.
#
# The corresponding empirical metric distribution function is
# obtained by averaging this indicator over the sample.
#
# Backend convention:
#   * If GPU=False, computations are carried out with NumPy on CPU.
#   * If GPU=True, computations are carried out with PyTorch on CUDA.
#
# Throughout, strict shape matching is imposed and no broadcasting
# is performed except where a single query point is explicitly
# expanded against a batch of observations.
# ============================================================


def delta_single(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    x: Union[np.ndarray, torch.Tensor],
    space_type: Literal["euclidean", "sphere", "spd"],
    atol: float = 1e-12,
    GPU: bool = False,
) -> Union[int, np.ndarray, torch.Tensor]:
    r"""
    Compute the single-space ball indicator
    ``delta(u, v, x) = 1{ d(u, x) <= d(u, v) + atol }``.

    This function supports three input regimes:

    (i) a single query, where ``u``, ``v``, and ``x`` have identical shapes;
    (ii) a batched evaluation in ``x``, where ``u`` and ``v`` represent single
    points and ``x`` is a batch of observations;
    (iii) a matched batched evaluation, where ``u``, ``v``, and ``x`` are all
    batches with identical shapes.

    Parameters
    ----------
    u, v, x : np.ndarray or torch.Tensor
        Input objects in a common metric space.

        - If ``space_type`` is ``"euclidean"`` or ``"sphere"``, the admissible
          shapes are ``(d,)`` and ``(M, d)``.
        - If ``space_type`` is ``"spd"``, the admissible shapes are
          ``(p, p)`` and ``(M, p, p)``.

        If ``GPU=False``, inputs are assumed to be NumPy arrays.
        If ``GPU=True``, inputs are assumed to be CUDA torch tensors.

    space_type : {"euclidean", "sphere", "spd"}
        The metric space in which the indicator is evaluated.

    atol : float, default=1e-12
        Absolute tolerance in the comparison
        ``d(u, x) <= d(u, v) + atol``.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    int or np.ndarray or torch.Tensor
        The indicator value(s).

        - Returns an ``int`` for a single query.
        - Returns a one-dimensional NumPy array when ``GPU=False`` and the
          evaluation is batched.
        - Returns a one-dimensional torch tensor when ``GPU=True`` and the
          evaluation is batched.

    Raises
    ------
    ValueError
        If the input shapes are incompatible.

    Notes
    -----
    This function assumes that the input types are already consistent with the
    selected backend. No implicit conversion between NumPy arrays and torch
    tensors is performed.
    """
    if x.ndim == u.ndim:
        if u.shape != v.shape or u.shape != x.shape:
            raise ValueError(
                f"When x.ndim == u.ndim, expected u, v, and x to have identical "
                f"shapes, but got u.shape={tuple(u.shape)}, "
                f"v.shape={tuple(v.shape)}, and x.shape={tuple(x.shape)}."
            )

        dux = compute_distance(u, x, space_type, GPU=GPU)
        duv = compute_distance(u, v, space_type, GPU=GPU)

        if GPU:
            if isinstance(dux, torch.Tensor):
                return (dux <= duv + atol).to(torch.int32)
            return int(dux <= duv + atol)

        if isinstance(dux, np.ndarray):
            return (dux <= duv + atol).astype(int)
        return int(dux <= duv + atol)

    if x.ndim == u.ndim + 1:
        if u.shape != v.shape:
            raise ValueError(
                f"When x is batched and u, v are single points, expected "
                f"u.shape == v.shape, but got {tuple(u.shape)} and {tuple(v.shape)}."
            )

        if GPU:
            U = u.unsqueeze(0).expand(x.shape[0], *u.shape)
            V = v.unsqueeze(0).expand(x.shape[0], *v.shape)
            dux = compute_distance(U, x, space_type, GPU=True)
            duv = compute_distance(U, V, space_type, GPU=True)
            return (dux <= duv + atol).to(torch.int32)

        U = np.broadcast_to(u, (x.shape[0],) + u.shape)
        V = np.broadcast_to(v, (x.shape[0],) + v.shape)
        dux = compute_distance(U, x, space_type, GPU=False)
        duv = compute_distance(U, V, space_type, GPU=False)
        return (dux <= duv + atol).astype(int)

    raise ValueError(
        f"Incompatible shapes for delta_single: "
        f"u.shape={tuple(u.shape)}, v.shape={tuple(v.shape)}, "
        f"x.shape={tuple(x.shape)}."
    )


def emdf_single(
    samples: Union[np.ndarray, torch.Tensor],
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    space_type: Literal["euclidean", "sphere", "spd"],
    atol: float = 1e-12,
    GPU: bool = False,
) -> float:
    r"""
    Empirical MDF with pre-stacked samples.
    """
    if samples.shape[0] == 0:
        raise ValueError("samples must be non-empty.")

    vals = delta_single(u, v, samples, space_type, atol=atol, GPU=GPU)

    if GPU:
        return float(vals.to(torch.float64).mean().item())

    return float(np.mean(vals))


# ============================================================
# Product-space indicator
# ------------------------------------------------------------
# This section implements the product-space ball indicator together
# with a helper routine for grouping components that share the same
# metric-space type.
#
# For each sample index m and component index k, a componentwise
# indicator is first computed in the corresponding metric space.
# The product-space indicator is then obtained by taking the product
# over all components k = 1, ..., K.
#
# To improve efficiency in repeated evaluations, components having
# the same metric-space type are grouped in advance and processed
# jointly in batch form.
#
# Backend convention:
#   * If GPU=False, computations are carried out with NumPy on CPU.
#   * If GPU=True, computations are carried out with PyTorch on CUDA.
#
# Throughout, inputs are assumed to represent matched batches in
# product space, with common leading shape (M, K, ...).
# ============================================================


def build_component_groups(
    space_types: Sequence[str],
) -> Dict[str, List[int]]:
    r"""
    Construct index groups for components sharing the same metric-space type.

    Given a sequence of component-wise space types, this function partitions
    the component indices into groups such that all indices in the same group
    correspond to the same metric-space type.

    Parameters
    ----------
    space_types : sequence of str
        A sequence of length ``K`` specifying the metric-space type for each
        component in a product space.

    Returns
    -------
    dict[str, list[int]]
        A dictionary mapping each distinct space type to a list of indices.

        Specifically, for each key ``s``, the value is a list of indices
        ``k`` such that ``space_types[k] == s``.

    Notes
    -----
    This function is intended to be called once and reused across repeated
    evaluations (e.g., in permutation tests or Monte Carlo simulations).
    By precomputing the grouping structure, one avoids repeated traversal
    of ``space_types`` inside performance-critical routines.
    """
    grouped_indices = defaultdict(list)

    for k, s in enumerate(space_types):
        grouped_indices[s].append(k)

    return dict(grouped_indices)


def delta_product(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    xyz: Union[np.ndarray, torch.Tensor],
    component_groups: Dict[str, List[int]],
    atol: float = 1e-12,
    GPU: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Compute the product-space ball indicator for a matched batch of inputs.

    For each sample index ``m = 1, \ldots, M``, this function evaluates

    .. math::
        \delta_m
        =
        \prod_{k=1}^K
        1\{ d_k(u_{m,k}, x_{m,k}) \le d_k(u_{m,k}, v_{m,k}) + \mathrm{atol} \},

    where ``K`` is the number of product components and ``d_k`` denotes the
    metric associated with the ``k``-th component.

    Parameters
    ----------
    u, v, xyz : np.ndarray or torch.Tensor
        Arrays/tensors of shape ``(M, K, ...)`` representing a matched batch
        of ``M`` observations in a product space.

        If ``GPU=False``, inputs are assumed to be NumPy arrays.
        If ``GPU=True``, inputs are assumed to be CUDA torch tensors.

    component_groups : dict[str, list[int]]
        A dictionary mapping each metric-space type to the list of component
        indices having that type.

        For example,
        ``{"euclidean": [0, 2], "sphere": [1], "spd": [3]}``
        indicates that components 0 and 2 are Euclidean, component 1 is
        spherical, and component 3 is SPD-valued.

    atol : float, default=1e-12
        Absolute tolerance in the comparison
        ``d_k(u_{m,k}, x_{m,k}) <= d_k(u_{m,k}, v_{m,k}) + atol``.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    np.ndarray or torch.Tensor
        A one-dimensional object of length ``M`` containing the product-space
        indicator values.

        - Returns a NumPy array when ``GPU=False``.
        - Returns a torch tensor when ``GPU=True``.

    Raises
    ------
    ValueError
        If the input shapes are incompatible, do not represent batched
        product-space observations, or if the component indices are invalid.

    Notes
    -----
    Components sharing the same metric-space type are processed jointly in
    batch form. The grouping structure is assumed to be precomputed, so that
    repeated evaluations can avoid repeated traversal of the component-wise
    space-type specification.
    """
    if u.shape != v.shape or u.shape != xyz.shape:
        raise ValueError(
            f"u, v, and xyz must have identical shapes, but got "
            f"{tuple(u.shape)}, {tuple(v.shape)}, and {tuple(xyz.shape)}."
        )

    if u.ndim < 3:
        raise ValueError(
            f"Expected input shape (M, K, ...), but got {tuple(u.shape)}."
        )

    M, K = u.shape[:2]
    tail_shape = tuple(u.shape[2:])

    if GPU:
        indicators = torch.empty((M, K), dtype=torch.int8, device=u.device)
    else:
        indicators = np.empty((M, K), dtype=np.int8)

    covered = set()

    for space_type, idx_list in component_groups.items():
        if len(idx_list) == 0:
            continue

        covered.update(idx_list)

        group_size = len(idx_list)

        u_sub = u[:, idx_list, ...]
        v_sub = v[:, idx_list, ...]
        xyz_sub = xyz[:, idx_list, ...]

        u_flat = u_sub.reshape(M * group_size, *tail_shape)
        v_flat = v_sub.reshape(M * group_size, *tail_shape)
        xyz_flat = xyz_sub.reshape(M * group_size, *tail_shape)

        dux = compute_distance(u_flat, xyz_flat, space_type, GPU=GPU)
        duv = compute_distance(u_flat, v_flat, space_type, GPU=GPU)

        if GPU:
            ind_sub = (dux <= duv + atol).to(torch.int8).reshape(M, group_size)
            indicators[:, idx_list] = ind_sub
        else:
            ind_sub = (dux <= duv + atol).astype(np.int8).reshape(M, group_size)
            indicators[:, idx_list] = ind_sub

    if covered != set(range(K)):
        missing = sorted(set(range(K)) - covered)
        raise ValueError(
            f"component_groups does not cover all component indices. "
            f"Missing indices: {missing}."
        )

    if GPU:
        return torch.prod(indicators, dim=1)

    return indicators.prod(axis=1)


# ============================================================
# Helper: stack component arrays into product-space samples
# ------------------------------------------------------------
# This helper constructs a single array/tensor S whose second axis
# indexes the components of a product space.
#
# Given K component arrays with a common leading dimension n,
# the function returns a stacked object of shape (n, K, ...),
# where each slice S[:, k, ...] corresponds to the k-th component.
#
# Backend convention:
#   * GPU=False -> NumPy (CPU)
#   * GPU=True  -> PyTorch (CUDA)
#
# This routine assumes that all inputs are already aligned and
# compatible for stacking.
# ============================================================


def stack_product_samples(
    *arrays: Union[np.ndarray, torch.Tensor],
    GPU: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Stack multiple component arrays into a unified product-space array.

    This function takes a collection of arrays/tensors representing different
    components of a product space and stacks them along a new axis so that
    the second dimension indexes the components.

    Parameters
    ----------
    *arrays : np.ndarray or torch.Tensor
        A variable number of input arrays/tensors, each representing one
        component of the product space.

        All inputs are assumed to:
        - have the same leading dimension ``n`` (sample size);
        - have identical trailing shapes.

        If ``GPU=False``, inputs are assumed to be NumPy arrays.
        If ``GPU=True``, inputs are assumed to be CUDA torch tensors.

    GPU : bool, default=False
        Indicator of the computational backend.

    Returns
    -------
    S : np.ndarray or torch.Tensor
        A stacked object of shape ``(n, K, ...)``, where ``K`` is the number
        of components.

        - Returns a NumPy array when ``GPU=False``.
        - Returns a torch tensor when ``GPU=True``.

    Notes
    -----
    This routine performs no runtime validation of input shapes for efficiency.
    In particular, it assumes that all inputs share the same leading dimension
    and are compatible for stacking. Violations of these assumptions may result
    in runtime errors raised by the underlying NumPy or PyTorch stack operation.
    """
    if GPU:
        return torch.stack(arrays, dim=1)

    return np.stack(arrays, axis=1)


# ============================================================
# Product-space empirical MDF
# ------------------------------------------------------------
# This section implements empirical metric distribution functions
# on product spaces, for both single query pairs and batched query
# pairs.
#
# Given product-space samples x_1, ..., x_n, the empirical MDF is
# obtained by averaging the product-space indicator over the sample.
# The underlying indicator is evaluated componentwise according to
# the precomputed grouping structure in ``component_groups``.
#
# The routine ``emdf_product`` handles a single query pair or a
# matched batch of query points, whereas
# ``emdf_product_pair_batch`` evaluates empirical MDF values for a
# batch of query pairs simultaneously.
#
# Backend convention:
#   * If GPU=False, computations are carried out with NumPy on CPU.
#   * If GPU=True, computations are carried out with PyTorch on CUDA.
#
# Throughout, product-space samples are assumed to have shape
# (n, K, ...), and query points are required to be compatible with
# this representation.
# ============================================================


def emdf_product_pair_batch(
    xyz_samples: Union[np.ndarray, torch.Tensor],
    u_batch: Union[np.ndarray, torch.Tensor],
    v_batch: Union[np.ndarray, torch.Tensor],
    component_groups: Dict[str, List[int]],
    atol: float = 1e-12,
    GPU: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Compute empirical metric distribution function values for a batch of
    product-space query pairs.

    For each query pair index ``b = 1, \ldots, B``, this function evaluates

    .. math::
        \hat{F}_n(u_b, v_b)
        =
        \frac{1}{n}
        \sum_{i=1}^n
        \delta(u_b, v_b, x_i),

    where ``x_1, \ldots, x_n`` are the observed product-space samples and
    ``\delta`` denotes the product-space indicator.

    Parameters
    ----------
    xyz_samples : np.ndarray or torch.Tensor
        Product-space samples of shape ``(n, K, ...)``.

        If ``GPU=False``, inputs are assumed to be NumPy arrays.
        If ``GPU=True``, inputs are assumed to be CUDA torch tensors.

    u_batch, v_batch : np.ndarray or torch.Tensor
        Batches of query points of shape ``(B, K, ...)``.

        The shapes of ``u_batch`` and ``v_batch`` must be identical, and each
        query point must have the same product-space shape as one sample point
        in ``xyz_samples``.

    component_groups : dict[str, list[int]]
        A dictionary mapping each metric-space type to the list of component
        indices having that type.

    atol : float, default=1e-12
        Absolute tolerance in the comparison used by the product-space
        indicator.

    GPU : bool, default=False
        Indicator of the computational backend. If ``False``, the computation
        is carried out with NumPy on CPU. If ``True``, the computation is
        carried out with PyTorch on CUDA.

    Returns
    -------
    np.ndarray or torch.Tensor
        A one-dimensional object of length ``B`` containing the empirical MDF
        values for the query pairs.

        - Returns a NumPy array when ``GPU=False``.
        - Returns a torch tensor when ``GPU=True``.

    Raises
    ------
    ValueError
        If the input shapes are incompatible.

    Notes
    -----
    This routine assumes that all inputs are already consistent with the
    selected backend. No implicit conversion between NumPy arrays and torch
    tensors is performed.

    The computation is vectorized over both the sample index and the query-pair
    index by forming a matched batch of size ``B * n``.
    """
    if xyz_samples.ndim < 3:
        raise ValueError(
            f"xyz_samples must have shape (n, K, ...), but got {tuple(xyz_samples.shape)}."
        )

    if u_batch.shape != v_batch.shape:
        raise ValueError(
            f"u_batch and v_batch must have identical shapes, but got "
            f"{tuple(u_batch.shape)} and {tuple(v_batch.shape)}."
        )

    point_shape = tuple(xyz_samples.shape[1:])   # (K, ...)
    if tuple(u_batch.shape[1:]) != point_shape:
        raise ValueError(
            f"Each query point must have shape {point_shape}, but got "
            f"{tuple(u_batch.shape[1:])}."
        )

    n = xyz_samples.shape[0]
    B = u_batch.shape[0]

    if GPU:
        # Shapes: (B, n, K, ...)
        u_exp = u_batch[:, None, ...].expand(B, n, *point_shape)
        v_exp = v_batch[:, None, ...].expand(B, n, *point_shape)
        x_exp = xyz_samples[None, :, ...].expand(B, n, *point_shape)
    else:
        batch_shape = (B, n) + point_shape
        u_exp = np.broadcast_to(u_batch[:, None, ...], batch_shape)
        v_exp = np.broadcast_to(v_batch[:, None, ...], batch_shape)
        x_exp = np.broadcast_to(xyz_samples[None, :, ...], batch_shape)

    vals = delta_product(
        u=u_exp.reshape(B * n, *point_shape),
        v=v_exp.reshape(B * n, *point_shape),
        xyz=x_exp.reshape(B * n, *point_shape),
        component_groups=component_groups,
        atol=atol,
        GPU=GPU,
    )

    if GPU:
        return vals.reshape(B, n).to(torch.float64).mean(dim=1)

    return vals.reshape(B, n).mean(axis=1)


def emdf_product(
    xyz_samples: Union[np.ndarray, torch.Tensor],
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    component_groups: Dict[str, List[int]],
    atol: float = 1e-12,
    GPU: bool = False,
) -> float:
    r"""
    Compute a single empirical metric distribution function value on a product
    space.

    This is a thin wrapper around ``emdf_product_pair_batch`` for the case of
    a single query pair.

    Parameters
    ----------
    xyz_samples : np.ndarray or torch.Tensor
        Product-space samples of shape ``(n, K, ...)``.

    u, v : np.ndarray or torch.Tensor
        Query points of shape ``(K, ...)``.

    component_groups : dict[str, list[int]]
        Precomputed grouping of component indices by metric-space type.

    atol : float, default=1e-12
        Absolute tolerance in the product-space indicator.

    GPU : bool, default=False
        Indicator of the computational backend.

    Returns
    -------
    float
        The empirical MDF value.
    """
    if GPU:
        u_batch = u.unsqueeze(0)
        v_batch = v.unsqueeze(0)
    else:
        u_batch = u[None, ...]
        v_batch = v[None, ...]

    out = emdf_product_pair_batch(
        xyz_samples=xyz_samples,
        u_batch=u_batch,
        v_batch=v_batch,
        component_groups=component_groups,
        atol=atol,
        GPU=GPU,
    )

    if GPU:
        return float(out[0].item())

    return float(out[0])