import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple, Literal
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from metric_package.geometry_opt import (
    DataBundle, SPDDataBundle, BaseDataBundle
)

ArrayLike = np.ndarray | torch.Tensor


# ============================================================
# Transformation utilities
# ------------------------------------------------------------
# This section provides transformations between original
# metric-space data and Euclidean coordinates used for model
# training and sampling.
#
# Main utilities:
#   - spd_to_euclidean:
#       map SPD matrices to log-Cholesky Euclidean coordinates.
#
#   - euclidean_to_spd:
#       map log-Cholesky Euclidean coordinates back to SPD matrices.
#
#   - pre_transfer:
#       convert a DataBundle to torch Euclidean coordinates.
#
#   - post_transfer:
#       convert torch Euclidean coordinates back to the target backend.
#
# Backend convention:
#   - NumPy inputs may be converted to torch on the requested device.
#   - Torch inputs keep their existing device and dtype.
# ============================================================


def spd_to_euclidean(
    S: SPDDataBundle,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    r"""
    Map SPD matrices to log-Cholesky Euclidean coordinates.

    Parameters
    ----------
    S : SPDDataBundle
        SPD data bundle with ``S.matrix`` of shape
        ``(p, p)`` or ``(*batch_shape, p, p)``.

    dtype : torch.dtype, default=torch.float64
        Torch dtype used when converting NumPy data.

    device : str or torch.device, default="cuda"
        Torch device used when converting NumPy data.

    Returns
    -------
    torch.Tensor
        Log-Cholesky coordinates with shape
        ``(*batch_shape, q)``, where ``q = p(p+1)/2``.
    """
    if not isinstance(S, SPDDataBundle):
        raise TypeError("S must be an SPDDataBundle.")

    # ------------------------------------------------------------
    # NumPy backend: convert all cached fields to torch
    # ------------------------------------------------------------
    if isinstance(S.matrix, np.ndarray):
        device = torch.device(device)

        S.matrix = torch.as_tensor(S.matrix, dtype=dtype, device=device)

        if S.inv_half is not None:
            S.inv_half = torch.as_tensor(S.inv_half, dtype=dtype, device=device)

        if S.eigvals is not None:
            S.eigvals = torch.as_tensor(S.eigvals, dtype=dtype, device=device)

        if S.eigvecs is not None:
            S.eigvecs = torch.as_tensor(S.eigvecs, dtype=dtype, device=device)

        if S.cholesky is not None:
            S.cholesky = torch.as_tensor(S.cholesky, dtype=dtype, device=device)

    # ------------------------------------------------------------
    # Torch backend: preserve existing matrix device/dtype
    # ------------------------------------------------------------
    elif isinstance(S.matrix, torch.Tensor):
        device = S.matrix.device
        dtype = S.matrix.dtype

        if S.inv_half is not None:
            S.inv_half = S.inv_half.to(device=device, dtype=dtype)

        if S.eigvals is not None:
            S.eigvals = S.eigvals.to(device=device, dtype=dtype)

        if S.eigvecs is not None:
            S.eigvecs = S.eigvecs.to(device=device, dtype=dtype)

        if S.cholesky is not None:
            S.cholesky = S.cholesky.to(device=device, dtype=dtype)

    else:
        raise TypeError("S.matrix must be np.ndarray or torch.Tensor.")

    # ------------------------------------------------------------
    # Lazy Cholesky cache
    # ------------------------------------------------------------
    if S.cholesky is None:
        S.cholesky = torch.linalg.cholesky(S.matrix)

    L = S.cholesky

    # ------------------------------------------------------------
    # Log-Cholesky map
    # (*batch_shape, p, p) -> (*batch_shape, q)
    # ------------------------------------------------------------
    p = L.shape[-1]

    L_log = L.clone()

    diag_idx = torch.arange(p, device=L.device)
    L_log[..., diag_idx, diag_idx] = torch.log(L_log[..., diag_idx, diag_idx])

    tril_i, tril_j = torch.tril_indices(p, p, device=L.device)
    x = L_log[..., tril_i, tril_j]

    return x


def infer_p_from_q(q: int) -> int:
    r"""
    Infer SPD matrix dimension ``p`` from ``q = p(p+1)/2``.

    Parameters
    ----------
    q : int
        Number of lower-triangular log-Cholesky coordinates.

    Returns
    -------
    int
        SPD matrix dimension ``p``.
    """
    p = (math.isqrt(1 + 8 * q) - 1) // 2
    if p * (p + 1) // 2 != q:
        raise ValueError(
            f"Input length q={q} is invalid: it must satisfy q = p(p+1)/2 for some integer p."
        )
    return int(p)


def euclidean_to_spd(
    x: torch.Tensor,
    dtype_global: np.dtype | torch.dtype = np.float64,
    device_global: str | torch.device = "numpy",
) -> ArrayLike:
    r"""
    Map log-Cholesky Euclidean coordinates back to SPD matrices.

    Parameters
    ----------
    x : torch.Tensor
        Log-Cholesky coordinates with shape ``(q,)`` or
        ``(*batch_shape, q)``.

    dtype_global : numpy dtype or torch.dtype, default=np.float64
        Output dtype.

    device_global : {"numpy"} or torch.device, default="numpy"
        Output backend/device. If ``"numpy"``, returns ``np.ndarray``;
        otherwise returns ``torch.Tensor``.

    Returns
    -------
    np.ndarray or torch.Tensor
        SPD matrices with shape ``(p, p)`` or ``(*batch_shape, p, p)``.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")

    if x.ndim < 1:
        raise ValueError("x must have shape (q,) or (*batch_shape, q).")

    q = x.shape[-1]
    p = infer_p_from_q(q)

    batch_shape = x.shape[:-1]

    L_tilde = torch.zeros(
        (*batch_shape, p, p),
        dtype=x.dtype,
        device=x.device,
    )

    tril_i, tril_j = torch.tril_indices(p, p, device=x.device)
    L_tilde[..., tril_i, tril_j] = x

    L = L_tilde.clone()

    diag_idx = torch.arange(p, device=x.device)
    L[..., diag_idx, diag_idx] = torch.exp(L[..., diag_idx, diag_idx])

    S = L @ L.transpose(-1, -2)

    if device_global == "numpy":
        return S.detach().cpu().numpy().astype(dtype_global, copy=False)

    return S


def pre_transfer(
    Bundle_Z: DataBundle,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.device]:
    r"""
    Convert a DataBundle to torch Euclidean coordinates.

    Parameters
    ----------
    Bundle_Z : DataBundle
        Input data bundle.

    dtype : torch.dtype, default=torch.float64
        Torch dtype used when converting NumPy data.

    device : str, torch.device, or None, default=None
        Torch device used when converting NumPy data. If ``None``,
        uses CUDA when available, otherwise CPU.

    Returns
    -------
    Z_trf : torch.Tensor
        Torch Euclidean representation.

    device : torch.device
        Device of ``Z_trf``.
    """
    if isinstance(Bundle_Z.data, np.ndarray):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

    elif isinstance(Bundle_Z.data, torch.Tensor):
        device = Bundle_Z.data.device
        dtype = Bundle_Z.data.dtype

    else:
        raise TypeError("Bundle_Z.data must be either np.ndarray or torch.Tensor.")

    if Bundle_Z.space_type == "spd":
        return spd_to_euclidean(Bundle_Z, dtype=dtype, device=device), device

    if Bundle_Z.space_type in {"euclidean", "sphere"}:
        return torch.as_tensor(Bundle_Z.data, device=device, dtype=dtype), device

    raise NotImplementedError(
        f"pre_transfer not implemented for space_type={Bundle_Z.space_type}"
    )
    

def post_transfer(
    Z: torch.Tensor,
    space_type: Literal["euclidean", "sphere", "spd"] = "euclidean",
    dtype_global: np.dtype | torch.dtype | None = None,
    device_global: str | torch.device | None = None,
) -> ArrayLike:
    r"""
    Convert torch Euclidean coordinates back to the target space/backend.

    Parameters
    ----------
    Z : torch.Tensor
        Torch Euclidean representation.

    space_type : {"euclidean", "sphere", "spd"}, default="euclidean"
        Target metric space.

    dtype_global : numpy dtype, torch.dtype, or None, default=None
        Output dtype.

    device_global : {"numpy"} or torch.device or None, default=None
        Output backend/device. If ``"numpy"`` or ``None``, returns
        ``np.ndarray``.

    Returns
    -------
    np.ndarray or torch.Tensor
        Data in the requested target space and backend.
    """
    if device_global is None:
        device_global = "numpy"

    if device_global == "numpy":
        if dtype_global is None:
            dtype_global = np.float64
    else:
        if dtype_global is None:
            dtype_global = Z.dtype

    if space_type == "spd":
        return euclidean_to_spd(
            Z,
            dtype_global=dtype_global,
            device_global=device_global,
        )

    if space_type in {"euclidean", "sphere"}:
        if device_global == "numpy":
            return Z.detach().cpu().numpy().astype(dtype_global, copy=False)

        return Z

    raise NotImplementedError(
        f"post_transfer not implemented for space_type={space_type}"
    )


# ============================================================
# Data splitting utilities
# ------------------------------------------------------------
# This section provides helpers for splitting matched data bundles
# into training and validation subsets.
#
# A shared random permutation is used so that the correspondence
# among X, Y, and Z is preserved.
#
# Backend convention:
#   - device="numpy" returns NumPy index arrays.
#   - Torch devices return torch.LongTensor indices.
# ============================================================


def train_val_split_indices(
    n: int,
    val_ratio: float = 0.25,
    seed: int = 2026,
    device: str | torch.device = "numpy",
) -> tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    r"""
    Generate shared train/validation split indices.

    Parameters
    ----------
    n : int
        Number of samples.

    val_ratio : float, default=0.25
        Fraction assigned to validation.

    seed : int, default=2026
        Random seed.

    device : {"numpy", "cpu", "cuda"} or torch.device, default="numpy"
        Backend/device of returned indices.

    Returns
    -------
    train_idx, val_idx : tuple[np.ndarray, np.ndarray] or tuple[torch.Tensor, torch.Tensor]
        Split indices.

        - NumPy backend:
            arrays with dtype ``np.int64``.
        - Torch backend:
            tensors with dtype ``torch.long``.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must lie in the open interval (0, 1).")

    n_val = max(1, int(round(n * val_ratio)))

    if n > 1:
        n_val = min(n_val, n - 1)

    # ============================================================
    # NumPy backend
    # ============================================================

    if device == "numpy":

        rng = np.random.default_rng(seed)

        perm = rng.permutation(n)

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        return train_idx.astype(np.int64), val_idx.astype(np.int64)

    # ============================================================
    # Torch backend
    # ============================================================

    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    g = torch.Generator(device=device)

    g.manual_seed(seed)

    perm = torch.randperm(
        n,
        generator=g,
        device=device,
    )

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return train_idx, val_idx


def train_val_split_triplet(
    X_bundle: DataBundle,
    Y_bundle: DataBundle,
    Z_bundle: DataBundle,
    val_ratio: float = 0.25,
    seed: int = 2026,
) -> tuple[
    DataBundle, DataBundle, DataBundle,
    DataBundle, DataBundle, DataBundle,
]:
    r"""
    Split matched bundles ``(X, Y, Z)`` into train/validation subsets.

    Parameters
    ----------
    X_bundle, Y_bundle, Z_bundle : DataBundle
        Data bundles with the same leading sample size ``n``.

    val_ratio : float, default=0.25
        Fraction assigned to validation.

    seed : int, default=2026
        Random seed.

    Returns
    -------
    X_train, Y_train, Z_train, X_val, Y_val, Z_val : tuple[DataBundle, ...]
        Sliced bundles preserving the original concrete bundle types.
    """
    n = X_bundle.data.shape[0]

    train_idx, val_idx = train_val_split_indices(
        n=n,
        val_ratio=val_ratio,
        seed=seed,
        device=X_bundle.device,
    )

    X_train = X_bundle.slice(train_idx)
    Y_train = Y_bundle.slice(train_idx)
    Z_train = Z_bundle.slice(train_idx)

    X_val = X_bundle.slice(val_idx)
    Y_val = Y_bundle.slice(val_idx)
    Z_val = Z_bundle.slice(val_idx)

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val


# ============================================================
# Conditional sampling utilities
# ------------------------------------------------------------
# This section provides helper routines for conditional sample
# generation on metric spaces.
#
# Main utilities:
#   - make_generator:
#       create NumPy or torch random generators.
#
#   - generate_conditional_samples:
#       generate conditional samples given conditioning data.
#
# The interface is shared by oracle and fitted generators so
# that downstream simulation and testing code can use a unified
# sampling pipeline.
# ============================================================


def make_generator(
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> np.random.Generator | torch.Generator:
    r"""
    Create a NumPy or torch random number generator.

    Parameters
    ----------
    device : {"numpy", "cpu", "cuda"} or torch.device, default="cpu"
        Backend/device of the generator.

    seed : int or None, default=None
        Random seed.

    Returns
    -------
    np.random.Generator or torch.Generator
        Random number generator compatible with the selected backend.
    """
    # ============================================================
    # NumPy backend
    # ============================================================
    if device == "numpy":
        return np.random.default_rng(seed)

    # ============================================================
    # Torch backend
    # ============================================================
    device = torch.device(device)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def generate_conditional_samples(
    Bundle_Z: DataBundle,
    M: int,
    generators,
    generator: np.random.Generator | torch.Generator | None = None,
    bundle_kwargs: dict | None = None,
    **kwargs,
) -> tuple[DataBundle, DataBundle]:
    r"""
    Generate conditional samples given conditioning observations.

    Parameters
    ----------
    Bundle_Z : DataBundle
        Conditioning bundle with leading sample size ``n``.

    M : int
        Number of generated samples per observation.

    generators : callable
        Conditional generator callable.

    generator : np.random.Generator, torch.Generator, or None, default=None
        Random number generator passed to ``generators``.

    bundle_kwargs : dict or None, default=None
        Additional arguments passed to ``DataBundle.from_data``.

    **kwargs
        Additional keyword arguments passed to ``generators``.

    Returns
    -------
    Bundle_X, Bundle_Y : tuple[DataBundle, DataBundle]
        Generated conditional bundles with leading shape ``(n, M, ...)``.
    """
    if not isinstance(Bundle_Z, DataBundle):
        raise TypeError("Bundle_Z must be a DataBundle.")

    if bundle_kwargs is None:
        bundle_kwargs = {}

    space_type = Bundle_Z.space_type

    X_all, Y_all = generators(
        Bundle_Z,
        M=M,
        generator=generator,
        **kwargs,
    )

    if isinstance(X_all, DataBundle):
        Bundle_X = X_all
    else:
        Bundle_X = DataBundle.from_data(
            X_all,
            space_type=space_type,
            **bundle_kwargs,
        )

    if isinstance(Y_all, DataBundle):
        Bundle_Y = Y_all
    else:
        Bundle_Y = DataBundle.from_data(
            Y_all,
            space_type=space_type,
            **bundle_kwargs,
        )

    return Bundle_X, Bundle_Y


# ============================================================
# Oracle conditional generators
# ------------------------------------------------------------
# This section implements oracle conditional generators for the
# simulation models on metric spaces.
#
# The class provides a unified callable interface compatible with
# ``generate_conditional_samples``.
#
# Supported spaces:
#   - Euclidean space;
#   - unit sphere;
#   - SPD manifold.
#
# Backend convention:
#   - Input bundles may be NumPy or torch.
#   - Sampling is performed internally with torch on ``self.device``.
#   - Outputs are converted back to the original input backend.
# ============================================================


class OracleGenerators:
    r"""
    Oracle conditional generator for metric-space simulations.

    Parameters
    ----------
    sigma_perm : float, default=2.0
        Noise scale for Euclidean and spherical settings.

    space_type : str or None, default=None
        Metric space type.

    dtype : torch.dtype, default=torch.float64
        Internal torch dtype.

    device : torch.device or None, default=None
        Internal torch device.
    """
    def __init__(
        self,
        sigma_perm: float = 2.0,
        space_type: str | None = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        r"""
        Initialize an oracle generator object.
        """
        self.sigma_perm = sigma_perm
        self.space_type = space_type
        self.dtype = dtype
        self.device = device

    @classmethod
    def fit(
        cls,
        Bundle_Z: DataBundle,
        sigma_perm: float = 2.0,
        dtype: torch.dtype = torch.float64,
    ) -> "OracleGenerators":
        r"""
        Initialize oracle generator settings from conditioning data.

        Parameters
        ----------
        Bundle_Z : DataBundle
            Conditioning data bundle.

        sigma_perm : float, default=2.0
            Noise scale for Euclidean and spherical settings.

        dtype : torch.dtype, default=torch.float64
            Internal dtype used when NumPy data are converted to torch.

        Returns
        -------
        OracleGenerators
            Configured oracle generator.
        """
        Z = Bundle_Z.data
        if isinstance(Z, np.ndarray):
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                device = torch.device("cpu")
        elif isinstance(Z, torch.Tensor):
            device = Z.device
            dtype = Z.dtype
        else:
            raise TypeError("Bundle_Z.data must be np.ndarray or torch.Tensor.")

        return cls(
            sigma_perm=sigma_perm,
            space_type=Bundle_Z.space_type,
            dtype=dtype,
            device=device,
        )

    def _pre_transfer(
        self,
        bundle: DataBundle,
    ) -> DataBundle:
        r"""
        Convert all non-None bundle fields to torch tensors on ``self.device``.
        """
        if not isinstance(bundle, DataBundle):
            raise TypeError("bundle must be a DataBundle.")

        def _to_torch(x):
            if x is None:
                return None

            if isinstance(x, np.ndarray):
                return torch.as_tensor(
                    x,
                    dtype=self.dtype,
                    device=self.device,
                )

            if isinstance(x, torch.Tensor):
                if x.device != self.device:
                    raise ValueError(
                        f"Tensor device {x.device} does not match self.device={self.device}."
                    )
                if x.dtype != self.dtype:
                    raise ValueError(
                        f"Tensor dtype {x.dtype} does not match self.dtype={self.dtype}."
                    )
                return x
            raise TypeError("Bundle fields must be np.ndarray, torch.Tensor, or None.")
        
        if isinstance(bundle, BaseDataBundle):
            return BaseDataBundle(
                data=_to_torch(bundle.data),
                space_type=bundle.space_type,
            )
        
        if isinstance(bundle, SPDDataBundle):
            return SPDDataBundle(
                matrix=_to_torch(bundle.matrix),
                inv_half=_to_torch(bundle.inv_half),
                eigvals=_to_torch(bundle.eigvals),
                eigvecs=_to_torch(bundle.eigvecs),
                cholesky=_to_torch(bundle.cholesky),
                space_type=bundle.space_type,
            )
        
        raise TypeError("bundle must be BaseDataBundle or SPDDataBundle.")

    def _post_transfer(
        self,
        X: torch.Tensor,
        dtype_global,
        device_global,
    ) -> ArrayLike:
        r"""
        Convert torch output back to the original backend and dtype.
        """
        if device_global == "numpy":
            return X.detach().cpu().numpy().astype(dtype_global, copy=False)

        return X.to(
            device=device_global,
            dtype=dtype_global,
        )

    @staticmethod
    def _oracle_sample_chunk(
        Z_chunk: torch.Tensor,
        space_type: str,
        sigma: float,
        generator: torch.Generator,
        L_chunk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Generate one flattened chunk of oracle conditional samples.

        Parameters
        ----------
        Z_chunk : torch.Tensor
            Conditioning chunk.

            Shapes:
            - Euclidean / sphere: ``(b, d)``
            - SPD: ``(b, p, p)``

        space_type : {"euclidean", "sphere", "spd"}
            Metric space type.

        sigma : float
            Noise scale for Euclidean and spherical settings.

        generator : torch.Generator
            Torch random number generator.

        L_chunk : torch.Tensor or None, default=None
            Optional Cholesky factors for SPD conditioning matrices.

        Returns
        -------
        X_chunk, Y_chunk : tuple[torch.Tensor, torch.Tensor]
            Generated samples for the chunk.
        """
        device = Z_chunk.device
        dtype = Z_chunk.dtype

        if space_type == "euclidean":
            b, d = Z_chunk.shape

            U = torch.randn(
                b,
                d,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            V = torch.randn(
                b,
                d,
                device=device,
                dtype=dtype,
                generator=generator,
            )

            X_chunk = Z_chunk + sigma * U
            Y_chunk = Z_chunk + sigma * V

            return X_chunk, Y_chunk

        if space_type == "sphere":
            b, d = Z_chunk.shape
            if d < 2:
                raise ValueError("For spherical data, dimension must be at least 2.")

            eps = 1e-14
            GX = torch.randn(
                b,
                d,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            U = GX - torch.sum(GX * Z_chunk, dim=1, keepdim=True) * Z_chunk
            U_norm = torch.linalg.norm(U, dim=1, keepdim=True)
            GY = torch.randn(
                b,
                d,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            V = GY - torch.sum(GY * Z_chunk, dim=1, keepdim=True) * Z_chunk
            V_norm = torch.linalg.norm(V, dim=1, keepdim=True)

            bad_u = U_norm[:, 0] < eps
            while torch.any(bad_u):
                bad_idx = torch.where(bad_u)[0]

                U_new = torch.randn(
                    bad_idx.numel(),
                    d,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                Z_bad = Z_chunk[bad_idx]

                U_new = (
                    U_new
                    - torch.sum(U_new * Z_bad, dim=1, keepdim=True) * Z_bad
                )
                U_new_norm = torch.linalg.norm(U_new, dim=1, keepdim=True)

                good_new = U_new_norm[:, 0] >= eps
                good_idx = bad_idx[good_new]

                U[good_idx] = U_new[good_new]
                U_norm[good_idx] = U_new_norm[good_new]
                bad_u[good_idx] = False

            bad_v = V_norm[:, 0] < eps
            while torch.any(bad_v):
                bad_idx = torch.where(bad_v)[0]

                V_new = torch.randn(
                    bad_idx.numel(),
                    d,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                Z_bad = Z_chunk[bad_idx]

                V_new = (
                    V_new
                    - torch.sum(V_new * Z_bad, dim=1, keepdim=True) * Z_bad
                )
                V_new_norm = torch.linalg.norm(V_new, dim=1, keepdim=True)

                good_new = V_new_norm[:, 0] >= eps
                good_idx = bad_idx[good_new]

                V[good_idx] = V_new[good_new]
                V_norm[good_idx] = V_new_norm[good_new]
                bad_v[good_idx] = False

            U = U / U_norm
            V = V / V_norm

            xi_x = sigma * torch.randn(
                b,
                1,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            xi_y = sigma * torch.randn(
                b,
                1,
                device=device,
                dtype=dtype,
                generator=generator,
            )

            X_chunk = Z_chunk + xi_x * U
            Y_chunk = Z_chunk + xi_y * V

            X_chunk = X_chunk / torch.linalg.norm(X_chunk, dim=1, keepdim=True)
            Y_chunk = Y_chunk / torch.linalg.norm(Y_chunk, dim=1, keepdim=True)

            return X_chunk, Y_chunk

        if space_type == "spd":
            b, p, q = Z_chunk.shape

            if p != q:
                raise ValueError(f"SPD matrices must be square, got {(p, q)}.")

            nu = p + 6

            if L_chunk is None:
                L = torch.linalg.cholesky(Z_chunk)
            else:
                L = L_chunk

            LT = L.transpose(-1, -2)
            A = torch.randn(
                b,
                nu,
                p,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            B = torch.randn(
                b,
                nu,
                p,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            Sx = torch.matmul(A.transpose(-1, -2), A) / nu
            Sy = torch.matmul(B.transpose(-1, -2), B) / nu
            X_chunk = L @ Sx @ LT
            Y_chunk = L @ Sy @ LT

            return X_chunk, Y_chunk

        raise ValueError("space_type must be one of {'euclidean', 'sphere', 'spd'}.")

    def __call__(
        self,
        Bundle_Z: DataBundle,
        M: int,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Generate conditional samples for each conditioning observation.

        Parameters
        ----------
        Bundle_Z : DataBundle
            Conditioning data bundle with leading size ``n``.

        M : int
            Number of generated samples per observation.

        generator : torch.Generator or None, default=None
            Torch random number generator.

        chunk_size : int or None, default=None
            Number of flattened samples processed per chunk.

        Returns
        -------
        X_all, Y_all : tuple[np.ndarray, np.ndarray] or tuple[torch.Tensor, torch.Tensor]
            Generated conditional samples with leading shape ``(n, M, ...)``.
            Backend matches ``Bundle_Z.data``.
        """
        if not isinstance(Bundle_Z, DataBundle):
            raise TypeError("Bundle_Z must be a DataBundle.")

        dtype = self.dtype
        device = self.device

        dtype_global = Bundle_Z.data.dtype
        if isinstance(Bundle_Z.data, np.ndarray):
            device_global = "numpy"
        elif isinstance(Bundle_Z.data, torch.Tensor):
            device_global = Bundle_Z.data.device
        else:
            raise TypeError("Bundle_Z.data must be np.ndarray or torch.Tensor.")

        Bundle_Z_torch = self._pre_transfer(Bundle_Z)
        Z_torch = Bundle_Z_torch.data

        L_torch = None
        if (
            self.space_type == "spd"
            and isinstance(Bundle_Z_torch, SPDDataBundle)
            and Bundle_Z_torch.cholesky is not None
        ):
            L_torch = Bundle_Z_torch.cholesky

        n = Z_torch.shape[0]
        n_total = n * M

        if generator is None:
            generator = torch.Generator(device=device)

        if chunk_size is None:
            chunk_size = 1024

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        X_out = None
        Y_out = None
        x_tail_shape = None
        y_tail_shape = None

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)

            flat_idx = torch.arange(start, end, device=device)
            z_idx = torch.div(flat_idx, M, rounding_mode="floor")

            Z_chunk = Z_torch[z_idx]
            L_chunk = None if L_torch is None else L_torch[z_idx]

            X_chunk, Y_chunk = self._oracle_sample_chunk(
                Z_chunk=Z_chunk,
                space_type=self.space_type,
                sigma=self.sigma_perm,
                generator=generator,
                L_chunk=L_chunk,
            )

            if X_out is None:
                x_tail_shape = X_chunk.shape[1:]
                y_tail_shape = Y_chunk.shape[1:]

                X_out = torch.empty(
                    (n_total, *x_tail_shape),
                    device=device,
                    dtype=dtype,
                )
                Y_out = torch.empty(
                    (n_total, *y_tail_shape),
                    device=device,
                    dtype=dtype,
                )

            X_out[start:end] = X_chunk
            Y_out[start:end] = Y_chunk

        X_mat = X_out.reshape(n, M, *x_tail_shape)
        Y_mat = Y_out.reshape(n, M, *y_tail_shape)

        X_res = self._post_transfer(
            X_mat,
            dtype_global=dtype_global,
            device_global=device_global,
        )
        Y_res = self._post_transfer(
            Y_mat,
            dtype_global=dtype_global,
            device_global=device_global,
        )

        return X_res, Y_res


# ============================================================
# Utility networks
# ------------------------------------------------------------
# This section implements lightweight neural-network utilities
# used throughout the conditional generative models.
#
# In particular, it provides a configurable multi-layer
# perceptron (MLP) used as a generic function approximator,
# for example inside affine coupling layers to produce scale
# and shift parameters.
#
# The implementation is intentionally simple and modular so
# that it can be reused across different flow-based components.
# ============================================================


class MLP(nn.Module):
    r"""
    Configurable multi-layer perceptron (MLP).

    Parameters
    ----------
    in_dim : int
        Input dimension.

    out_dim : int
        Output dimension.

    hidden_dim : int, default=128
        Hidden-layer width.

    num_hidden_layers : int, default=2
        Number of hidden layers.

    activation : nn.Module or None, default=None
        Hidden-layer activation function.
        If None, ``nn.ReLU()`` is used.

    dropout : float, default=0.0
        Dropout probability applied after activations.

    Notes
    -----
    The final layer is linear and has no activation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        r"""
        Initialize the MLP architecture.
        """
        super().__init__()

        if activation is None:
            activation = nn.ReLU()

        layers: list[nn.Module] = []
        prev_dim: int = in_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))

        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Apply the forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with trailing dimension ``in_dim``.

        Returns
        -------
        torch.Tensor
            Output tensor with trailing dimension ``out_dim``.
        """
        return self.net(x)


# ============================================================
# Conditional affine coupling layer
# ------------------------------------------------------------
# This section implements one conditional affine coupling layer
# for RealNVP-style normalizing flows.
#
# Given input x and condition z, a binary mask keeps part of x
# unchanged and applies an affine transformation to the remaining
# coordinates. The scale and shift parameters are produced by an
# MLP using the masked input and condition.
#
# The layer provides:
#   - explicit forward transformation;
#   - explicit inverse transformation;
#   - efficient log-determinant computation.
# ============================================================


class ConditionalAffineCoupling(nn.Module):
    r"""
    Conditional affine coupling layer.

    Parameters
    ----------
    x_dim : int
        Dimension of input variable ``x``.

    z_dim : int
        Dimension of conditioning variable ``z``.

    mask : torch.Tensor
        Binary mask of shape ``(x_dim,)``.

    hidden_dim : int, default=128
        Hidden-layer width of the internal MLP.

    num_hidden_layers : int, default=2
        Number of hidden layers in the internal MLP.

    scale_limit : float, default=2.0
        Bound applied to the scale output after ``tanh``.

    dropout : float, default=0.0
        Dropout probability in the internal MLP.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        mask: torch.Tensor,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale_limit: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        r"""
        Initialize the conditional affine coupling layer.
        """
        super().__init__()

        if mask.ndim != 1 or mask.shape[0] != x_dim:
            raise ValueError(
                f"mask must have shape ({x_dim},), but got {tuple(mask.shape)}."
            )

        self.x_dim: int = x_dim
        self.z_dim: int = z_dim
        self.scale_limit: float = float(scale_limit)

        self.register_buffer("mask", mask.to(dtype=torch.float64))

        self.net: MLP = MLP(
            in_dim=x_dim + z_dim,
            out_dim=2 * x_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation=nn.ReLU(),
            dropout=dropout,
        )

    def _st(
        self,
        x_masked: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute scale and shift parameters.

        Parameters
        ----------
        x_masked : torch.Tensor
            Masked input with trailing dimension ``x_dim``.

        z : torch.Tensor
            Conditioning tensor with trailing dimension ``z_dim``.

        Returns
        -------
        s, t : tuple[torch.Tensor, torch.Tensor]
            Scale and shift tensors with trailing dimension ``x_dim``.
        """
        h = torch.cat([x_masked, z], dim=-1)
        st = self.net(h)
        s, t = torch.chunk(st, 2, dim=-1)

        s = (1.0 - self.mask) * torch.tanh(s) * self.scale_limit
        t = (1.0 - self.mask) * t
        return s, t

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the forward coupling transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with trailing dimension ``x_dim``.

        z : torch.Tensor
            Conditioning tensor with trailing dimension ``z_dim``.

        Returns
        -------
        y : torch.Tensor
            Transformed tensor with trailing dimension ``x_dim``.

        logdet : torch.Tensor
            Forward log-determinant with shape equal to the batch shape.
        """
        x_masked = self.mask * x
        s, t = self._st(x_masked, z)

        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = torch.sum(s, dim=-1)
        return y, logdet

    def inverse(
        self,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the inverse coupling transformation.

        Parameters
        ----------
        y : torch.Tensor
            Transformed tensor with trailing dimension ``x_dim``.

        z : torch.Tensor
            Conditioning tensor with trailing dimension ``z_dim``.

        Returns
        -------
        x : torch.Tensor
            Recovered input tensor with trailing dimension ``x_dim``.

        logdet_inv : torch.Tensor
            Inverse log-determinant with shape equal to the batch shape.
        """
        y_masked = self.mask * y
        s, t = self._st(y_masked, z)

        x = y_masked + (1.0 - self.mask) * (y - t) * torch.exp(-s)
        logdet_inv = -torch.sum(s, dim=-1)
        return x, logdet_inv


# ============================================================
# Conditional RealNVP model
# ------------------------------------------------------------
# This section implements a conditional RealNVP flow with two
# affine coupling layers and complementary masks.
#
# The model represents an invertible conditional map between
# latent Gaussian variables and target variables:
#   - forward:  u -> x given z
#   - inverse:  x -> u given z
#
# Main methods:
#   - log_prob:
#       evaluate conditional log-density log p(x | z).
#
#   - sample_tensor:
#       generate torch conditional samples.
#
#   - sample_numpy:
#       generate NumPy conditional samples.
# ============================================================


class ConditionalRealNVP(nn.Module):
    r"""
    Conditional RealNVP model with two affine coupling layers.

    Parameters
    ----------
    x_dim : int
        Dimension of target variable ``x``.

    z_dim : int
        Dimension of conditioning variable ``z``.

    hidden_dim : int, default=128
        Hidden-layer width in coupling networks.

    num_hidden_layers : int, default=2
        Number of hidden layers in coupling networks.

    scale_limit : float, default=2.0
        Bound applied to scale outputs.

    dropout : float, default=0.0
        Dropout probability in coupling networks.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale_limit: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        r"""
        Initialize the conditional RealNVP model.
        """
        super().__init__()

        if x_dim < 2:
            raise ValueError("x_dim must be at least 2.")

        self.x_dim: int = x_dim
        self.z_dim: int = z_dim

        mask1: torch.Tensor = self._make_alternating_mask(
            x_dim, start_with_one=True
        )
        mask2: torch.Tensor = 1.0 - mask1

        self.layers: nn.ModuleList = nn.ModuleList([
            ConditionalAffineCoupling(
                x_dim=x_dim,
                z_dim=z_dim,
                mask=mask1,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                scale_limit=scale_limit,
                dropout=dropout,
            ),
            ConditionalAffineCoupling(
                x_dim=x_dim,
                z_dim=z_dim,
                mask=mask2,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                scale_limit=scale_limit,
                dropout=dropout,
            ),
        ])

    @staticmethod
    def _make_alternating_mask(
        dim: int,
        start_with_one: bool = True,
    ) -> torch.Tensor:
        r"""
        Construct an alternating binary mask.

        Parameters
        ----------
        dim : int
            Mask length.

        start_with_one : bool, default=True
            Whether the mask starts with one.

        Returns
        -------
        torch.Tensor
            Binary mask of shape ``(dim,)``.
        """
        mask = torch.zeros(dim, dtype=torch.float64)
        if start_with_one:
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        return mask

    def _standard_normal_logprob(
        self,
        u: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute standard Gaussian log-density.

        Parameters
        ----------
        u : torch.Tensor
            Latent tensor with trailing dimension ``x_dim``.

        Returns
        -------
        torch.Tensor
            Log-density values with shape equal to the batch shape.
        """
        return -0.5 * (
            self.x_dim * math.log(2.0 * math.pi) + torch.sum(u ** 2, dim=-1)
        )

    def inverse(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Map observed variables to latent variables.

        Parameters
        ----------
        x : torch.Tensor
            Observed tensor of shape ``(batch_size, x_dim)``.

        z : torch.Tensor
            Conditioning tensor of shape ``(batch_size, z_dim)``.

        Returns
        -------
        u : torch.Tensor
            Latent tensor of shape ``(batch_size, x_dim)``.

        total_logdet_inv : torch.Tensor
            Inverse log-determinant of shape ``(batch_size,)``.
        """
        u = x
        total_logdet_inv = torch.zeros(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        for layer in reversed(self.layers):
            u, logdet_inv = layer.inverse(u, z)
            total_logdet_inv = total_logdet_inv + logdet_inv

        return u, total_logdet_inv

    def forward(
        self,
        u: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Map latent variables to observed variables.

        Parameters
        ----------
        u : torch.Tensor
            Latent tensor of shape ``(batch_size, x_dim)``.

        z : torch.Tensor
            Conditioning tensor of shape ``(batch_size, z_dim)``.

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape ``(batch_size, x_dim)``.

        total_logdet : torch.Tensor
            Forward log-determinant of shape ``(batch_size,)``.
        """
        x = u
        total_logdet = torch.zeros(
            u.shape[0],
            device=u.device,
            dtype=u.dtype,
        )

        for layer in self.layers:
            x, logdet = layer.forward(x, z)
            total_logdet = total_logdet + logdet

        return x, total_logdet

    def log_prob(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute conditional log-density ``log p(x | z)``.

        Parameters
        ----------
        x : torch.Tensor
            Observed tensor of shape ``(batch_size, x_dim)``.

        z : torch.Tensor
            Conditioning tensor of shape ``(batch_size, z_dim)``.

        Returns
        -------
        torch.Tensor
            Conditional log-density values of shape ``(batch_size,)``.
        """
        u, logdet_inv = self.inverse(x, z)
        log_base = self._standard_normal_logprob(u)
        return log_base + logdet_inv

    @torch.no_grad()
    def sample_tensor(
        self,
        z: torch.Tensor,
        n_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Generate conditional samples as a torch tensor.

        Parameters
        ----------
        z : torch.Tensor
            Conditioning tensor.

            Supported shapes:
            - ``(z_dim,)``
            - ``(batch_size, z_dim)``

        n_samples : int or None, default=None
            Number of samples when ``z`` is one-dimensional.

        generator : torch.Generator or None, default=None
            Random number generator.

        Returns
        -------
        torch.Tensor
            Generated samples.

            Shapes:
            - ``(n_samples, x_dim)`` for one-dimensional ``z``.
            - ``(batch_size, x_dim)`` for batched ``z``.
        """
        if z.ndim == 1:
            if n_samples is None:
                n_samples = 1
            z = z.unsqueeze(0).expand(n_samples, -1)
        elif z.ndim == 2:
            if n_samples is not None:
                raise ValueError("When z is batched, n_samples must be None.")
        else:
            raise ValueError("z must be one- or two-dimensional.")

        u = torch.randn(
            z.shape[0],
            self.x_dim,
            device=z.device,
            dtype=z.dtype,
            generator=generator,
        )

        x, _ = self.forward(u, z)
        return x

    @torch.no_grad()
    def sample_numpy(
        self,
        z: Union[np.ndarray, torch.Tensor],
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Generate conditional samples and return NumPy arrays.

        Parameters
        ----------
        z : np.ndarray or torch.Tensor
            Conditioning values with shape ``(z_dim,)`` or
            ``(batch_size, z_dim)``.

        n_samples : int or None, default=None
            Number of samples when ``z`` is one-dimensional.

        seed : int or None, default=None
            Random seed.

        Returns
        -------
        np.ndarray
            Generated samples with shape ``(n_samples, x_dim)`` or
            ``(batch_size, x_dim)``.
        """
        ref_param = next(self.parameters())
        device = ref_param.device
        dtype = ref_param.dtype

        if isinstance(z, torch.Tensor):
            z_t = z.to(device=device, dtype=dtype)
        else:
            z_t = torch.as_tensor(z, device=device, dtype=dtype)

        x_t = self.sample_tensor(z_t, n_samples=n_samples, seed=seed)
        return x_t.detach().cpu().numpy()


# ============================================================
# Training configuration
# ------------------------------------------------------------
# This section defines a lightweight configuration container
# for training conditional flow models.
#
# The configuration stores optimization hyperparameters such as:
#   - number of epochs;
#   - batch size;
#   - learning rate;
#   - weight decay;
#   - gradient clipping;
#   - learning-rate scheduling;
#   - early-stopping patience.
#
# The dataclass interface provides a compact and readable way
# to pass training settings across fitting routines.
# ============================================================


@dataclass
class FlowTrainConfig:
    r"""
    Training configuration for conditional flow models.

    Parameters
    ----------
    epochs : int, default=200
        Number of training epochs.

    batch_size : int, default=128
        Mini-batch size.

    lr : float, default=1e-3
        Initial learning rate.

    weight_decay : float, default=1e-5
        Weight decay coefficient.

    grad_clip_norm : float or None, default=5.0
        Gradient clipping threshold.
        If None, gradient clipping is disabled.

    scheduler_gamma : float, default=0.98
        Exponential learning-rate decay factor.

    verbose : bool, default=True
        Whether to print training progress.

    patience : int, default=20
        Early-stopping patience based on validation loss.
    """
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = 5.0
    scheduler_gamma: float = 0.98
    verbose: bool = True
    patience: int = 20

 
# ============================================================
# Training routines
# ------------------------------------------------------------
# This section implements maximum-likelihood training utilities
# for conditional RealNVP models.
#
# Main utilities:
#   - train_conditional_realnvp:
#       optimize an existing ConditionalRealNVP model.
#
#   - fit_conditional_realnvp:
#       construct a ConditionalRealNVP model from data dimensions
#       and train it.
#
# Training includes mini-batch optimization, optional validation,
# gradient clipping, learning-rate scheduling, and early stopping.
# ============================================================


def train_conditional_realnvp(
    model: ConditionalRealNVP,
    x_train: torch.Tensor,
    z_train: torch.Tensor,
    config: Optional[FlowTrainConfig] = None,
    x_val: Optional[torch.Tensor] = None,
    z_val: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    r"""
    Train a conditional RealNVP model by maximum likelihood.

    Parameters
    ----------
    model : ConditionalRealNVP
        Model to train.

    x_train : torch.Tensor
        Training samples with shape ``(n_train, x_dim)``.

    z_train : torch.Tensor
        Training conditions with shape ``(n_train, z_dim)``.

    config : FlowTrainConfig or None, default=None
        Training configuration. If None, uses ``FlowTrainConfig()``.

    x_val : torch.Tensor or None, default=None
        Optional validation samples with shape ``(n_val, x_dim)``.

    z_val : torch.Tensor or None, default=None
        Optional validation conditions with shape ``(n_val, z_dim)``.

    Returns
    -------
    Dict[str, object]
        Training history, including training loss, validation loss,
        best validation loss, and best epoch.
    """
    if config is None:
        config = FlowTrainConfig()

    if not isinstance(x_train, torch.Tensor) or not isinstance(z_train, torch.Tensor):
        raise TypeError("x_train and z_train must both be torch.Tensor.")

    if x_val is not None and not isinstance(x_val, torch.Tensor):
        raise TypeError("x_val must be torch.Tensor or None.")
    if z_val is not None and not isinstance(z_val, torch.Tensor):
        raise TypeError("z_val must be torch.Tensor or None.")

    if x_train.ndim != 2 or z_train.ndim != 2:
        raise ValueError("x_train and z_train must both be 2D.")

    if x_train.shape[0] != z_train.shape[0]:
        raise ValueError("x_train and z_train must have the same number of samples.")

    if x_train.shape[1] != model.x_dim:
        raise ValueError(f"x_train second dimension must be {model.x_dim}.")

    if z_train.shape[1] != model.z_dim:
        raise ValueError(f"z_train second dimension must be {model.z_dim}.")

    if (x_val is None) != (z_val is None):
        raise ValueError("x_val and z_val must either both be provided or both be None.")

    model.to(device=x_train.device, dtype=x_train.dtype)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.scheduler_gamma,
    )

    history: Dict[str, object] = {
        "train_nll": [],
        "val_nll": [],
        "best_val": None,
        "best_epoch": None,
    }

    patience = config.patience
    best_val = float("inf")
    best_state = None
    best_epoch = None
    counter = 0

    n_train = x_train.shape[0]

    for epoch in range(config.epochs):
        model.train()

        perm = torch.randperm(n_train, device=x_train.device)

        running_loss = 0.0
        n_seen = 0

        for start in range(0, n_train, config.batch_size):
            end = min(start + config.batch_size, n_train)
            idx = perm[start:end]

            xb = x_train[idx]
            zb = z_train[idx]

            optimizer.zero_grad()

            log_prob = model.log_prob(xb, zb)
            loss = -log_prob.mean()

            loss.backward()

            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.grad_clip_norm,
                )

            optimizer.step()

            bs = xb.shape[0]
            running_loss += float(loss.item()) * bs
            n_seen += bs

        scheduler.step()

        train_nll = running_loss / max(n_seen, 1)
        history["train_nll"].append(train_nll)

        if x_val is not None and z_val is not None:
            model.eval()
            with torch.no_grad():
                val_nll = float((-model.log_prob(x_val, z_val).mean()).item())

            history["val_nll"].append(val_nll)

            if val_nll < best_val:
                best_val = val_nll
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                counter = 0
            else:
                counter += 1

            if patience is not None and counter >= patience:
                if config.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        else:
            val_nll = None

        if config.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            if val_nll is None:
                print(f"[Epoch {epoch + 1:4d}] train_nll = {train_nll:.6f}")
            else:
                print(
                    f"[Epoch {epoch + 1:4d}] "
                    f"train_nll = {train_nll:.6f}, val_nll = {val_nll:.6f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)
        history["best_val"] = best_val
        history["best_epoch"] = best_epoch

        if config.verbose:
            print(
                f"Restored best model from epoch {best_epoch}, "
                f"best_val = {best_val:.6f}"
            )
    else:
        history["best_val"] = None
        history["best_epoch"] = None

    return history


def fit_conditional_realnvp(
    x_train: torch.Tensor,
    z_train: torch.Tensor,
    config: Optional[FlowTrainConfig] = None,
    hidden_dim: int = 128,
    num_hidden_layers: int = 2,
    scale_limit: float = 2.0,
    dropout: float = 0.0,
    x_val: Optional[torch.Tensor] = None,
    z_val: Optional[torch.Tensor] = None,
) -> tuple[ConditionalRealNVP, Dict[str, object]]:
    r"""
    Construct and train a conditional RealNVP model.

    Parameters
    ----------
    x_train : torch.Tensor
        Training samples with shape ``(n_train, x_dim)``.

    z_train : torch.Tensor
        Training conditions with shape ``(n_train, z_dim)``.

    config : FlowTrainConfig or None, default=None
        Training configuration. If None, uses ``FlowTrainConfig()``.

    hidden_dim : int, default=128
        Hidden-layer width.

    num_hidden_layers : int, default=2
        Number of hidden layers.

    scale_limit : float, default=2.0
        Bound applied to scale outputs.

    dropout : float, default=0.0
        Dropout probability.

    x_val : torch.Tensor or None, default=None
        Optional validation samples.

    z_val : torch.Tensor or None, default=None
        Optional validation conditions.

    Returns
    -------
    model : ConditionalRealNVP
        Trained conditional flow model.

    history : Dict[str, object]
        Training history returned by ``train_conditional_realnvp``.
    """
    if config is None:
        config = FlowTrainConfig()

    # ------------------------------------------------------------
    # Type validation
    # ------------------------------------------------------------
    if not isinstance(x_train, torch.Tensor) or not isinstance(z_train, torch.Tensor):
        raise TypeError("x_train and z_train must both be torch.Tensor.")

    if x_val is not None and not isinstance(x_val, torch.Tensor):
        raise TypeError("x_val must be torch.Tensor or None.")
    if z_val is not None and not isinstance(z_val, torch.Tensor):
        raise TypeError("z_val must be torch.Tensor or None.")


    # ------------------------------------------------------------
    # Input validation (must be (n_samples, dim))
    # ------------------------------------------------------------
    if x_train.ndim != 2 or z_train.ndim != 2:
        raise ValueError("x_train and z_train must both be 2D.")

    # ------------------------------------------------------------
    # Infer dimensions from data
    # ------------------------------------------------------------
    x_dim: int = x_train.shape[1]
    z_dim: int = z_train.shape[1]

    # ------------------------------------------------------------
    # Construct model
    # ------------------------------------------------------------
    model = ConditionalRealNVP(
        x_dim=x_dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        scale_limit=scale_limit,
        dropout=dropout,
    )

    # ------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------
    history = train_conditional_realnvp(
        model=model,
        x_train=x_train,
        z_train=z_train,
        config=config,
        x_val=x_val,
        z_val=z_val,
    )

    # ------------------------------------------------------------
    # Return trained model and training history
    # ------------------------------------------------------------
    return model, history


# ============================================================
# Fitted conditional generators
# ------------------------------------------------------------
# This section implements fitted conditional generators for
# estimating and sampling from X | Z and Y | Z.
#
# Two conditional RealNVP models are trained:
#   - one for X | Z;
#   - one for Y | Z.
#
# The class mirrors the interface of OracleGenerators, so it can
# be passed directly to generate_conditional_samples(...).
#
# For SPD data, samples are transformed to log-Cholesky Euclidean
# coordinates before fitting, and generated outputs are mapped
# back to SPD matrices after sampling.
# ============================================================


class FittedConditionalGenerators:
    r"""
    Fitted conditional generators for ``X | Z`` and ``Y | Z``.

    Parameters
    ----------
    x_model : ConditionalRealNVP
        Trained conditional flow model for ``X | Z``.

    y_model : ConditionalRealNVP
        Trained conditional flow model for ``Y | Z``.

    space_type : str
        Metric space type.

    dtype : torch.dtype, default=torch.float64
        Internal torch dtype.

    device : torch.device or None, default=None
        Internal torch device.
    """

    def __init__(
        self,
        x_model: ConditionalRealNVP,
        y_model: ConditionalRealNVP,
        space_type: str,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        r"""
        Initialize fitted conditional generators.
        """
        self.x_model = x_model
        self.y_model = y_model
        self.space_type = space_type
        self.dtype = dtype
        self.device = device

    @classmethod
    def fit(
        cls,
        Bundle_X: DataBundle,
        Bundle_Y: DataBundle,
        Bundle_Z: DataBundle,
        dtype: torch.dtype = torch.float64,
        config: Optional[FlowTrainConfig] = None,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale_limit: float = 2.0,
        dropout: float = 0.0,
        Bundle_X_val: Optional[DataBundle] = None,
        Bundle_Y_val: Optional[DataBundle] = None,
        Bundle_Z_val: Optional[DataBundle] = None,
    ) -> tuple["FittedConditionalGenerators", Dict[str, Dict[str, object]]]:
        r"""
        Fit conditional flow models for ``X | Z`` and ``Y | Z``.

        Parameters
        ----------
        Bundle_X, Bundle_Y, Bundle_Z : DataBundle
            Training data bundles with shared leading sample size ``n``.

        dtype : torch.dtype, default=torch.float64
            Internal torch dtype used when converting NumPy data.

        config : FlowTrainConfig or None, default=None
            Training configuration.

        hidden_dim : int, default=128
            Hidden-layer width.

        num_hidden_layers : int, default=2
            Number of hidden layers.

        scale_limit : float, default=2.0
            Bound applied to scale outputs.

        dropout : float, default=0.0
            Dropout probability.

        Bundle_X_val, Bundle_Y_val, Bundle_Z_val : DataBundle or None, default=None
            Optional validation bundles.

        Returns
        -------
        fitted : FittedConditionalGenerators
            Fitted generator wrapper.

        history : dict
            Training histories for the X-model and Y-model.
        """
        if (Bundle_X_val is None) != (Bundle_Z_val is None):
            raise ValueError(
                "Bundle_X_val and Bundle_Z_val must either both be provided or both be None."
            )
        if (Bundle_Y_val is None) != (Bundle_Z_val is None):
            raise ValueError(
                "Bundle_Y_val and Bundle_Z_val must either both be provided or both be None."
            )

        space_type = Bundle_X.space_type

        if config is None:
            config = FlowTrainConfig()

        X_val_trf = None
        Y_val_trf = None
        Z_val_trf = None

        X_trf, device = pre_transfer(Bundle_X, dtype=dtype)
        Y_trf, _ = pre_transfer(Bundle_Y, dtype=dtype, device=device)
        Z_trf, _ = pre_transfer(Bundle_Z, dtype=dtype, device=device)

        if Bundle_X_val is not None:
            X_val_trf, _ = pre_transfer(Bundle_X_val, dtype=dtype, device=device)
            Z_val_trf, _ = pre_transfer(Bundle_Z_val, dtype=dtype, device=device)
        if Bundle_Y_val is not None:
            Y_val_trf, _ = pre_transfer(Bundle_Y_val, dtype=dtype, device=device)
            if Z_val_trf is None:
                Z_val_trf, _ = pre_transfer(Bundle_Z_val, dtype=dtype, device=device)
        
        # ------------------------------------------------------------
        # Fit conditional model for X | Z
        # ------------------------------------------------------------
        x_model, x_history = fit_conditional_realnvp(
            x_train=X_trf,
            z_train=Z_trf,
            config=config,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            scale_limit=scale_limit,
            dropout=dropout,
            x_val=X_val_trf,
            z_val=Z_val_trf,
        )

        # ------------------------------------------------------------
        # Fit conditional model for Y | Z
        # ------------------------------------------------------------
        y_model, y_history = fit_conditional_realnvp(
            x_train=Y_trf,
            z_train=Z_trf,
            config=config,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            scale_limit=scale_limit,
            dropout=dropout,
            x_val=Y_val_trf,
            z_val=Z_val_trf,
        )

        # ------------------------------------------------------------
        # Wrap the trained models into a fitted generator object
        # ------------------------------------------------------------
        fitted = cls(
            x_model=x_model,
            y_model=y_model,
            space_type=space_type,
            dtype=dtype,
            device=device,
        )

        history = {
            "x_history": x_history,
            "y_history": y_history,
        }
        return fitted, history

    def __call__(
        self,
        Bundle_Z: DataBundle,
        M: int,
        generator: Optional[torch.Generator] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Generate conditional samples from the fitted models.

        Parameters
        ----------
        Bundle_Z : DataBundle
            Conditioning bundle with leading sample size ``n``.

        M : int
            Number of generated samples per conditioning observation.

        generator : torch.Generator or None, default=None
            Torch random number generator.

        chunk_size : int or None, default=None
            Number of flattened samples processed per chunk.

        Returns
        -------
        X_res, Y_res : tuple[np.ndarray, np.ndarray] or tuple[torch.Tensor, torch.Tensor]
            Generated conditional samples with leading shape ``(n, M, ...)``.
            Backend matches ``Bundle_Z.data``.
        """
        dtype = self.dtype
        device = self.device
        dtype_global = Bundle_Z.data.dtype
        if isinstance(Bundle_Z.data, np.ndarray):
            device_global = "numpy"
        elif Bundle_Z.data.device.type == "cpu":
            device_global = "cpu"
        else:   
            device_global = "cuda"

        Bundle_Z_mid = copy.deepcopy(Bundle_Z)
        Z_euc = pre_transfer(Bundle_Z_mid, dtype=dtype, device=device)[0]

        n: int = Z_euc.shape[0]
        n_total: int = n * M

        if generator is None:
            generator = torch.Generator(device=device)

        if chunk_size is None:
            chunk_size = 1024

        X_out = None
        Y_out = None

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)

            flat_idx = torch.arange(start, end, device=device)
            z_idx = torch.div(flat_idx, M, rounding_mode="floor")

            Z_chunk = Z_euc[z_idx]

            # flow sampling (Euclidean)
            X_chunk_euc = self.x_model.sample_tensor(
                z=Z_chunk,
                n_samples=None,
                generator=generator,
            )
            Y_chunk_euc = self.y_model.sample_tensor(
                z=Z_chunk,
                n_samples=None,
                generator=generator,
            )

            # --------------------------------------
            # 第一次 chunk：确定 p 并分配内存
            # --------------------------------------
            if X_out is None:
                x_tail_shape = X_chunk_euc.shape[1:]
                y_tail_shape = Y_chunk_euc.shape[1:]

                X_out = torch.empty((n_total, *x_tail_shape), device=device, dtype=dtype)
                Y_out = torch.empty((n_total, *y_tail_shape), device=device, dtype=dtype)

            # --------------------------------------
            # 直接写入（无 list）
            # --------------------------------------
            X_out[start:end] = X_chunk_euc
            Y_out[start:end] = Y_chunk_euc
        
        # reshape
        X_mat = X_out.reshape(n, M, *x_tail_shape)
        Y_mat = Y_out.reshape(n, M, *y_tail_shape)

        X_res = post_transfer(X_mat, space_type=self.space_type, dtype_global=dtype_global, device_global=device_global)
        Y_res = post_transfer(Y_mat, space_type=self.space_type, dtype_global=dtype_global, device_global=device_global)

        return X_res, Y_res