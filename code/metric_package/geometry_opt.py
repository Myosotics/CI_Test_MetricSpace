from typing import Literal, Union, Optional
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass


ArrayLike = Union[np.ndarray, torch.Tensor]


# ============================================================
# Data bundle utilities
# ------------------------------------------------------------
# This section defines lightweight containers for metric-space data.
#
# Supported spaces:
#   - Euclidean space;
#   - unit sphere;
#   - SPD manifold.
#
# BaseDataBundle stores ordinary array/tensor data for Euclidean and
# spherical samples. SPDDataBundle stores SPD matrices and optional
# precomputed quantities such as inverse square roots, eigendecompositions,
# and Cholesky factors.
#
# Backend convention:
#   - NumPy arrays remain NumPy arrays.
#   - Torch tensors remain torch tensors on their existing device.
# ============================================================


class DataBundle:
    r"""
    Abstract base class for metric-space data bundles.

    This class mainly provides a unified factory method
    ``from_data`` and a common ``slice`` interface.
    """

    def slice(self, idx) -> "DataBundle":
        r"""
        Return a sliced data bundle along the leading sample dimension.
        """
        raise NotImplementedError

    @staticmethod
    def from_data(
        X: ArrayLike,
        space_type: str,
        **kwargs,
    ) -> "DataBundle":
        r"""
        Construct the appropriate data bundle from raw data.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input data.

        space_type : {"euclidean", "sphere", "spd"}
            Metric space type.

        **kwargs
            Additional arguments passed to ``SPDDataBundle.from_matrix``
            when ``space_type="spd"``.

        Returns
        -------
        DataBundle
            ``BaseDataBundle`` for Euclidean/sphere data, and
            ``SPDDataBundle`` for SPD data.
        """
        if space_type == "spd":
            return SPDDataBundle.from_matrix(X, **kwargs)

        if space_type in {"euclidean", "sphere"}:
            return BaseDataBundle(data=X, space_type=space_type)

        raise ValueError("space_type must be one of {'euclidean', 'sphere', 'spd'}.")


@dataclass
class BaseDataBundle(DataBundle):
    r"""
    Data bundle for Euclidean or spherical samples.

    Attributes
    ----------
    data : np.ndarray or torch.Tensor
        Sample data.

        Shape:
        - ``(n, d)``
        - or more generally ``(*batch_shape, d)``

    space_type : {"euclidean", "sphere"}
        Metric space type.
    """

    data: ArrayLike
    space_type: str

    def __post_init__(self):
        r"""
        Validate data type and supported space type.
        """
        if self.data is None:
            raise ValueError("data must not be None.")

        if not isinstance(self.data, (np.ndarray, torch.Tensor)):
            raise TypeError("data must be np.ndarray or torch.Tensor.")

        if self.space_type not in {"euclidean", "sphere"}:
            raise ValueError("BaseDataBundle only supports euclidean or sphere.")

    @property
    def device(self):
        r"""
        Return the torch device, or ``"numpy"`` for NumPy arrays.
        """
        return self.data.device if isinstance(self.data, torch.Tensor) else "numpy"

    @property
    def dtype(self):
        r"""
        Return the dtype of the stored data.
        """
        return self.data.dtype

    def slice(self, idx) -> "BaseDataBundle":
        r"""
        Slice the bundle along the leading dimension.
        """
        return BaseDataBundle(
            data=self.data[idx],
            space_type=self.space_type,
        )


@dataclass
class SPDDataBundle(DataBundle):
    r"""
    Data bundle for SPD matrices with optional precomputed quantities.

    Attributes
    ----------
    matrix : np.ndarray or torch.Tensor
        SPD matrices.

        Shape:
        - ``(p, p)``
        - ``(*batch_shape, p, p)``

    inv_half : np.ndarray or torch.Tensor or None
        Inverse square roots of SPD matrices.

    eigvals : np.ndarray or torch.Tensor or None
        Eigenvalues of SPD matrices.

    eigvecs : np.ndarray or torch.Tensor or None
        Eigenvectors of SPD matrices.

    cholesky : np.ndarray or torch.Tensor or None
        Cholesky factors of SPD matrices.

    space_type : str, default="spd"
        Must be ``"spd"``.
    """

    matrix: ArrayLike
    inv_half: Optional[ArrayLike] = None
    eigvals: Optional[ArrayLike] = None
    eigvecs: Optional[ArrayLike] = None
    cholesky: Optional[ArrayLike] = None
    space_type: str = "spd"

    def __post_init__(self):
        r"""
        Validate matrix type, shape, and space type.
        """
        if self.matrix is None:
            raise ValueError("matrix must not be None.")

        if not isinstance(self.matrix, (np.ndarray, torch.Tensor)):
            raise TypeError("matrix must be np.ndarray or torch.Tensor.")

        if self.matrix.ndim < 2:
            raise ValueError("matrix must have shape (p, p) or (*batch_shape, p, p).")

        if self.matrix.shape[-1] != self.matrix.shape[-2]:
            raise ValueError(f"SPD matrices must be square, got {self.matrix.shape}.")

        if self.space_type != "spd":
            raise ValueError("SPDDataBundle only supports space_type='spd'.")

    @property
    def data(self) -> ArrayLike:
        r"""
        Alias for ``matrix`` to match the generic DataBundle interface.
        """
        return self.matrix

    @property
    def device(self):
        r"""
        Return the torch device, or ``"numpy"`` for NumPy arrays.
        """
        return self.matrix.device if isinstance(self.matrix, torch.Tensor) else "numpy"

    @property
    def dtype(self):
        r"""
        Return the dtype of the stored SPD matrices.
        """
        return self.matrix.dtype

    @classmethod
    def from_matrix(
        cls,
        X: ArrayLike,
        atol: float = 1e-12,
        compute_eig: bool = True,
        compute_inv_half: bool = True,
        compute_cholesky: bool = True,
    ) -> "SPDDataBundle":
        r"""
        Construct an SPD bundle from raw SPD matrices.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            SPD matrices with shape ``(p, p)`` or ``(*batch_shape, p, p)``.

        atol : float, default=1e-12
            Lower bound used when clamping eigenvalues.

        compute_eig : bool, default=True
            Whether to compute eigenvalues and eigenvectors.

        compute_inv_half : bool, default=True
            Whether to compute inverse square roots. Requires ``compute_eig=True``.

        compute_cholesky : bool, default=True
            Whether to compute Cholesky factors.

        Returns
        -------
        SPDDataBundle
            Bundle containing ``X`` and requested precomputed quantities.
        """
        if not compute_eig:
            compute_inv_half = False

        eigvals = None
        eigvecs = None
        inv_half = None
        cholesky = None

        if isinstance(X, torch.Tensor):
            if compute_eig:
                eigvals, eigvecs = torch.linalg.eigh(X)
                eigvals = torch.clamp(eigvals, min=atol)

                if compute_inv_half:
                    inv_sqrt = 1.0 / torch.sqrt(eigvals)
                    inv_half = torch.matmul(
                        eigvecs * inv_sqrt[..., None, :],
                        eigvecs.transpose(-1, -2),
                    )

            if compute_cholesky:
                cholesky = torch.linalg.cholesky(X)

        elif isinstance(X, np.ndarray):
            if compute_eig:
                eigvals, eigvecs = np.linalg.eigh(X)
                eigvals = np.clip(eigvals, atol, None)

                if compute_inv_half:
                    inv_sqrt = 1.0 / np.sqrt(eigvals)
                    inv_half = np.matmul(
                        eigvecs * inv_sqrt[..., None, :],
                        np.swapaxes(eigvecs, -1, -2),
                    )

            if compute_cholesky:
                cholesky = np.linalg.cholesky(X)

        else:
            raise TypeError("X must be np.ndarray or torch.Tensor.")

        return cls(
            matrix=X,
            inv_half=inv_half,
            eigvals=eigvals,
            eigvecs=eigvecs,
            cholesky=cholesky,
        )

    def slice(self, idx) -> "SPDDataBundle":
        r"""
        Slice all non-None SPD bundle fields along the leading dimension.
        """
        return SPDDataBundle(
            matrix=self.matrix[idx],
            inv_half=None if self.inv_half is None else self.inv_half[idx],
            eigvals=None if self.eigvals is None else self.eigvals[idx],
            eigvecs=None if self.eigvecs is None else self.eigvecs[idx],
            cholesky=None if self.cholesky is None else self.cholesky[idx],
            space_type=self.space_type,
        )


# ============================================================
# Distance functions on a single metric space
# ------------------------------------------------------------
# This section implements matched-sample distance functions for
# the metric spaces used in the simulations:
#   - Euclidean space;
#   - unit sphere;
#   - SPD manifold with affine-invariant distance.
#
# Inputs must have matching shapes and matching backends.
# NumPy inputs return NumPy outputs; torch inputs return torch outputs.
# ============================================================


def euclidean_distance(
    x: ArrayLike,
    y: ArrayLike,
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Compute Euclidean distances between matched inputs.

    Parameters
    ----------
    x, y : np.ndarray or torch.Tensor
        Inputs with identical shape.

        Supported shapes:
        - ``(d,)``
        - ``(*batch_shape, d)``

    Returns
    -------
    float or np.ndarray or torch.Tensor
        Euclidean distance.

        - Shape ``(d,)`` returns ``float``.
        - Shape ``(*batch_shape, d)`` returns ``(*batch_shape,)``.
    """
    if type(x) is not type(y):
        raise TypeError(
            "x and y must use the same backend."
        )

    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError(
            "x and y must be np.ndarray or torch.Tensor."
        )

    if x.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}."
        )

    if x.ndim < 1:
        raise ValueError(
            "Inputs must have shape (d,) or (*batch_shape, d)."
        )

    # ============================================================
    # NumPy backend
    # ============================================================
    if isinstance(x, np.ndarray):
        out = np.linalg.norm(x - y, axis=-1)
        if x.ndim == 1:
            return float(out.item())
        return out

    # ============================================================
    # Torch backend
    # ============================================================
    if x.device != y.device:
        raise ValueError(
            f"Device mismatch: "
            f"x.device={x.device}, y.device={y.device}."
        )
    if x.dtype != y.dtype:
        raise ValueError(
            f"Dtype mismatch: "
            f"x.dtype={x.dtype}, y.dtype={y.dtype}."
        )
    out = torch.linalg.norm(x - y, dim=-1)
    if x.ndim == 1:
        return float(out.item())
    return out


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
    Compute Cholesky distances between matched SPD matrices.

    Parameters
    ----------
    P1, P2 : np.ndarray or torch.Tensor
        SPD matrices with identical shape.

        Supported shapes:
        - ``(p, p)``
        - ``(M, p, p)``

    GPU : bool, default=False
        Legacy backend flag.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        Cholesky distance.

        - Shape ``(p, p)`` returns ``float``.
        - Shape ``(M, p, p)`` returns ``(M,)``.
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
    P1_inv_half: ArrayLike,
    P2_matrix: ArrayLike,
    atol: float = 1e-12,
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Compute affine-invariant distances between matched SPD matrices.

    Parameters
    ----------
    P1_inv_half : np.ndarray or torch.Tensor
        Inverse square root of the first SPD matrix.

        Shape:
        - ``(p, p)``
        - ``(*batch_shape, p, p)``

    P2_matrix : np.ndarray or torch.Tensor
        Second SPD matrix.

        Shape must match ``P1_inv_half``.

    atol : float, default=1e-12
        Lower bound applied to eigenvalues.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        Affine-invariant SPD distance.

        - Shape ``(p, p)`` returns ``float``.
        - Shape ``(*batch_shape, p, p)`` returns ``(*batch_shape,)``.
    """
    if type(P1_inv_half) is not type(P2_matrix):
        raise TypeError(
            "Bundle_P1.inv_half and Bundle_P2.matrix must have the same backend: "
            "both np.ndarray or both torch.Tensor."
        )

    if P1_inv_half.shape != P2_matrix.shape:
        raise ValueError(
            f"Shape mismatch: P1_inv_half has shape {tuple(P1_inv_half.shape)}, "
            f"but P2_matrix has shape {tuple(P2_matrix.shape)}."
        )

    # ============================================================
    # Torch backend
    # ============================================================
    if isinstance(P1_inv_half, torch.Tensor):

        if P1_inv_half.device != P2_matrix.device:
            raise ValueError(
                f"Device mismatch: "
                f"P1_inv_half.device={P1_inv_half.device}, "
                f"P2_matrix.device={P2_matrix.device}."
            )
        if P1_inv_half.dtype != P2_matrix.dtype:
            raise ValueError(
                f"Dtype mismatch: "
                f"P1_inv_half.dtype={P1_inv_half.dtype}, "
                f"P2_matrix.dtype={P2_matrix.dtype}."
            )

        if P1_inv_half.ndim == 2:
            P1_inv_half = P1_inv_half.unsqueeze(0)
            P2_matrix = P2_matrix.unsqueeze(0)

        p = P1_inv_half.shape[-1]

        G = P1_inv_half @ P2_matrix @ P1_inv_half
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

            out = torch.sqrt(torch.log(lam1) ** 2 + torch.log(lam2) ** 2)
        else:
            lam = torch.linalg.eigvalsh(G)
            lam = torch.clamp(lam, min=atol)
            out = torch.sqrt(torch.sum(torch.log(lam) ** 2, dim=-1))

        if P1_inv_half.ndim == 2:
            return float(out.item())
        
        return out

    # ============================================================
    # NumPy backend
    # ============================================================
    if isinstance(P1_inv_half, np.ndarray):

        if P1_inv_half.ndim == 2:
            P1_inv_half = P1_inv_half[None, ...]
            P2_matrix = P2_matrix[None, ...]

        p = P1_inv_half.shape[-1]

        G = P1_inv_half @ P2_matrix @ P1_inv_half
        G = 0.5 * (G + np.swapaxes(G, -1, -2))

        if p == 2:
            a = G[..., 0, 0]
            b = G[..., 0, 1]
            c = G[..., 1, 1]

            tr_half = 0.5 * (a + c)
            rad = np.sqrt(
                np.clip((0.5 * (a - c)) ** 2 + b ** 2, a_min=0.0, a_max=None)
            )

            lam1 = np.clip(tr_half - rad, a_min=atol, a_max=None)
            lam2 = np.clip(tr_half + rad, a_min=atol, a_max=None)

            out = np.sqrt(np.log(lam1) ** 2 + np.log(lam2) ** 2)
        else:
            lam = np.linalg.eigvalsh(G)
            lam = np.clip(lam, a_min=atol, a_max=None)
            out = np.sqrt(np.sum(np.log(lam) ** 2, axis=-1))
        
        if P1_inv_half.ndim == 2:
            return float(out.item())
        
        return out


def compute_distance(
    X: ArrayLike,
    Y: ArrayLike,
    space_type: Literal["euclidean", "sphere", "spd"],
) -> Union[float, np.ndarray, torch.Tensor]:
    r"""
    Dispatch to the distance function for the given metric space.

    Parameters
    ----------
    X, Y : np.ndarray or torch.Tensor
        Matched inputs with identical shape.

    space_type : {"euclidean", "sphere", "spd"}
        Metric space type.

    Returns
    -------
    float or np.ndarray or torch.Tensor
        Distance values.
    """
    if type(X) is not type(Y):
        raise TypeError(
            "X and Y must use the same backend."
        )

    if space_type == "euclidean":
        return euclidean_distance(X, Y)

    if space_type == "sphere":
        return sphere_geodesic_distance(X, Y)

    if space_type == "spd":
        return spd_affine_invariant_distance(X, Y)

    raise ValueError(
        "space_type must be one of "
        "{'euclidean', 'sphere', 'spd'}."
    )


# ============================================================
# Auxiliary utilities
# ------------------------------------------------------------
# This section provides small helper functions used by the
# statistic computation.
#
# Main utilities:
#   - _main_left:
#       return the left-side representation used in distance calls.
#
#   - _main_right:
#       return the right-side representation used in distance calls.
#
#   - broadcast_pair_array:
#       create pairwise or pairwise-Monte-Carlo broadcasted views.
#
# Backend convention:
#   - NumPy arrays use np.broadcast_to.
#   - Torch tensors use unsqueeze + expand.
# ============================================================


def _main_left(bundle: DataBundle) -> ArrayLike:
    r"""
    Return the left-side representation for distance computation.

    Parameters
    ----------
    bundle : DataBundle
        Input data bundle.

    Returns
    -------
    np.ndarray or torch.Tensor
        Left-side representation used in distance evaluation.

        - SPD bundle:
            returns ``bundle.inv_half``

        - Euclidean / spherical bundle:
            returns ``bundle.data``
    """
    if bundle.space_type == "spd":
        return bundle.inv_half
    return bundle.data


def _main_right(bundle: DataBundle) -> ArrayLike:
    r"""
    Return the right-side representation for distance computation.

    Parameters
    ----------
    bundle : DataBundle
        Input data bundle.

    Returns
    -------
    np.ndarray or torch.Tensor
        Right-side representation used in distance evaluation.

        - SPD bundle:
            returns ``bundle.matrix``

        - Euclidean / spherical bundle:
            returns ``bundle.data``
    """
    if bundle.space_type == "spd":
        return bundle.matrix
    return bundle.data


def broadcast_pair_array(
    X: ArrayLike | None,
    mode: str,
    rep: int | None = None,
) -> ArrayLike | None:
    r"""
    Broadcast an array/tensor for pairwise statistic computation.

    Parameters
    ----------
    X : np.ndarray, torch.Tensor, or None
        Input array/tensor.

        Expected shapes:
        - ``(n, *tail_shape)`` for ``mode="ij"``, ``"ji"``, ``"ijm"``
        - ``(n, M, *tail_shape)`` for ``mode="jim"``

    mode : {"ij", "ji", "ijm", "jim"}
        Broadcasting mode.

        - ``"ij"``:
            ``(n, *tail) -> (n, rep, *tail)``

        - ``"ji"``:
            ``(n, *tail) -> (rep, n, *tail)``

        - ``"ijm"``:
            ``(n, *tail) -> (n, n, rep, *tail)``

        - ``"jim"``:
            ``(n, M, *tail) -> (n, n, M, *tail)``

    rep : int or None, default=None
        Repetition size. Required for ``mode="ijm"``.
        For ``mode="ij"`` and ``"ji"``, defaults to ``n``.

    Returns
    -------
    np.ndarray, torch.Tensor, or None
        Broadcasted view with the requested pairwise shape.
    """
    if X is None:
        return None

    if not isinstance(X, (np.ndarray, torch.Tensor)):
        raise TypeError("X must be np.ndarray, torch.Tensor, or None.")
        
    if X.ndim < 2:
        raise ValueError("X must have shape (n, *tail_shape).")

    n = X.shape[0]

    if mode in {"ij", "ji"}:
        rep_eff = n if rep is None else rep
        if rep_eff <= 0:
            raise ValueError("rep must be positive.")

        if isinstance(X, torch.Tensor):
            if mode == "ij":
                return X[:, None, ...].expand(n, rep_eff, *X.shape[1:])
            return X[None, :, ...].expand(rep_eff, n, *X.shape[1:])
        
        if mode == "ij":
            return np.broadcast_to(
                X[:, None, ...],
                (n, rep_eff, *X.shape[1:]),
            )

        return np.broadcast_to(
            X[None, :, ...],
            (rep_eff, n, *X.shape[1:]),
        )

    elif mode == "ijm":
        if rep is None:
            raise ValueError("For mode='ijm', rep must be provided and represents M.")
        if rep <= 0:
            raise ValueError("rep must be positive.")

        if isinstance(X, torch.Tensor):
            return X[:, None, None, ...].expand(n, n, rep, *X.shape[1:])

        return np.broadcast_to(
            X[:, None, None, ...],
            (n, n, rep, *X.shape[1:]),
        )

    elif mode == "jim":
        if X.ndim < 3:
            raise ValueError("mode='jim' requires shape (n, M, *tail_shape).")

        M = X.shape[1]

        if isinstance(X, torch.Tensor):
            return X[None, ...].expand(n, n, M, *X.shape[2:])

        return np.broadcast_to(
            X[None, ...],
            (n, n, M, *X.shape[2:]),
        )
    else:
        raise ValueError("mode must be one of {'ij', 'ji', 'ijm', 'jim'}.")