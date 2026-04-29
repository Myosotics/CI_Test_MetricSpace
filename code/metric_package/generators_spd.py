import torch
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple
import torch.nn as nn
import torch.optim as optim
from metric_package.statistics_computation_spd import (
    SPDDataBundle, Bundle_spd
)


# ============================================================
# Data splitting utilities
# ------------------------------------------------------------
# This section implements helper routines for slicing SPDDataBundle
# objects and for splitting paired SPD samples (X, Y, Z) into
# training and validation subsets.
#
# The splitting is performed through a shared random permutation
# so that the correspondence among X, Y, and Z is preserved.
# All routines operate directly on PyTorch tensors / SPDDataBundle
# objects and are designed for GPU-based workflows.
# ============================================================


def slice_spd_bundle(
    bundle: SPDDataBundle,
    idx: torch.Tensor,
) -> SPDDataBundle:
    r"""
    Slice an SPDDataBundle along the batch dimension.

    Parameters
    ----------
    bundle : SPDDataBundle
        Input bundle with batch size ``M``.

    idx : torch.Tensor
        Index tensor of shape ``(k,)`` with dtype ``torch.long``.
        It should be on the same device as ``bundle.matrix``.

    Returns
    -------
    SPDDataBundle
        Sliced bundle with batch size ``k``.
    """
    if idx.dtype != torch.long:
        raise ValueError("idx must have dtype torch.long.")

    if idx.device != bundle.matrix.device:
        raise ValueError("idx must be on the same device as bundle.matrix.")

    return SPDDataBundle(
        matrix=bundle.matrix[idx],
        inv_half=bundle.inv_half[idx],
        eigvals=bundle.eigvals[idx],
        eigvecs=bundle.eigvecs[idx],
        cholesky=bundle.cholesky[idx],
    )


def train_val_split_indices(
    n: int,
    val_ratio: float = 0.25,
    seed: int = 2026,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Generate shared train/validation indices.

    Parameters
    ----------
    n : int
        Number of samples.

    val_ratio : float, default=0.25
        Fraction of samples assigned to the validation set.

    seed : int, default=2026
        Random seed for reproducibility.

    device : torch.device or None, default=None
        Device of the returned index tensors.

    Returns
    -------
    train_idx : torch.Tensor
        Training indices of shape ``(n_train,)`` and dtype ``torch.long``.

    val_idx : torch.Tensor
        Validation indices of shape ``(n_val,)`` and dtype ``torch.long``.

    Notes
    -----
    At least one validation sample is always included.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must lie in the open interval (0, 1).")

    n_val: int = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1) if n > 1 else 1

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    perm: torch.Tensor = torch.randperm(n, generator=g, device=device)

    val_idx: torch.Tensor = perm[:n_val]
    train_idx: torch.Tensor = perm[n_val:]

    return train_idx, val_idx


def train_val_split_spd_triplet(
    X_bundle: SPDDataBundle,
    Y_bundle: SPDDataBundle,
    Z_bundle: SPDDataBundle,
    val_ratio: float = 0.25,
    seed: int = 2026,
) -> Tuple[
    SPDDataBundle, SPDDataBundle, SPDDataBundle,
    SPDDataBundle, SPDDataBundle, SPDDataBundle,
]:
    r"""
    Split paired SPD bundles ``(X, Y, Z)`` into training and validation subsets
    using a shared random permutation.

    Parameters
    ----------
    X_bundle : SPDDataBundle
        Bundle corresponding to ``X`` with batch size ``M``.

    Y_bundle : SPDDataBundle
        Bundle corresponding to ``Y`` with the same batch size as ``X_bundle``.

    Z_bundle : SPDDataBundle
        Bundle corresponding to ``Z`` with the same batch size as ``X_bundle``.

    val_ratio : float, default=0.25
        Fraction of samples assigned to the validation set.

    seed : int, default=2026
        Random seed for reproducibility.

    Returns
    -------
    X_train : SPDDataBundle
        Training subset of ``X_bundle``.

    Y_train : SPDDataBundle
        Training subset of ``Y_bundle``.

    Z_train : SPDDataBundle
        Training subset of ``Z_bundle``.

    X_val : SPDDataBundle
        Validation subset of ``X_bundle``.

    Y_val : SPDDataBundle
        Validation subset of ``Y_bundle``.

    Z_val : SPDDataBundle
        Validation subset of ``Z_bundle``.

    Notes
    -----
    A shared random permutation is used so that the triplet correspondence
    ``(X_i, Y_i, Z_i)`` is preserved after splitting.
    """
    n: int = X_bundle.matrix.shape[0]

    if Y_bundle.matrix.shape[0] != n or Z_bundle.matrix.shape[0] != n:
        raise ValueError(
            "X_bundle, Y_bundle, and Z_bundle must have the same batch size."
        )

    device = X_bundle.matrix.device
    if Y_bundle.matrix.device != device or Z_bundle.matrix.device != device:
        raise ValueError(
            "X_bundle, Y_bundle, and Z_bundle must be on the same device."
        )

    train_idx, val_idx = train_val_split_indices(
        n=n,
        val_ratio=val_ratio,
        seed=seed,
        device=device,
    )

    X_train: SPDDataBundle = slice_spd_bundle(X_bundle, train_idx)
    Y_train: SPDDataBundle = slice_spd_bundle(Y_bundle, train_idx)
    Z_train: SPDDataBundle = slice_spd_bundle(Z_bundle, train_idx)

    X_val: SPDDataBundle = slice_spd_bundle(X_bundle, val_idx)
    Y_val: SPDDataBundle = slice_spd_bundle(Y_bundle, val_idx)
    Z_val: SPDDataBundle = slice_spd_bundle(Z_bundle, val_idx)

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val


# ============================================================
# Conditional sampling utilities
# ------------------------------------------------------------
# This section provides helper routines for conditional sample
# generation in the SPD setting.
#
# The main entry point is `generate_conditional_samples(...)`,
# which takes a batch of conditioning SPD matrices together with
# a generator object (oracle or fitted) and returns generated
# conditional samples in bundle form.
#
# The interface is designed so that downstream simulation and
# testing routines can call a unified sampling wrapper without
# depending on the concrete generation mechanism.
# ============================================================


def make_generator(
    device: torch.device,
    seed: int | None = None,
) -> torch.Generator:
    r"""
    Create a torch random number generator on the specified device.

    Parameters
    ----------
    device : torch.device
        Device on which the generator will be used.

    seed : int or None, default=None
        Optional seed for reproducibility.

    Returns
    -------
    torch.Generator
        Random number generator bound to the given device.
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def generate_conditional_samples(
    Bundle_Z: SPDDataBundle,
    M: int,
    generators,
    generator: torch.Generator | None = None,
    **kwargs,
) -> tuple[SPDDataBundle, SPDDataBundle]:
    r"""
    Generate conditional SPD samples for a batch of conditioning values.

    For each conditioning matrix ``Z_i``, this function generates ``M``
    conditional sample pairs from the user-supplied generator.

    Parameters
    ----------
    Bundle_Z : SPDDataBundle
        Bundle of conditioning SPD matrices.

        Shape of ``Bundle_Z.matrix``:
        - ``(n, p, p)``

    M : int
        Number of conditional replications generated for each observation.

    generators : callable
        Conditional generator with signature

        ``generators(Bundle_Z, M=..., generator=..., **kwargs)``

        The generator may return either:

        - two tensors of shape ``(n, M, p, p)``, or
        - two ``SPDDataBundle`` objects whose ``matrix`` fields have shape
          ``(n, M, p, p)``.

    generator : torch.Generator or None, default=None
        Random number generator used in conditional sampling.

    **kwargs
        Additional keyword arguments passed to the generator.

    Returns
    -------
    Bundle_X : SPDDataBundle
        Bundle of generated conditional samples for ``X``.

    Bundle_Y : SPDDataBundle
        Bundle of generated conditional samples for ``Y``.

    Notes
    -----
    This function provides a unified sampling wrapper so that downstream
    routines can work with either oracle generators or fitted generators
    through the same interface.
    """
    if not isinstance(Bundle_Z, SPDDataBundle):
        raise TypeError("Bundle_Z must be an SPDBundle.")

    if Bundle_Z.matrix.ndim != 3:
        raise ValueError("Bundle_Z.matrix must have shape (n, p, p).")

    if M <= 0:
        raise ValueError("M must be a positive integer.")

    n, p, q = Bundle_Z.matrix.shape
    if p != q:
        raise ValueError(
            f"Each conditioning matrix must be square, but got shape {(p, q)}."
        )

    X_all, Y_all = generators(
        Bundle_Z,
        M=M,
        generator=generator,
        **kwargs,
    )

    if X_all.shape != (n, M, p, p) or Y_all.shape != (n, M, p, p):
        raise ValueError(
            "Generator must return tensors of shape (n, M, p, p)."
        )

    X_bundle = Bundle_spd(X_all)
    Y_bundle = Bundle_spd(Y_all)

    return X_bundle, Y_bundle


# ============================================================
# Oracle conditional generators
# ------------------------------------------------------------
# This class implements oracle conditional generators for the
# SPD simulation model considered in the paper.
#
# Given a batch of conditioning SPD matrices Z, the generator
# produces M conditional sample pairs for each conditioning
# value using the exact model-based conditional law.
#
# The implementation is batch-oriented and GPU-based. To reduce
# memory usage when nM is large, generation is performed in
# chunks along the flattened replication dimension.
#
# The returned samples are compatible with the unified
# conditional sampling interface used throughout this module.
# ============================================================


class OracleGenerators:
    r"""
    Oracle conditional generator for the SPD case.

    This class generates conditional sample pairs ``(X, Y)`` from the
    oracle conditional law given a batch of SPD conditioning matrices
    ``Z``. For each conditioning matrix ``Z_i``, the generator produces
    ``M`` independent conditional replications.

    The conditional law is defined through Wishart-type perturbations.
    Given an SPD matrix ``Z``, let ``L`` be its Cholesky factor such that
    ``Z = L L^T``. Then conditional samples are generated as

    .. math::
        X = L S_x L^T, \qquad Y = L S_y L^T,

    where ``S_x`` and ``S_y`` are independent Wishart-type random matrices
    constructed from Gaussian factors.

    Notes
    -----
    This implementation is GPU-only, fully batch-oriented, and supports
    chunked generation along the flattened replication dimension.
    """

    def __init__(self, sigma_perm: float = 2.0) -> None:
        self.sigma_perm = sigma_perm

    def __call__(
        self,
        Bundle_Z: SPDDataBundle,
        M: int,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Generate batched conditional samples from the oracle SPD model.

        Parameters
        ----------
        Bundle_Z : SPDDataBundle
            Batch of conditioning SPD matrices.

            Shape of ``Bundle_Z.matrix``:
            - ``(n, p, p)``

        M : int
            Number of conditional replications generated for each conditioning
            matrix.

        generator : torch.Generator or None, default=None
            Random number generator used for Gaussian sampling.

        chunk_size : int or None, default=None
            Number of replicated conditioning points processed at a time after
            flattening the leading batch dimensions ``(n, M)`` into a single
            dimension of size ``nM``. If None, a default chunk size is used.

        Returns
        -------
        X_all : torch.Tensor
            Generated conditional samples for ``X`` with shape ``(n, M, p, p)``.

        Y_all : torch.Tensor
            Generated conditional samples for ``Y`` with shape ``(n, M, p, p)``.

        Notes
        -----
        The generation is performed by flattening the replication structure
        ``(n, M)`` into a single dimension and processing it in chunks. This
        reduces peak memory usage when ``nM`` is large.
        """

        Z = Bundle_Z.matrix
        L = Bundle_Z.cholesky

        if Z.ndim != 3:
            raise ValueError("Bundle_Z.matrix must have shape (n, p, p).")

        if not Z.is_cuda:
            raise RuntimeError("Bundle_Z must be on CUDA.")

        n, p, q = Z.shape
        device = Z.device

        if p != q:
            raise ValueError(f"Each SPD matrix must be square, but got {(p, q)}.")

        if M <= 0:
            raise ValueError("M must be positive.")

        dtype = Z.dtype
        nu = p + 6
        n_total = n * M

        if generator is None:
            generator = torch.Generator(device=device)

        if chunk_size is None:
            chunk_size = 1024

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        X_out = torch.empty((n_total, p, p), device=device, dtype=dtype)
        Y_out = torch.empty((n_total, p, p), device=device, dtype=dtype)

        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            b = end - start

            # flat indices in {0, ..., nM-1}
            flat_idx = torch.arange(start, end, device=device)

            # map flat index -> conditioning index in {0, ..., n-1}
            z_idx = torch.div(flat_idx, M, rounding_mode="floor")

            # select only the needed Cholesky factors
            L_chunk = L[z_idx]                    # (b, p, p)
            LT_chunk = L_chunk.transpose(-1, -2) # (b, p, p)

            # Gaussian matrix factors
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

            # Wishart-type factors
            Sx = torch.matmul(A.transpose(-1, -2), A) / nu   # (b, p, p)
            Sy = torch.matmul(B.transpose(-1, -2), B) / nu   # (b, p, p)

            # Congruence transforms
            X_out[start:end] = torch.matmul(torch.matmul(L_chunk, Sx), LT_chunk)
            Y_out[start:end] = torch.matmul(torch.matmul(L_chunk, Sy), LT_chunk)

        X_all = X_out.reshape(n, M, p, p)
        Y_all = Y_out.reshape(n, M, p, p)

        return X_all, Y_all


# ============================================================
# SPD <-> Euclidean transform
# ============================================================


def spd_to_euclidean(
    S: SPDDataBundle,
) -> torch.Tensor:
    r"""
    Map SPD matrix/matrices to Euclidean coordinates via the log-Cholesky map.

    Parameters
    ----------
    S : SPDDataBundle
        Precomputed SPD bundle on GPU.

        Shape of ``S.matrix``:
        - ``(*batch_shape, p, p)``

    Returns
    -------
    torch.Tensor
        Euclidean coordinates under the log-Cholesky map.

        Shape:
        - ``(*batch_shape, q)``

        where ``q = p(p+1)/2``.
    """
    if not isinstance(S, SPDDataBundle):
        raise TypeError("S must be an SPDDataBundle.")

    L: torch.Tensor = S.cholesky

    p, _ = L.shape[-2:]

    # Use the cached Cholesky factor and apply log to its diagonal.
    L_log: torch.Tensor = L.clone()
    diag_idx = torch.arange(p, device=L.device)
    L_log[..., diag_idx, diag_idx] = torch.log(L_log[..., diag_idx, diag_idx])

    # Extract lower-triangular entries.
    tril_i, tril_j = torch.tril_indices(p, p, device=L.device)
    x: torch.Tensor = L_log[..., tril_i, tril_j]

    return x


def infer_p_from_q(q: int) -> int:
    r"""
    Infer the SPD matrix size ``p`` from ``q = p(p+1)/2``.

    Parameters
    ----------
    q : int
        Length of the Euclidean coordinate vector.

    Returns
    -------
    int
        Matrix size ``p``.
    """
    p = (math.isqrt(1 + 8 * q) - 1) // 2
    if p * (p + 1) // 2 != q:
        raise ValueError(
            f"Input length q={q} is invalid: it must satisfy q = p(p+1)/2 "
            f"for some integer p."
        )
    return int(p)


def euclidean_to_spd(
    x: torch.Tensor,
    atol: float = 1e-12,
    bundle: bool = True,
) -> Union[SPDDataBundle, torch.Tensor]:
    r"""
    Map Euclidean coordinates back to SPD matrix/matrices via the inverse
    log-Cholesky map.

    Parameters
    ----------
    x : torch.Tensor
        Euclidean coordinates on GPU.

        Shape:
        - ``(q,)``, or
        - ``(*batch_shape, q)``

        where ``q = p(p+1)/2`` for some integer ``p``.

    atol : float, default=1e-12
        Lower bound applied to eigenvalues for numerical stability when
        constructing the SPD bundle (only used if ``bundle=True``).

    bundle : bool, default=True
        If True, return a ``SPDDataBundle`` with precomputed quantities.
        If False, return only the SPD matrix/matrices as a tensor.

    Returns
    -------
    SPDDataBundle or torch.Tensor
        - If ``bundle=True``:
            returns a bundle with fields (matrix, inv_half, eigvals, ...)
        - If ``bundle=False``:
            returns SPD matrices of shape:
                ``(p, p)`` or ``(*batch_shape, p, p)``
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")

    if not x.is_cuda:
        raise RuntimeError("x must be a CUDA tensor.")

    if x.ndim < 1:
        raise ValueError("x must have shape (q,) or (*batch_shape, q).")

    q: int = x.shape[-1]
    p: int = infer_p_from_q(q)


    batch_shape = x.shape[:-1]

    # Fill the lower-triangular entries of the log-Cholesky factor.
    L_tilde = torch.zeros(
        (*batch_shape, p, p),
        dtype=x.dtype,
        device=x.device,
    )

    tril_i, tril_j = torch.tril_indices(p, p, device=x.device)
    L_tilde[..., tril_i, tril_j] = x

    # Recover the Cholesky factor by exponentiating the diagonal.
    L = L_tilde.clone()
    diag_idx = torch.arange(p, device=x.device)
    L[..., diag_idx, diag_idx] = torch.exp(L[..., diag_idx, diag_idx])

    # Reconstruct the SPD matrix.
    S = L @ L.transpose(-1, -2)

    # ---- return ----
    if bundle:
        return Bundle_spd(S, atol=atol)
    else:
        return S
    

# ============================================================
# Utility: small MLP
# ------------------------------------------------------------
# This section implements a lightweight multi-layer perceptron
# (MLP) used as a generic function approximator in flow-based
# models, in particular for producing transformation parameters
# such as scale and shift in affine coupling layers.
#
# The architecture is configurable in terms of input/output
# dimensions, hidden width, number of hidden layers, activation
# function, and optional dropout. The final layer is linear,
# without activation, so that the network can represent
# unrestricted real-valued outputs.
#
# This module operates purely in Euclidean coordinates. For SPD
# data, the manifold-valued inputs are first mapped to Euclidean
# space outside this module.
# ============================================================


class MLP(nn.Module):
    r"""
    Multi-layer perceptron (MLP) module.

    This module implements a fully connected feedforward neural network
    with configurable depth, hidden width, activation function, and dropout.
    It is primarily used as a function approximator within coupling layers
    to model transformation parameters.

    Parameters
    ----------
    in_dim : int
        Input dimension.

    out_dim : int
        Output dimension.

    hidden_dim : int, default=128
        Width of each hidden layer.

    num_hidden_layers : int, default=2
        Number of hidden layers.

    activation : nn.Module or None, default=None
        Activation function applied after each hidden linear layer.
        If None, ReLU is used.

    dropout : float, default=0.0
        Dropout probability applied after each activation.
        If 0.0, dropout is not used.

    Attributes
    ----------
    net : nn.Sequential
        Sequential container of linear layers, activation functions,
        and optional dropout layers.

    Notes
    -----
    The network has the following structure:

        Linear(in_dim → hidden_dim)
        → activation → (dropout)
        → ...
        → Linear(hidden_dim → hidden_dim)
        → activation → (dropout)
        → Linear(hidden_dim → out_dim)

    The final layer does not include an activation function.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., in_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., out_dim)``.
        """
        return self.net(x)


# ============================================================
# Affine Coupling Layer (conditional)
# ------------------------------------------------------------
# This section implements one conditional affine coupling layer,
# which serves as the fundamental invertible transformation block
# in the conditional normalizing flow model.
#
# Given an input variable x and a conditioning variable z, the
# layer keeps one subset of coordinates of x fixed according to
# a binary mask, and applies an affine transformation to the
# remaining coordinates. The scale and shift parameters are
# generated by a neural network depending on the masked input
# and the condition.
#
# The construction is designed so that:
#   - the forward map is explicit,
#   - the inverse map is available in closed form,
#   - the log-determinant of the Jacobian can be computed
#     efficiently.
#
# These properties make the layer suitable for likelihood-based
# training and conditional sample generation in RealNVP-style
# flow models.
# ============================================================


class ConditionalAffineCoupling(nn.Module):
    r"""
    Conditional affine coupling layer.

    This module implements one conditional affine coupling transformation
    used in conditional normalizing flows. Given an input tensor ``x`` and
    conditioning tensor ``z``, the layer keeps one subset of coordinates of
    ``x`` fixed according to a binary mask, and applies an affine
    transformation to the remaining coordinates. The scale and shift
    parameters are produced by a neural network depending on the masked input
    and the conditioning variable.

    More precisely, let ``m`` denote the binary mask. The transformation has
    the form

    .. math::
        x_{\mathrm{masked}} = m \odot x,

    .. math::
        s, t = g(x_{\mathrm{masked}}, z),

    .. math::
        y = x_{\mathrm{masked}}
            + (1-m) \odot \bigl(x \odot \exp(s) + t\bigr),

    where ``g`` is an MLP. The inverse transformation is available in closed
    form, and the log-determinant of the Jacobian is easy to compute.

    Parameters
    ----------
    x_dim : int
        Dimension of the input variable ``x``.

    z_dim : int
        Dimension of the conditioning variable ``z``.

    mask : torch.Tensor
        One-dimensional binary mask of shape ``(x_dim,)`` indicating which
        coordinates are kept fixed and which are transformed.

    hidden_dim : int, default=128
        Width of each hidden layer in the internal MLP.

    num_hidden_layers : int, default=2
        Number of hidden layers in the internal MLP.

    scale_limit : float, default=2.0
        Multiplicative bound applied to the output of ``tanh`` when producing
        the scale variable ``s``. This stabilizes the exponential scaling term.

    dropout : float, default=0.0
        Dropout probability used in the internal MLP.

    Attributes
    ----------
    x_dim : int
        Dimension of the input variable.

    z_dim : int
        Dimension of the conditioning variable.

    scale_limit : float
        Bound controlling the range of the scaling output.

    mask : torch.Tensor
        Binary mask stored as a registered buffer.

    net : MLP
        Neural network producing the scale and shift parameters.

    Notes
    -----
    The mask is stored as a buffer rather than a trainable parameter. This
    ensures that it moves together with the module across devices and is saved
    in the model state, but is not updated during optimization.

    The output of the internal MLP has dimension ``2 * x_dim`` and is split
    into two vectors:

    - ``s``: scale parameters;
    - ``t``: shift parameters.

    Only the coordinates corresponding to ``1 - mask`` are active in the
    transformation; masked coordinates remain unchanged.
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute scale and shift parameters from the masked input and condition.

        Parameters
        ----------
        x_masked : torch.Tensor
            Masked input tensor of shape ``(..., x_dim)``.

        z : torch.Tensor
            Conditioning tensor of shape ``(..., z_dim)``.

        Returns
        -------
        s, t : tuple[torch.Tensor, torch.Tensor]
            Scale and shift tensors, each of shape ``(..., x_dim)``.

        Notes
        -----
        The internal MLP takes as input the concatenation of ``x_masked`` and
        ``z`` along the last dimension, producing an output of shape
        ``(..., 2 * x_dim)``. This output is split into ``s`` and ``t``.

        The scale output is passed through ``tanh`` and multiplied by
        ``scale_limit`` to stabilize the exponential term in the affine
        transformation. Both ``s`` and ``t`` are masked so that only the
        unmasked coordinates are active.
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the forward affine coupling transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., x_dim)``.

        z : torch.Tensor
            Conditioning tensor of shape ``(..., z_dim)``.

        Returns
        -------
        y : torch.Tensor
            Transformed tensor of shape ``(..., x_dim)``.

        logdet : torch.Tensor
            Log-determinant of the Jacobian of the transformation, of shape
            equal to the batch shape ``(...)``.

        Notes
        -----
        The masked coordinates remain unchanged. The unmasked coordinates are
        transformed according to

        .. math::
            y = x_{\mathrm{masked}}
                + (1-m) \odot \bigl(x \odot \exp(s) + t\bigr).

        Because the Jacobian is triangular, its log-determinant is simply the
        sum of the scale components:

        .. math::
            \log |\det J| = \sum_i s_i.
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the inverse affine coupling transformation.

        Parameters
        ----------
        y : torch.Tensor
            Transformed tensor of shape ``(..., x_dim)``.

        z : torch.Tensor
            Conditioning tensor of shape ``(..., z_dim)``.

        Returns
        -------
        x : torch.Tensor
            Recovered input tensor of shape ``(..., x_dim)``.

        logdet_inv : torch.Tensor
            Log-determinant of the Jacobian of the inverse transformation, of
            shape equal to the batch shape ``(...)``.

        Notes
        -----
        Since the masked coordinates are unchanged by the forward map, the
        inverse transformation can be computed explicitly:

        .. math::
            x = y_{\mathrm{masked}}
                + (1-m) \odot (y - t) \odot \exp(-s).

        The log-determinant of the inverse Jacobian is the negative of the
        forward log-determinant:

        .. math::
            \log |\det J^{-1}| = -\sum_i s_i.
        """
        y_masked = self.mask * y
        s, t = self._st(y_masked, z)

        x = y_masked + (1.0 - self.mask) * (y - t) * torch.exp(-s)
        logdet_inv = -torch.sum(s, dim=-1)
        return x, logdet_inv
    

# ============================================================
# Conditional RealNVP with exactly 2 coupling layers
# ------------------------------------------------------------
# This section implements a conditional RealNVP model composed
# of exactly two conditional affine coupling layers with
# complementary masks.
#
# The model defines an invertible transformation between a
# simple latent Gaussian variable u and the target variable x,
# conditional on an observed variable z. It supports:
#   - forward transformation   : u -> x
#   - inverse transformation   : x -> u
#   - conditional log-density  : log p(x | z)
#   - conditional sampling     : x ~ p(. | z)
#
# The two-layer construction is the simplest nontrivial RealNVP
# architecture: the first layer updates one subset of coordinates,
# and the second layer updates the complementary subset, so that
# all coordinates are transformed across the full model.
# ============================================================


class ConditionalRealNVP(nn.Module):
    r"""
    Conditional RealNVP model with exactly two affine coupling layers.

    This module implements a conditional normalizing flow that maps a
    standard Gaussian latent variable ``u`` to an observed variable ``x``,
    conditional on an external variable ``z``. The transformation is built
    from two conditional affine coupling layers with complementary masks.

    More precisely, the model defines an invertible mapping

    .. math::
        x = f_\theta(u; z),

    where

    .. math::
        u \sim \mathcal{N}(0, I),

    and the conditional density of ``x`` given ``z`` is computed via the
    change-of-variables formula.

    Parameters
    ----------
    x_dim : int
        Dimension of the target variable ``x``.

    z_dim : int
        Dimension of the conditioning variable ``z``.

    hidden_dim : int, default=128
        Width of each hidden layer in the internal MLPs used by the coupling
        layers.

    num_hidden_layers : int, default=2
        Number of hidden layers in the internal MLPs.

    scale_limit : float, default=2.0
        Multiplicative bound applied to the ``tanh`` output in each coupling
        layer when constructing scale parameters.

    dropout : float, default=0.0
        Dropout probability used in the internal MLPs.

    Attributes
    ----------
    x_dim : int
        Dimension of the target variable.

    z_dim : int
        Dimension of the conditioning variable.

    layers : nn.ModuleList
        Ordered list of the two conditional affine coupling layers.

    Notes
    -----
    The model uses two alternating masks:

    - the first coupling layer updates one subset of coordinates;
    - the second coupling layer updates the complementary subset.

    This ensures that, across the two layers, all coordinates of the input
    are transformed.

    The latent base distribution is standard multivariate Gaussian, and the
    model supports both density evaluation and conditional sampling.
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

        Parameters
        ----------
        x_dim : int
            Dimension of the target variable ``x``.

        z_dim : int
            Dimension of the conditioning variable ``z``.

        hidden_dim : int, default=128
            Width of each hidden layer in the internal MLPs.

        num_hidden_layers : int, default=2
            Number of hidden layers in the internal MLPs.

        scale_limit : float, default=2.0
            Multiplicative bound for the scale outputs.

        dropout : float, default=0.0
            Dropout probability used in the internal MLPs.

        Raises
        ------
        ValueError
            If ``x_dim < 2``.
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
        start_with_one: bool = False,
    ) -> torch.Tensor:
        r"""
        Construct an alternating binary mask.

        Parameters
        ----------
        dim : int
            Length of the mask.

        start_with_one : bool, default=True
            If True, the mask starts with 1 at index 0 and alternates as
            ``[1, 0, 1, 0, ...]``.
            If False, the mask starts with 0 at index 0 and alternates as
            ``[0, 1, 0, 1, ...]``.

        Returns
        -------
        torch.Tensor
            One-dimensional mask tensor of shape ``(dim,)`` with dtype
            ``torch.float64``.
        """
        mask = torch.zeros(dim, dtype=torch.float64)
        if start_with_one:
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        return mask

    def _standard_normal_logprob(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the log-density of a standard multivariate normal distribution.

        Parameters
        ----------
        u : torch.Tensor
            Tensor of latent variables of shape ``(..., x_dim)``.

        Returns
        -------
        torch.Tensor
            Log-density values of shape equal to the batch shape ``(...)``.

        Notes
        -----
        This computes

        .. math::
            \log p(u)
            =
            -\frac{1}{2}
            \left(
                x_{\mathrm{dim}} \log(2\pi)
                + \|u\|_2^2
            \right),
            \qquad
            u \sim \mathcal{N}(0, I).
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
        Apply the inverse conditional flow transformation.

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
            Total log-determinant of the inverse transformation, of shape
            ``(batch_size,)``.

        Notes
        -----
        The inverse map is computed by applying the inverse of each coupling
        layer in reverse order:

        .. math::
            u = f_\theta^{-1}(x; z).

        The total inverse log-determinant is the sum of the inverse
        log-determinants from all coupling layers.
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
        Apply the forward conditional flow transformation.

        Parameters
        ----------
        u : torch.Tensor
            Latent tensor of shape ``(batch_size, x_dim)``.

        z : torch.Tensor
            Conditioning tensor of shape ``(batch_size, z_dim)``.

        Returns
        -------
        x : torch.Tensor
            Transformed tensor of shape ``(batch_size, x_dim)``.

        total_logdet : torch.Tensor
            Total log-determinant of the forward transformation, of shape
            ``(batch_size,)``.

        Notes
        -----
        The forward map is computed by applying the coupling layers in order:

        .. math::
            x = f_\theta(u; z).

        The total log-determinant is obtained by summing the forward
        log-determinants of all coupling layers.
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

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the conditional log-density ``log p(x | z)``.

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

        Notes
        -----
        The conditional density is evaluated using the change-of-variables
        formula:

        .. math::
            \log p(x \mid z)
            =
            \log p(u)
            +
            \log \left| \det \frac{\partial u}{\partial x} \right|,

        where

        .. math::
            u = f_\theta^{-1}(x; z).
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

            Supported shapes are:

            - ``(z_dim,)`` for a single conditioning value;
            - ``(batch_size, z_dim)`` for a batch of conditioning values.

        n_samples : int or None, default=None
            Number of samples to generate when ``z`` is one-dimensional.
            If ``z`` is already batched, this must be None.

        generator : torch.Generator or None, default=None
            Random number generator used for sampling the latent Gaussian noise.


        Returns
        -------
        torch.Tensor
            Generated tensor of shape:

            - ``(n_samples, x_dim)`` if ``z`` has shape ``(z_dim,)``;
            - ``(batch_size, x_dim)`` if ``z`` has shape ``(batch_size, z_dim)``.

        Raises
        ------
        ValueError
            If ``z`` has invalid dimension, or if ``n_samples`` is provided
            when ``z`` is already batched.

        Notes
        -----
        Sampling is performed by first drawing

        .. math::
            u \sim \mathcal{N}(0, I),

        and then applying the forward conditional flow transformation

        .. math::
            x = f_\theta(u; z).

        This method operates entirely in PyTorch and returns samples on the
        same device and with the same dtype as the input tensor ``z``.
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
    

# ============================================================
# Training configuration for conditional flow models
# ------------------------------------------------------------
# This section defines a lightweight configuration container
# for training conditional normalizing flow models.
#
# The configuration groups together all hyperparameters related
# to optimization, including training epochs, batch size,
# learning rate, weight decay, gradient clipping, and learning
# rate scheduling. It is implemented as a dataclass for clarity,
# immutability-like behavior, and ease of passing between
# training routines.
#
# This configuration is typically consumed by training functions
# such as `train_conditional_realnvp(...)` or higher-level
# wrappers that fit conditional generators.
# ============================================================


@dataclass
class FlowTrainConfig:
    r"""
    Configuration for training a conditional normalizing flow.

    This dataclass stores all hyperparameters required during the
    optimization of a conditional flow model. It controls training
    duration, mini-batch size, optimizer settings, gradient
    stabilization, and learning rate decay.

    Parameters
    ----------
    epochs : int, default=200
        Number of full passes over the training dataset.

    batch_size : int, default=128
        Number of samples per mini-batch during stochastic training.

    lr : float, default=1e-3
        Initial learning rate used by the optimizer (typically Adam).

    weight_decay : float, default=1e-5
        L2 regularization coefficient applied to model parameters.

    grad_clip_norm : float or None, default=5.0
        Maximum allowed norm of gradients. If not None, gradients are
        clipped to this value to improve training stability.
        If None, gradient clipping is disabled.

    scheduler_gamma : float, default=0.98
        Multiplicative factor for exponential learning rate decay:

        .. math::
            \text{lr}_{t+1} = \gamma \cdot \text{lr}_t.

    verbose : bool, default=True
        Whether to print training progress (e.g., loss values per epoch).

    Attributes
    ----------
    epochs : int
        Total number of training epochs.

    batch_size : int
        Mini-batch size.

    lr : float
        Initial learning rate.

    weight_decay : float
        Weight decay coefficient.

    grad_clip_norm : float or None
        Gradient clipping threshold.

    scheduler_gamma : float
        Learning rate decay factor.

    verbose : bool
        Flag controlling verbosity.

    Notes
    -----
    This configuration object is designed to be passed as a single
    argument to training routines, improving code readability and
    reducing the need for long argument lists.
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
# Training function
# ------------------------------------------------------------
# This section implements the optimization routines for fitting
# conditional RealNVP models by maximum likelihood.
#
# The main training function handles mini-batch stochastic
# optimization, including data conversion, device placement,
# loss evaluation, backpropagation, gradient clipping, learning
# rate scheduling, and optional validation tracking.
#
# A higher-level fitting wrapper is also provided, which
# constructs a conditional flow model from raw training data by
# inferring the input dimensions automatically and then invoking
# the training routine.
#
# These functions form the main entry point for estimating
# conditional flow-based generators from observed training
# samples.
# ============================================================


# def train_conditional_realnvp(
#     model: ConditionalRealNVP,
#     x_train: torch.Tensor,
#     z_train: torch.Tensor,
#     config: Optional[FlowTrainConfig] = None,
#     x_val: Optional[torch.Tensor] = None,
#     z_val: Optional[torch.Tensor] = None,
# ) -> Dict[str, List[float]]:
#     r"""
#     Train a conditional RealNVP model via maximum likelihood estimation.

#     This function optimizes the model parameters to approximate the
#     conditional density p(x | z) by minimizing the negative log-likelihood (NLL)
#     on the training data. Optionally, validation data can be provided for
#     monitoring generalization performance and enabling early stopping.

#     Parameters
#     ----------
#     model : ConditionalRealNVP
#         The conditional normalizing flow model to be trained.

#     x_train : torch.Tensor, shape (n_samples, x_dim)
#         Training samples of the target variable.

#     z_train : torch.Tensor, shape (n_samples, z_dim)
#         Conditioning variables corresponding to ``x_train``.

#     config : FlowTrainConfig, default=FlowTrainConfig()
#         Training configuration containing optimization hyperparameters such as:
#             - number of epochs
#             - batch size
#             - learning rate
#             - weight decay
#             - gradient clipping
#             - learning rate scheduler
#             - verbosity

#     x_val : torch.Tensor, shape (n_val, x_dim), optional
#         Validation samples of the target variable.

#     z_val : torch.Tensor, shape (n_val, z_dim), optional
#         Validation conditioning variables corresponding to ``x_val``.

#     Returns
#     -------
#     history : Dict[str, List[float]]
#         Dictionary containing training history:
#             - "train_nll": list of training negative log-likelihood per epoch
#             - "val_nll"  : list of validation negative log-likelihood per epoch
#               (empty if validation data is not provided)

#     Raises
#     ------
#     TypeError
#         If input data are not torch tensors.

#     ValueError
#         If input tensors have invalid dimensions or incompatible shapes.

#     Notes
#     -----
#     - The model is trained using mini-batch stochastic gradient descent.
#     - The loss function is the negative log-likelihood:
#         NLL = -E[log p_theta(x | z)].
#     - If validation data is provided, early stopping is applied based on
#       validation NLL.
#     - The model parameters are automatically moved to the same device and dtype
#       as ``x_train``.
#     - At the end of training, the model is restored to the best-performing
#       parameters (according to validation NLL).
#     """

#     if config is None:
#         config = FlowTrainConfig()

#     if not isinstance(x_train, torch.Tensor) or not isinstance(z_train, torch.Tensor):
#         raise TypeError("x_train and z_train must both be torch.Tensor.")

#     if x_val is not None and not isinstance(x_val, torch.Tensor):
#         raise TypeError("x_val must be torch.Tensor or None.")
#     if z_val is not None and not isinstance(z_val, torch.Tensor):
#         raise TypeError("z_val must be torch.Tensor or None.")

#     if x_train.ndim != 2 or z_train.ndim != 2:
#         raise ValueError("x_train and z_train must both be 2D.")

#     if x_train.shape[0] != z_train.shape[0]:
#         raise ValueError("x_train and z_train must have the same number of samples.")

#     if x_train.shape[1] != model.x_dim:
#         raise ValueError(f"x_train second dimension must be {model.x_dim}.")

#     if z_train.shape[1] != model.z_dim:
#         raise ValueError(f"z_train second dimension must be {model.z_dim}.")

#     if (x_val is None) != (z_val is None):
#         raise ValueError("x_val and z_val must either both be provided or both be None.")

#     model.to(device=x_train.device, dtype=x_train.dtype)

#     # train_loader = DataLoader(
#     #     TensorDataset(x_train, z_train),
#     #     batch_size=config.batch_size,
#     #     shuffle=True,
#     #     drop_last=False,
#     # )

#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=config.lr,
#         weight_decay=config.weight_decay,
#     )

#     scheduler = optim.lr_scheduler.ExponentialLR(
#         optimizer,
#         gamma=config.scheduler_gamma,
#     )

#     history: Dict[str, List[float]] = {
#         "train_nll": [],
#         "val_nll": [],
#     }

#     # ============================
#     # Early stopping parameters
#     # ============================
#     patience = config.patience
#     best_val = float("inf")
#     best_state = None
#     counter = 0

#     n_train = x_train.shape[0]

#     for epoch in range(config.epochs):
#         model.train()

#         perm = torch.randperm(n_train, device=x_train.device)

#         running_loss = 0.0
#         n_seen = 0

#         for start in range(0, n_train, config.batch_size):
#             end = min(start + config.batch_size, n_train)
#             idx = perm[start:end]
            
#             xb = x_train[idx]
#             zb = z_train[idx]

#             optimizer.zero_grad()

#             log_prob = model.log_prob(xb, zb)
#             loss = -log_prob.mean()

#             loss.backward()

#             if config.grad_clip_norm is not None:
#                 nn.utils.clip_grad_norm_(
#                     model.parameters(),
#                     max_norm=config.grad_clip_norm,
#                 )

#             optimizer.step()

#             bs = xb.shape[0]
#             running_loss += float(loss.item()) * bs
#             n_seen += bs

#         scheduler.step()

#         train_nll = running_loss / max(n_seen, 1)
#         history["train_nll"].append(train_nll)

#         # ============================
#         # Validation
#         # ============================
#         if x_val is not None and z_val is not None:
#             model.eval()
#             with torch.no_grad():
#                 val_nll = float(
#                     (-model.log_prob(x_val, z_val).mean()).item()
#                 )
#             history["val_nll"].append(val_nll)

#             # Early stopping
#             if val_nll < best_val:
#                 best_val = val_nll
#                 best_state = {
#                     k: v.detach().cpu().clone()
#                     for k, v in model.state_dict().items()
#                 }
#                 counter = 0
#             else:
#                 counter += 1

#             if counter >= patience:
#                 if config.verbose:
#                     print(f"Early stopping at epoch {epoch+1}")
#                 break

#         else:
#             val_nll = None

#         if config.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
#             if val_nll is None:
#                 print(f"[Epoch {epoch + 1:4d}] train_nll = {train_nll:.6f}")
#             else:
#                 print(
#                     f"[Epoch {epoch + 1:4d}] "
#                     f"train_nll = {train_nll:.6f}, val_nll = {val_nll:.6f}"
#                 )

#     # Restore best model parameters
#     if best_state is not None:
#         model.load_state_dict(best_state)

#     return history


def train_conditional_realnvp(
    model: ConditionalRealNVP,
    x_train: torch.Tensor,
    z_train: torch.Tensor,
    config: Optional[FlowTrainConfig] = None,
    x_val: Optional[torch.Tensor] = None,
    z_val: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    r"""
    Train a conditional RealNVP model via maximum likelihood estimation.

    This function optimizes the model parameters to approximate the
    conditional density p(x | z) by minimizing the negative log-likelihood (NLL)
    on the training data. Optionally, validation data can be provided for
    monitoring generalization performance and enabling early stopping.

    Parameters
    ----------
    model : ConditionalRealNVP
        The conditional normalizing flow model to be trained.

    x_train : torch.Tensor, shape (n_samples, x_dim)
        Training samples of the target variable.

    z_train : torch.Tensor, shape (n_samples, z_dim)
        Conditioning variables corresponding to ``x_train``.

    config : FlowTrainConfig, default=FlowTrainConfig()
        Training configuration containing optimization hyperparameters such as:
            - number of epochs
            - batch size
            - learning rate
            - weight decay
            - gradient clipping
            - learning rate scheduler
            - verbosity

    x_val : torch.Tensor, shape (n_val, x_dim), optional
        Validation samples of the target variable.

    z_val : torch.Tensor, shape (n_val, z_dim), optional
        Validation conditioning variables corresponding to ``x_val``.

    Returns
    -------
    history : Dict[str, List[float]]
        Dictionary containing training history:
            - "train_nll": list of training negative log-likelihood per epoch
            - "val_nll"  : list of validation negative log-likelihood per epoch
              (empty if validation data is not provided)

    Raises
    ------
    TypeError
        If input data are not torch tensors.

    ValueError
        If input tensors have invalid dimensions or incompatible shapes.

    Notes
    -----
    - The model is trained using mini-batch stochastic gradient descent.
    - The loss function is the negative log-likelihood:
        NLL = -E[log p_theta(x | z)].
    - If validation data is provided, early stopping is applied based on
      validation NLL.
    - The model parameters are automatically moved to the same device and dtype
      as ``x_train``.
    - At the end of training, the model is restored to the best-performing
      parameters (according to validation NLL).
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

    # train_loader = DataLoader(
    #     TensorDataset(x_train, z_train),
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     drop_last=False,
    # )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.scheduler_gamma,
    )

    history: Dict[str, List[float]] = {
        "train_nll": [],
        "val_nll": [],
    }

    # ============================
    # Early stopping parameters
    # ============================
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

        # ============================
        # Validation
        # ============================
        if x_val is not None and z_val is not None:
            model.eval()
            with torch.no_grad():
                val_nll = float(
                    (-model.log_prob(x_val, z_val).mean()).item()
                )
            history["val_nll"].append(val_nll)

            # Early stopping
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
                    print(f"Early stopping at epoch {epoch+1}")
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

    # Restore best model parameters
    if best_state is not None:
        model.load_state_dict(best_state)
        if config.verbose:
            print(
                f"Restored best model from epoch {best_epoch}, "
                f"best_val = {best_val:.6f}"
            )

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
) -> tuple[ConditionalRealNVP, Dict[str, List[float]]]:
    r"""
    Fit a conditional RealNVP model from training data.

    This is a high-level convenience wrapper that:
    1. infers input dimensions from the data,
    2. constructs a ``ConditionalRealNVP`` model,
    3. trains the model using maximum likelihood.

    Parameters
    ----------
    x_train : torch.Tensor
        Training samples of shape ``(n_samples, x_dim)``.

    z_train : torch.Tensor
        Conditioning variables of shape ``(n_samples, z_dim)``.

    config : FlowTrainConfig, default=FlowTrainConfig()
        Training configuration containing hyperparameters such as
        number of epochs, batch size, learning rate, etc.

    hidden_dim : int, default=128
        Width of hidden layers in the MLPs inside coupling layers.

    num_hidden_layers : int, default=2
        Number of hidden layers in each MLP.

    scale_limit : float, default=2.0
        Bound applied to scale outputs in coupling layers to stabilize
        exponential transformations.

    dropout : float, default=0.0
        Dropout probability used in MLPs.

    x_val : torch.Tensor, optional
        Validation samples of shape ``(n_val, x_dim)``.

    z_val : torch.Tensor, optional
        Validation conditioning variables of shape ``(n_val, z_dim)``.

    Returns
    -------
    model : ConditionalRealNVP
        Trained conditional flow model.

    history : dict[str, list[float]]
        Training history containing:

        - ``"train_nll"`` : training negative log-likelihood per epoch
        - ``"val_nll"``   : validation negative log-likelihood (if provided)

    Raises
    ------
    ValueError
        If input tensors are not two-dimensional.

    Notes
    -----
    The model defines a conditional invertible mapping

    .. math::
        x = f_\theta(u; z), \quad u \sim \mathcal{N}(0, I),

    and is trained by maximizing the conditional log-likelihood

    .. math::
        \log p_\theta(x \mid z).

    This function is the recommended entry point when training from raw
    data, as it automatically infers dimensions and handles model creation.
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
# This class implements fitted conditional generators for
# sampling from estimated conditional laws X | Z and Y | Z
# in the SPD setting.
#
# The fitted models are conditional RealNVP flows trained in
# Euclidean coordinates obtained via the log-Cholesky transform.
# During training, SPD inputs are mapped to Euclidean space;
# during sampling, generated Euclidean samples are mapped back
# to SPD matrices and returned as SPDDataBundle objects.
#
# The interface mirrors that of OracleGenerators so that the
# fitted generator can be used directly in conditional sampling
# pipelines and downstream testing routines.
# ============================================================


class FittedConditionalGenerators:
    r"""
    Fitted conditional generators for ``X | Z`` and ``Y | Z`` in the SPD case.

    This class wraps two trained conditional RealNVP models:
    one approximating the conditional law of ``X | Z``, and
    one approximating the conditional law of ``Y | Z``.

    All training and sampling are performed in Euclidean
    coordinates induced by the log-Cholesky transform, while
    external inputs and outputs are represented as SPDDataBundle
    objects.

    Notes
    -----
    - GPU-only (torch.Tensor on CUDA)
    - SPD-only
    - Outputs are returned as SPDDataBundle so that geometric
      quantities are cached for downstream use
    """

    def __init__(
        self,
        x_model: ConditionalRealNVP,
        y_model: ConditionalRealNVP,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        r"""
        Initialize a fitted conditional generator.

        Parameters
        ----------
        x_model : ConditionalRealNVP
            Trained conditional flow model for approximating
            the conditional law ``X | Z``.

        y_model : ConditionalRealNVP
            Trained conditional flow model for approximating
            the conditional law ``Y | Z``.

        space_type : {"euclidean", "spd"}
            Type of the sample space.

        GPU : bool, default=False
            Whether sampling-time transformations should use GPU.

        dtype : torch.dtype, default=torch.float64
            Torch dtype used for tensor conversion during
            transformation and sampling.

        Raises
        ------
        ValueError
            If ``space_type`` is invalid.
        """

        self.x_model = x_model
        self.y_model = y_model
        self.dtype = dtype

    @classmethod
    def fit(
        cls,
        Bundle_X: SPDDataBundle,
        Bundle_Y: SPDDataBundle,
        Bundle_Z: SPDDataBundle,
        config: Optional[FlowTrainConfig] = None,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale_limit: float = 2.0,
        dropout: float = 0.0,
        Bundle_X_val: Optional[SPDDataBundle] = None,
        Bundle_Y_val: Optional[SPDDataBundle] = None,
        Bundle_Z_val: Optional[SPDDataBundle] = None,
    ) -> tuple["FittedConditionalGenerators", Dict[str, Dict[str, List[float]]]]:
        r"""
        Fit conditional flow models for ``X | Z`` and ``Y | Z`` in the SPD case.

        Parameters
        ----------
        Bundle_X : SPDDataBundle
            Observed SPD samples of ``X``.

            Shape of ``Bundle_X.matrix``:
            - ``(n, p_x, p_x)``

        Bundle_Y : SPDDataBundle
            Observed SPD samples of ``Y``.

            Shape of ``Bundle_Y.matrix``:
            - ``(n, p_y, p_y)``

        Bundle_Z : SPDDataBundle
            Observed SPD conditioning variables.

            Shape of ``Bundle_Z.matrix``:
            - ``(n, p_z, p_z)``

        config : FlowTrainConfig or None, default=None
            Training configuration for both conditional flow models.

        hidden_dim : int, default=128
            Width of hidden layers in coupling networks.

        num_hidden_layers : int, default=2
            Number of hidden layers in each coupling network.

        scale_limit : float, default=2.0
            Upper bound applied to scale outputs in coupling layers.

        dropout : float, default=0.0
            Dropout probability in coupling networks.

        Bundle_X_val : SPDDataBundle or None, default=None
            Optional validation set for ``X``.

        Bundle_Y_val : SPDDataBundle or None, default=None
            Optional validation set for ``Y``.

        Bundle_Z_val : SPDDataBundle or None, default=None
            Optional validation set for ``Z``.

        Returns
        -------
        fitted : FittedConditionalGenerators
            Fitted generator object wrapping the trained models.

        history : dict[str, dict[str, list[float]]]
            Training history dictionary with keys:
            - ``"x_history"``
            - ``"y_history"``
        """

        if config is None:
            config = FlowTrainConfig()

        if not isinstance(Bundle_X, SPDDataBundle) or not isinstance(Bundle_Y, SPDDataBundle) or not isinstance(Bundle_Z, SPDDataBundle):
            raise TypeError("X, Y, Z must all be SPDDataBundle.")
        
        n = Bundle_X.matrix.shape[0]

        device = Bundle_Z.matrix.device
        dtype = Bundle_Z.matrix.dtype

        # ------------------------------------------------------------
        # SPD -> Euclidean
        # ------------------------------------------------------------
        X_trf: torch.Tensor = spd_to_euclidean(Bundle_X)
        Y_trf: torch.Tensor = spd_to_euclidean(Bundle_Y)
        Z_trf: torch.Tensor = spd_to_euclidean(Bundle_Z)

        X_val_trf = None
        Y_val_trf = None
        Z_val_trf = None

        if (Bundle_X_val is None) != (Bundle_Z_val is None):
            raise ValueError(
                "Bundle_X_val and Bundle_Z_val must either both be provided or both be None."
            )
        if (Bundle_Y_val is None) != (Bundle_Z_val is None):
            raise ValueError(
                "Bundle_Y_val and Bundle_Z_val must either both be provided or both be None."
            )

        if Bundle_X_val is not None:
            if not isinstance(Bundle_X_val, SPDDataBundle):
                raise TypeError("Bundle_X_val must be SPDDataBundle or None.")
            if not isinstance(Bundle_Z_val, SPDDataBundle):
                raise TypeError("Bundle_Z_val must be SPDDataBundle or None.")
            X_val_trf = spd_to_euclidean(Bundle_X_val)
            Z_val_trf = spd_to_euclidean(Bundle_Z_val)

        if Bundle_Y_val is not None:
            if not isinstance(Bundle_Y_val, SPDDataBundle):
                raise TypeError("Bundle_Y_val must be SPDDataBundle or None.")
            if not isinstance(Bundle_Z_val, SPDDataBundle):
                raise TypeError("Bundle_Z_val must be SPDDataBundle or None.")
            Y_val_trf = spd_to_euclidean(Bundle_Y_val)
            if Z_val_trf is None:
                Z_val_trf = spd_to_euclidean(Bundle_Z_val)
        
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
            dtype=dtype,
        )

        history = {
            "x_history": x_history,
            "y_history": y_history,
        }
        return fitted, history


    @torch.no_grad()
    def __call__(
        self,
        Bundle_Z: SPDDataBundle,
        M: int,
        generator: Optional[torch.Generator] = None,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Generate fitted conditional samples for a batch of SPD conditioning values.

        For each conditioning matrix ``Z_i``, this method generates ``M``
        independent samples from the fitted conditional laws
        ``X | Z_i`` and ``Y | Z_i``.

        Parameters
        ----------
        Bundle_Z : SPDDataBundle
            Batch of SPD conditioning variables.

            Shape of ``Bundle_Z.matrix``:
            - ``(n, p_z, p_z)``

        M : int
            Number of conditional replications generated for each observation.

        generator : torch.Generator or None, default=None
            Random number generator used for latent Gaussian sampling.

        bundle_chunk_size : int or None, default=None
            Optional chunk size passed to ``Bundle_spd(...)`` when converting
            generated SPD matrices back into bundle form.

        Returns
        -------
        Bundle_X_gen : SPDDataBundle
            Generated conditional samples for ``X``.

            Shape of ``Bundle_X_gen.matrix``:
            - ``(n, M, p_x, p_x)``

        Bundle_Y_gen : SPDDataBundle
            Generated conditional samples for ``Y``.

            Shape of ``Bundle_Y_gen.matrix``:
            - ``(n, M, p_y, p_y)``
        """
        if not isinstance(Bundle_Z, SPDDataBundle):
            raise TypeError("Bundle_Z must be SPDDataBundle.")

        Z_euc: torch.Tensor = spd_to_euclidean(Bundle_Z)   # (n, z_dim)
        n: int = Z_euc.shape[0]
        dtype = Z_euc.dtype
        device = Z_euc.device
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

            # → SPD（这里只做 matrix，不 bundle）
            X_chunk = euclidean_to_spd(X_chunk_euc, bundle=False)
            Y_chunk = euclidean_to_spd(Y_chunk_euc, bundle=False)

            # --------------------------------------
            # 第一次 chunk：确定 p 并分配内存
            # --------------------------------------
            if X_out is None:
                p_x = X_chunk.shape[-1]
                p_y = Y_chunk.shape[-1]

                X_out = torch.empty((n_total, p_x, p_x), device=device, dtype=dtype)
                Y_out = torch.empty((n_total, p_y, p_y), device=device, dtype=dtype)

            # --------------------------------------
            # 直接写入（无 list）
            # --------------------------------------
            X_out[start:end] = X_chunk
            Y_out[start:end] = Y_chunk

        # reshape
        X_mat = X_out.reshape(n, M, p_x, p_x)
        Y_mat = Y_out.reshape(n, M, p_y, p_y)

        # # 最后再 bundle（只做一次）
        # Bundle_X_gen = Bundle_spd(X_mat, atol=1e-12)
        # Bundle_Y_gen = Bundle_spd(Y_mat, atol=1e-12)

        return X_mat, Y_mat