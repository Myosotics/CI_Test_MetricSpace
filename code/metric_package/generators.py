import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Oracle conditional generators
# ------------------------------------------------------------
# This class implements oracle conditional generators for the
# simulation models considered in the paper. It provides both
# a single-sample interface and a batched interface for
# generating conditional samples given conditioning values.
#
# The implementation serves as a reference baseline and is
# compatible with the unified generator interface defined in
# this module, allowing seamless integration with resampling
# and testing routines.
# ============================================================


class OracleGenerators:
    r"""
    Oracle conditional generator for simulation models on metric spaces.

    This class implements both a single-sample interface and a batched
    interface for generating conditional samples ``(X, Y)`` given
    conditioning values ``Z``. The batched interface is compatible with
    routines such as ``generate_conditional_samples(...)`` that check for
    a ``generate_batch`` method.

    Parameters
    ----------
    sigma_perm : float, default=2.0
        Scale parameter for perturbations in the Euclidean and spherical
        cases.

    Notes
    -----
    The conditional laws are specified according to the metric-space type:

    - Euclidean: additive Gaussian perturbations around ``Z``;
    - Sphere: random tangent perturbations followed by projection back to
      the unit sphere;
    - SPD: Wishart-type perturbations obtained through Gaussian matrix
      factors and congruence transformation.
    """

    def __init__(self, sigma_perm: float = 2.0) -> None:
        self.sigma_perm = sigma_perm

    def __call__(
        self,
        Z: np.ndarray,
        space_type: str = "euclidean",
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Generate a single conditional sample pair ``(X, Y)`` given one value
        of ``Z``.

        Parameters
        ----------
        Z : np.ndarray
            Conditioning value.

            - Shape ``(d,)`` for Euclidean and spherical cases;
            - Shape ``(p, p)`` for SPD case.

        space_type : {"euclidean", "sphere", "spd"}, default="euclidean"
            Metric-space type.

        rng : np.random.Generator or None, default=None
            Random number generator. If ``None``, a default generator is used.

        Returns
        -------
        X, Y : tuple of np.ndarray
            A single conditional sample pair.
        """
        if rng is None:
            rng = np.random.default_rng()

        sigma = self.sigma_perm

        # ------------------------------------------------------------
        # Case 1: Euclidean space
        # ------------------------------------------------------------
        if space_type == "euclidean":
            d = Z.shape[0]

            U = rng.normal(size=d)
            V = rng.normal(size=d)

            X = Z + sigma * U
            Y = Z + sigma * V

            return X, Y

        # ------------------------------------------------------------
        # Case 2: Unit sphere
        # ------------------------------------------------------------
        elif space_type == "sphere":
            d = Z.shape[0]
            eps = 1e-12

            gx = rng.normal(size=d)
            U = gx - np.dot(gx, Z) * Z
            U = U / max(np.linalg.norm(U), eps)

            gy = rng.normal(size=d)
            V = gy - np.dot(gy, Z) * Z
            V = V / max(np.linalg.norm(V), eps)

            xi_X = sigma * rng.normal()
            xi_Y = sigma * rng.normal()

            X = Z + xi_X * U
            Y = Z + xi_Y * V

            X = X / np.linalg.norm(X)
            Y = Y / np.linalg.norm(Y)

            return X, Y

        # ------------------------------------------------------------
        # Case 3: SPD space
        # ------------------------------------------------------------
        elif space_type == "spd":
            p = Z.shape[0]
            nu = p + 6

            Z_half = np.linalg.cholesky(Z)

            A = rng.normal(size=(nu, p))
            B = rng.normal(size=(nu, p))

            Sx = (A.T @ A) / nu
            Sy = (B.T @ B) / nu

            X = Z_half @ Sx @ Z_half.T
            Y = Z_half @ Sy @ Z_half.T

            return X, Y

        else:
            raise ValueError(
                "space_type must be one of {'euclidean', 'sphere', 'spd'}."
            )

    def generate_batch(
        self,
        Z: np.ndarray,
        M: int,
        space_type: str = "euclidean",
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Generate batched conditional samples for a batch of conditioning values.

        For each conditioning value ``Z_i``, this method generates ``M``
        independent conditional sample pairs from the oracle conditional law.

        Parameters
        ----------
        Z : np.ndarray
            Batch of conditioning values.

            - Shape ``(n, d)`` for Euclidean and spherical cases;
            - Shape ``(n, p, p)`` for SPD case.

        M : int
            Number of conditional replications generated for each observation.

        space_type : {"euclidean", "sphere", "spd"}, default="euclidean"
            Metric-space type.

        rng : np.random.Generator or None, default=None
            Random number generator. If ``None``, a default generator is used.

        Returns
        -------
        X_all, Y_all : tuple of np.ndarray
            Batched conditional samples.

            - Euclidean / sphere: shape ``(n, M, d)``;
            - SPD: shape ``(n, M, p, p)``.

        Raises
        ------
        ValueError
            If ``space_type`` is invalid, or if the spherical ambient
            dimension is smaller than two.
        """
        if rng is None:
            rng = np.random.default_rng()

        sigma = self.sigma_perm

        # ------------------------------------------------------------
        # Case 1: Euclidean space
        # ------------------------------------------------------------
        if space_type == "euclidean":
            n, d = Z.shape

            U = rng.normal(size=(n, M, d))
            V = rng.normal(size=(n, M, d))

            Z_exp = Z[:, None, :]
            X_all = Z_exp + sigma * U
            Y_all = Z_exp + sigma * V

            return X_all, Y_all

        # ------------------------------------------------------------
        # Case 2: Unit sphere
        # ------------------------------------------------------------
        elif space_type == "sphere":
            n, d = Z.shape

            if d < 2:
                raise ValueError("For spherical data, size must be at least 2.")

            eps = 1e-12
            Z_exp = Z[:, None, :]  # shape (n, 1, d)

            # Generate tangent directions for X.
            GX = rng.normal(size=(n, M, d))
            U = GX - np.sum(GX * Z_exp, axis=2, keepdims=True) * Z_exp
            U_norm = np.linalg.norm(U, axis=2, keepdims=True)

            # Generate tangent directions for Y.
            GY = rng.normal(size=(n, M, d))
            V = GY - np.sum(GY * Z_exp, axis=2, keepdims=True) * Z_exp
            V_norm = np.linalg.norm(V, axis=2, keepdims=True)

            # Resample numerically degenerate projected vectors if needed.
            bad_u = U_norm[..., 0] < eps
            bad_v = V_norm[..., 0] < eps

            Z_rep = np.broadcast_to(Z_exp, (n, M, d))

            while np.any(bad_u):
                GX_new = rng.normal(size=(bad_u.sum(), d))
                Z_bad = Z_rep[bad_u]
                U_new = GX_new - np.sum(GX_new * Z_bad, axis=1, keepdims=True) * Z_bad
                U[bad_u] = U_new
                U_norm = np.linalg.norm(U, axis=2, keepdims=True)
                bad_u = U_norm[..., 0] < eps

            while np.any(bad_v):
                GY_new = rng.normal(size=(bad_v.sum(), d))
                Z_bad = Z_rep[bad_v]
                V_new = GY_new - np.sum(GY_new * Z_bad, axis=1, keepdims=True) * Z_bad
                V[bad_v] = V_new
                V_norm = np.linalg.norm(V, axis=2, keepdims=True)
                bad_v = V_norm[..., 0] < eps

            U /= U_norm
            V /= V_norm

            xi_X = sigma * rng.normal(size=(n, M, 1))
            xi_Y = sigma * rng.normal(size=(n, M, 1))

            X_all = Z_exp + xi_X * U
            Y_all = Z_exp + xi_Y * V

            X_all /= np.linalg.norm(X_all, axis=2, keepdims=True)
            Y_all /= np.linalg.norm(Y_all, axis=2, keepdims=True)

            return X_all, Y_all

        # ------------------------------------------------------------
        # Case 3: SPD space
        # ------------------------------------------------------------
        elif space_type == "spd":
            n, p, _ = Z.shape
            nu = p + 6

            Z_half = np.linalg.cholesky(Z)
            Z_half_t = np.transpose(Z_half, (0, 2, 1))

            A = rng.normal(size=(n, M, nu, p))
            B = rng.normal(size=(n, M, nu, p))

            Sx = np.matmul(np.transpose(A, (0, 1, 3, 2)), A) / nu
            Sy = np.matmul(np.transpose(B, (0, 1, 3, 2)), B) / nu

            ZL = Z_half[:, None, :, :]
            ZR = Z_half_t[:, None, :, :]

            X_all = np.matmul(np.matmul(ZL, Sx), ZR)
            Y_all = np.matmul(np.matmul(ZL, Sy), ZR)

            return X_all, Y_all

        else:
            raise ValueError(
                "space_type must be one of {'euclidean', 'sphere', 'spd'}."
            )
        

# ============================================================
# SPD <-> Euclidean transform
# ============================================================

def spd_to_euclidean(
    S: Union[np.ndarray, torch.Tensor],
    GPU: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Map SPD matrix/matrices to Euclidean coordinates via the log-Cholesky map.
    """
    if not GPU:
        if S.ndim == 2:
            p, p2 = S.shape
            if p != p2:
                raise ValueError(
                    f"Input SPD matrix must be square, but got shape {tuple(S.shape)}."
                )

            L = np.linalg.cholesky(S).copy()
            idx = np.diag_indices(p)
            L[idx] = np.log(L[idx])

            tril_i, tril_j = np.tril_indices(p)
            return L[tril_i, tril_j]

        if S.ndim == 3:
            _, p, p2 = S.shape
            if p != p2:
                raise ValueError(
                    f"Each SPD matrix must be square, but got shape {tuple(S.shape[1:])}."
                )

            L = np.linalg.cholesky(S).copy()
            diag_idx = np.arange(p)
            L[:, diag_idx, diag_idx] = np.log(L[:, diag_idx, diag_idx])

            tril_i, tril_j = np.tril_indices(p)
            return L[:, tril_i, tril_j]

        raise ValueError("Input must be either two- or three-dimensional.")

    if S.ndim == 2:
        p, p2 = S.shape
        if p != p2:
            raise ValueError(
                f"Input SPD matrix must be square, but got shape {tuple(S.shape)}."
            )

        L = torch.linalg.cholesky(S).clone()
        diag_idx = torch.arange(p, device=S.device)
        L[diag_idx, diag_idx] = torch.log(L[diag_idx, diag_idx])

        tril_i, tril_j = torch.tril_indices(p, p, device=S.device)
        return L[tril_i, tril_j]

    if S.ndim == 3:
        _, p, p2 = S.shape
        if p != p2:
            raise ValueError(
                f"Each SPD matrix must be square, but got shape {tuple(S.shape[1:])}."
            )

        L = torch.linalg.cholesky(S).clone()
        diag_idx = torch.arange(p, device=S.device)
        L[:, diag_idx, diag_idx] = torch.log(L[:, diag_idx, diag_idx])

        tril_i, tril_j = torch.tril_indices(p, p, device=S.device)
        return L[:, tril_i, tril_j]

    raise ValueError("Input must be either two- or three-dimensional.")


def infer_p_from_q(q: int) -> int:
    r"""
    Infer the SPD matrix size p from q = p(p+1)/2.
    """
    p = (math.isqrt(1 + 8 * q) - 1) // 2
    if p * (p + 1) // 2 != q:
        raise ValueError(
            f"Input length q={q} is invalid: it must satisfy q = p(p+1)/2 for some integer p."
        )
    return int(p)


def euclidean_to_spd(
    x: Union[np.ndarray, torch.Tensor],
    GPU: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Map Euclidean coordinates back to SPD matrix/matrices via the inverse
    log-Cholesky map.
    """
    if not GPU:
        if x.ndim == 1:
            q = x.shape[0]
            p = infer_p_from_q(q)

            L_tilde = np.zeros((p, p), dtype=x.dtype)
            tril_i, tril_j = np.tril_indices(p)
            L_tilde[tril_i, tril_j] = x

            diag_idx = np.diag_indices(p)
            L = L_tilde.copy()
            L[diag_idx] = np.exp(L[diag_idx])

            return L @ L.T

        if x.ndim == 2:
            M, q = x.shape
            p = infer_p_from_q(q)

            L_tilde = np.zeros((M, p, p), dtype=x.dtype)
            tril_i, tril_j = np.tril_indices(p)
            L_tilde[:, tril_i, tril_j] = x

            L = L_tilde.copy()
            diag_idx = np.arange(p)
            L[:, diag_idx, diag_idx] = np.exp(L[:, diag_idx, diag_idx])

            return np.matmul(L, np.transpose(L, (0, 2, 1)))

        raise ValueError("Input must be either one- or two-dimensional.")

    if x.ndim == 1:
        q = x.shape[0]
        p = infer_p_from_q(q)

        L_tilde = torch.zeros((p, p), dtype=x.dtype, device=x.device)
        tril_i, tril_j = torch.tril_indices(p, p, device=x.device)
        L_tilde[tril_i, tril_j] = x

        L = L_tilde.clone()
        diag_idx = torch.arange(p, device=x.device)
        L[diag_idx, diag_idx] = torch.exp(L[diag_idx, diag_idx])

        return L @ L.transpose(-1, -2)

    if x.ndim == 2:
        M, q = x.shape
        p = infer_p_from_q(q)

        L_tilde = torch.zeros((M, p, p), dtype=x.dtype, device=x.device)
        tril_i, tril_j = torch.tril_indices(p, p, device=x.device)
        L_tilde[:, tril_i, tril_j] = x

        L = L_tilde.clone()
        diag_idx = torch.arange(p, device=x.device)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])

        return L @ L.transpose(-1, -2)

    raise ValueError("Input must be either one- or two-dimensional.")


# ============================================================
# Utility: small MLP
# ------------------------------------------------------------
# This section implements a lightweight multi-layer perceptron
# (MLP), which is used as a generic function approximator in
# the model, in particular for producing the scale and shift
# parameters in affine coupling layers.
#
# The architecture is fully configurable in terms of input/output
# dimensions, hidden width, number of hidden layers, activation
# function, and optional dropout. The final layer is linear,
# without activation, so that the network can freely represent
# real-valued outputs.
#
# This module is intentionally simple and modular, and can be
# reused across different components that require a feedforward
# neural network.
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
        start_with_one: bool = True,
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
        seed: Optional[int] = None,
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

        seed : int or None, default=None
            Random seed used for sampling the latent Gaussian noise.

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

        and then applying the forward conditional flow transformation.
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

        if seed is not None:
            gen = torch.Generator(device=z.device)
            gen.manual_seed(seed)
            u = torch.randn(
                z.shape[0],
                self.x_dim,
                device=z.device,
                dtype=z.dtype,
                generator=gen,
            )
        else:
            u = torch.randn(
                z.shape[0],
                self.x_dim,
                device=z.device,
                dtype=z.dtype,
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
        Generate conditional samples and return them as a NumPy array.

        Parameters
        ----------
        z : np.ndarray or torch.Tensor
            Conditioning value(s), with supported shapes:

            - ``(z_dim,)`` for a single conditioning value;
            - ``(batch_size, z_dim)`` for a batch of conditioning values.

        n_samples : int or None, default=None
            Number of samples to generate when ``z`` is one-dimensional.
            If ``z`` is already batched, this must be None.

        seed : int or None, default=None
            Random seed used for sampling the latent Gaussian noise.

        Returns
        -------
        np.ndarray
            Generated samples as a NumPy array, with shape:

            - ``(n_samples, x_dim)`` if ``z`` is one-dimensional;
            - ``(batch_size, x_dim)`` if ``z`` is batched.

        Notes
        -----
        This method accepts either a NumPy array or a torch tensor as input,
        converts it to the same device and dtype as the model parameters,
        generates samples via ``sample_tensor(...)``, and returns the result
        as a NumPy array.
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


def train_conditional_realnvp(
    model: ConditionalRealNVP,
    x_train: torch.Tensor,
    z_train: torch.Tensor,
    config: FlowTrainConfig = FlowTrainConfig(),
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

    train_loader = DataLoader(
        TensorDataset(x_train, z_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.scheduler_gamma,
    )

    history = {
        "train_nll": [],
        "val_nll": [],
    }

    # ============================
    # Early stopping parameters
    # ============================
    patience = 20
    best_val = float("inf")
    best_state = None
    counter = 0

    for epoch in range(config.epochs):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for xb, zb in train_loader:
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
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                counter = 0
            else:
                counter += 1

            if counter >= patience:
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

    return history


# def train_conditional_realnvp(
#     model: ConditionalRealNVP,
#     x_train: torch.Tensor,
#     z_train: torch.Tensor,
#     config: FlowTrainConfig = FlowTrainConfig(),
#     x_val: Optional[torch.Tensor] = None,
#     z_val: Optional[torch.Tensor] = None,
# ) -> Dict[str, List[float]]:
#     r"""
#     Train a conditional RealNVP model via maximum likelihood.

#     This function fits the conditional flow model by maximizing the
#     conditional log-likelihood

#     .. math::
#         \max_\theta \; \mathbb{E}[\log p_\theta(x \mid z)],

#     using mini-batch stochastic optimization with Adam.

#     Parameters
#     ----------
#     model : ConditionalRealNVP
#         Conditional normalizing flow model to be trained.

#     x_train : torch.Tensor
#         Training samples of shape ``(n_samples, x_dim)``.

#     z_train : torch.Tensor
#         Conditioning variables of shape ``(n_samples, z_dim)``.

#     config : FlowTrainConfig, default=FlowTrainConfig()
#         Training configuration containing hyperparameters such as
#         learning rate, batch size, number of epochs, etc.

#     x_val : torch.Tensor, optional
#         Validation samples of shape ``(n_val, x_dim)``.

#     z_val : torch.Tensor, optional
#         Validation conditioning variables of shape ``(n_val, z_dim)``.

#     Returns
#     -------
#     history : dict[str, list[float]]
#         Dictionary containing training history:

#         - ``"train_nll"`` : list of training negative log-likelihood values
#         - ``"val_nll"``   : list of validation negative log-likelihood values
#           (empty if no validation data is provided)

#     Raises
#     ------
#     TypeError
#         If input data are not torch tensors.

#     ValueError
#         If input shapes are invalid or incompatible with the model.

#     Notes
#     -----
#     The training objective is the negative log-likelihood (NLL):

#     .. math::
#         \mathrm{NLL} = -\log p_\theta(x \mid z).

#     This function assumes that all input tensors have already been prepared
#     on the desired device and with the desired dtype. The model is moved to
#     match the device and dtype of ``x_train``.
#     """

#     # ------------------------------------------------------------
#     # Type validation
#     # ------------------------------------------------------------
#     if not isinstance(x_train, torch.Tensor) or not isinstance(z_train, torch.Tensor):
#         raise TypeError("x_train and z_train must both be torch.Tensor.")

#     if x_val is not None and not isinstance(x_val, torch.Tensor):
#         raise TypeError("x_val must be torch.Tensor or None.")
#     if z_val is not None and not isinstance(z_val, torch.Tensor):
#         raise TypeError("z_val must be torch.Tensor or None.")

#     # ------------------------------------------------------------
#     # Shape validation for training data
#     # ------------------------------------------------------------
#     if x_train.ndim != 2 or z_train.ndim != 2:
#         raise ValueError("x_train and z_train must both be 2D.")

#     if x_train.shape[0] != z_train.shape[0]:
#         raise ValueError("x_train and z_train must have the same number of samples.")

#     if x_train.shape[1] != model.x_dim:
#         raise ValueError(f"x_train second dimension must be {model.x_dim}.")

#     if z_train.shape[1] != model.z_dim:
#         raise ValueError(f"z_train second dimension must be {model.z_dim}.")

#     # ------------------------------------------------------------
#     # Validation data preparation and checking
#     # ------------------------------------------------------------
#     if (x_val is None) != (z_val is None):
#         raise ValueError("x_val and z_val must either both be provided or both be None.")

#     if x_val is not None and z_val is not None:
#         if x_val.ndim != 2 or z_val.ndim != 2:
#             raise ValueError("x_val and z_val must both be 2D.")

#         if x_val.shape[0] != z_val.shape[0]:
#             raise ValueError("x_val and z_val must have the same number of samples.")

#         if x_val.shape[1] != model.x_dim:
#             raise ValueError(f"x_val second dimension must be {model.x_dim}.")

#         if z_val.shape[1] != model.z_dim:
#             raise ValueError(f"z_val second dimension must be {model.z_dim}.")

#     # ------------------------------------------------------------
#     # Move model to match training data
#     # ------------------------------------------------------------
#     model.to(device=x_train.device, dtype=x_train.dtype)

#     # ------------------------------------------------------------
#     # Create DataLoader for mini-batch training
#     # ------------------------------------------------------------
#     train_loader: DataLoader = DataLoader(
#         TensorDataset(x_train, z_train),
#         batch_size=config.batch_size,
#         shuffle=True,
#         drop_last=False,
#     )

#     # ------------------------------------------------------------
#     # Optimizer and scheduler
#     # ------------------------------------------------------------
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=config.lr,
#         weight_decay=config.weight_decay,
#     )

#     scheduler = optim.lr_scheduler.ExponentialLR(
#         optimizer,
#         gamma=config.scheduler_gamma,
#     )

#     # ------------------------------------------------------------
#     # Training history
#     # ------------------------------------------------------------
#     history: Dict[str, List[float]] = {
#         "train_nll": [],
#         "val_nll": [],
#     }

#     # ============================================================
#     # Training loop
#     # ============================================================
#     for epoch in range(config.epochs):
#         model.train()

#         running_loss: float = 0.0
#         n_seen: int = 0

#         for xb, zb in train_loader:
#             optimizer.zero_grad()

#             log_prob: torch.Tensor = model.log_prob(xb, zb)
#             loss: torch.Tensor = -log_prob.mean()

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

#         train_nll: float = running_loss / max(n_seen, 1)
#         history["train_nll"].append(train_nll)

#         # --------------------------------------------------------
#         # Validation evaluation
#         # --------------------------------------------------------
#         if x_val is not None and z_val is not None:
#             model.eval()
#             with torch.no_grad():
#                 val_nll: float = float(
#                     (-model.log_prob(x_val, z_val).mean()).item()
#                 )
#             history["val_nll"].append(val_nll)
#         else:
#             val_nll = None

#         # --------------------------------------------------------
#         # Logging
#         # --------------------------------------------------------
#         if config.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
#             if val_nll is None:
#                 print(f"[Epoch {epoch + 1:4d}] train_nll = {train_nll:.6f}")
#             else:
#                 print(
#                     f"[Epoch {epoch + 1:4d}] "
#                     f"train_nll = {train_nll:.6f}, val_nll = {val_nll:.6f}"
#                 )

#     return history


def fit_conditional_realnvp(
    x_train: torch.Tensor,
    z_train: torch.Tensor,
    config: FlowTrainConfig = FlowTrainConfig(),
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
    model: ConditionalRealNVP = ConditionalRealNVP(
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
    history: Dict[str, List[float]] = train_conditional_realnvp(
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
# sampling from estimated conditional laws X | Z and Y | Z.
#
# The design mirrors the interface of OracleGenerators so that
# the fitted generator can be passed directly into routines such
# as generate_conditional_samples(...).
#
# Supported spaces currently include:
#   - Euclidean space
#   - SPD matrix space
#
# In the SPD case, inputs are first mapped to Euclidean
# coordinates via the log-Cholesky transform before model
# fitting and sampling, and generated Euclidean samples are then
# mapped back to the SPD space.
# ============================================================


class FittedConditionalGenerators:
    r"""
    Fitted conditional generators for ``X | Z`` and ``Y | Z``.

    This class wraps two trained conditional RealNVP models:
    one for approximating the conditional law of ``X | Z``,
    and one for approximating the conditional law of ``Y | Z``.

    The class is designed to mirror the interface of
    ``OracleGenerators`` so that it can be passed directly into
    ``generate_conditional_samples(...)``.

    Supported space types currently:
    - ``"euclidean"``
    - ``"spd"``

    For ``"spd"``, the class automatically applies the
    log-Cholesky transform before training and sampling, and
    maps generated samples back to SPD matrices afterward.

    Notes
    -----
    All external data inputs are assumed to be NumPy arrays.
    Internal conversion to torch tensors is handled automatically
    when GPU-based computation is requested.
    """

    def __init__(
        self,
        x_model: ConditionalRealNVP,
        y_model: ConditionalRealNVP,
        space_type: str,
        GPU: bool = False,
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
        if space_type not in {"euclidean", "spd"}:
            raise ValueError("space_type must be one of {'euclidean', 'spd'}.")

        self.x_model = x_model
        self.y_model = y_model
        self.space_type = space_type
        self.GPU = GPU
        self.dtype = dtype

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        space_type: str = "euclidean",
        config: FlowTrainConfig = FlowTrainConfig(),
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale_limit: float = 2.0,
        dropout: float = 0.0,
        GPU: bool = False,
        dtype: torch.dtype = torch.float64,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        z_val: Optional[np.ndarray] = None,
    ) -> tuple["FittedConditionalGenerators", Dict[str, Dict[str, List[float]]]]:
        r"""
        Fit conditional flow models for ``X | Z`` and ``Y | Z``.

        Parameters
        ----------
        X : np.ndarray
            Observed samples of ``X``.

            - Euclidean case: shape ``(n, d)``
            - SPD case: shape ``(n, p, p)``

        Y : np.ndarray
            Observed samples of ``Y``.

            - Euclidean case: shape ``(n, d)``
            - SPD case: shape ``(n, p, p)``

        Z : np.ndarray
            Observed conditioning variables.

            - Euclidean case: shape ``(n, d_z)``
            - SPD case: shape ``(n, p_z, p_z)``

        space_type : {"euclidean", "spd"}, default="euclidean"
            Type of the sample space.

        config : FlowTrainConfig, default=FlowTrainConfig()
            Training configuration for both conditional flow models.

        hidden_dim : int, default=128
            Width of hidden layers in coupling networks.

        num_hidden_layers : int, default=2
            Number of hidden layers in each coupling network.

        scale_limit : float, default=2.0
            Upper bound applied to scale outputs in coupling layers.

        dropout : float, default=0.0
            Dropout probability in coupling networks.

        GPU : bool, default=False
            Whether training and transformation should use GPU.

        dtype : torch.dtype, default=torch.float64
            Torch dtype used for tensor conversion.

        x_val : np.ndarray or None, default=None
            Optional validation set for ``X``.

        y_val : np.ndarray or None, default=None
            Optional validation set for ``Y``.

        z_val : np.ndarray or None, default=None
            Optional validation set for ``Z``.

        Returns
        -------
        fitted : FittedConditionalGenerators
            A fitted generator object wrapping the trained models.

        history : dict[str, dict[str, list[float]]]
            Training history dictionary with keys:
            - ``"x_history"``
            - ``"y_history"``

        Raises
        ------
        ValueError
            If ``space_type`` is invalid.

        TypeError
            If input arrays are not NumPy arrays.
        """
        if space_type not in {"euclidean", "spd"}:
            raise ValueError("space_type must be one of {'euclidean', 'spd'}.")

        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray) or not isinstance(Z, np.ndarray):
            raise TypeError("X, Y, Z must all be np.ndarray.")

        if x_val is not None and not isinstance(x_val, np.ndarray):
            raise TypeError("x_val must be np.ndarray or None.")
        if y_val is not None and not isinstance(y_val, np.ndarray):
            raise TypeError("y_val must be np.ndarray or None.")
        if z_val is not None and not isinstance(z_val, np.ndarray):
            raise TypeError("z_val must be np.ndarray or None.")
        
        # ------------------------------------------------------------
        # Select computation device
        # ------------------------------------------------------------
        if GPU:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU=True but CUDA is not available.")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # ------------------------------------------------------------
        # Prepare training / validation data as torch.Tensor
        # ------------------------------------------------------------
        if space_type == "euclidean":
            X_trf = torch.as_tensor(X, device=device, dtype=dtype)
            Y_trf = torch.as_tensor(Y, device=device, dtype=dtype)
            Z_trf = torch.as_tensor(Z, device=device, dtype=dtype)

            X_val_trf = None if x_val is None else torch.as_tensor(
                x_val, device=device, dtype=dtype
            )
            Y_val_trf = None if y_val is None else torch.as_tensor(
                y_val, device=device, dtype=dtype
            )
            Z_val_trf = None if z_val is None else torch.as_tensor(
                z_val, device=device, dtype=dtype
            )

        else:  # space_type == "spd"
            if GPU:
                X_trf = spd_to_euclidean(
                    torch.as_tensor(X, device=device, dtype=dtype),
                    GPU=True,
                )
                Y_trf = spd_to_euclidean(
                    torch.as_tensor(Y, device=device, dtype=dtype),
                    GPU=True,
                )
                Z_trf = spd_to_euclidean(
                    torch.as_tensor(Z, device=device, dtype=dtype),
                    GPU=True,
                )

                X_val_trf = None if x_val is None else spd_to_euclidean(
                    torch.as_tensor(x_val, device=device, dtype=dtype),
                    GPU=True,
                )
                Y_val_trf = None if y_val is None else spd_to_euclidean(
                    torch.as_tensor(y_val, device=device, dtype=dtype),
                    GPU=True,
                )
                Z_val_trf = None if z_val is None else spd_to_euclidean(
                    torch.as_tensor(z_val, device=device, dtype=dtype),
                    GPU=True,
                )
            else:
                X_trf = torch.as_tensor(
                    spd_to_euclidean(X, GPU=False),
                    device=device,
                    dtype=dtype,
                )
                Y_trf = torch.as_tensor(
                    spd_to_euclidean(Y, GPU=False),
                    device=device,
                    dtype=dtype,
                )
                Z_trf = torch.as_tensor(
                    spd_to_euclidean(Z, GPU=False),
                    device=device,
                    dtype=dtype,
                )

                X_val_trf = None if x_val is None else torch.as_tensor(
                    spd_to_euclidean(x_val, GPU=False),
                    device=device,
                    dtype=dtype,
                )
                Y_val_trf = None if y_val is None else torch.as_tensor(
                    spd_to_euclidean(y_val, GPU=False),
                    device=device,
                    dtype=dtype,
                )
                Z_val_trf = None if z_val is None else torch.as_tensor(
                    spd_to_euclidean(z_val, GPU=False),
                    device=device,
                    dtype=dtype,
                )
        
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
            GPU=GPU,
            dtype=dtype,
        )

        history = {
            "x_history": x_history,
            "y_history": y_history,
        }
        return fitted, history

    def _transform_z(
        self,
        Z: np.ndarray,
    ) -> Union[np.ndarray, torch.Tensor]:
        r"""
        Transform conditioning variables into the internal Euclidean representation.

        Parameters
        ----------
        Z : np.ndarray
            Conditioning value(s) in the original space.

            - Euclidean case: shape ``(..., z_dim)``
            - SPD case: shape ``(..., p, p)``

        Returns
        -------
        np.ndarray or torch.Tensor
            Transformed conditioning value(s) suitable for the
            fitted conditional flow models.
        """
        if self.space_type == "euclidean":
            return Z

        if self.GPU:
            Z_t = torch.as_tensor(Z, device="cuda", dtype=self.dtype)
            return spd_to_euclidean(Z_t, GPU=True)

        return spd_to_euclidean(Z, GPU=False)

    def _inverse_transform_x(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        r"""
        Map generated Euclidean samples back to the original sample space.

        Parameters
        ----------
        X : np.ndarray
            Generated samples in the internal Euclidean representation.

        Returns
        -------
        np.ndarray
            Samples mapped back to the original space.

            - Euclidean case: unchanged
            - SPD case: transformed back to SPD matrices
        """
        if self.space_type == "euclidean":
            return X
        return euclidean_to_spd(X, GPU=False)

    def __call__(
        self,
        Z: np.ndarray,
        space_type: str = "euclidean",
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Generate one conditional sample pair ``(X, Y)`` given a single conditioning value.

        Parameters
        ----------
        Z : np.ndarray
            Conditioning value in the original space.

            - Euclidean case: shape ``(z_dim,)``
            - SPD case: shape ``(p, p)``

        space_type : {"euclidean", "spd"}, default="euclidean"
            Space type requested for sampling. Must match the type
            on which the generator was trained.

        rng : np.random.Generator or None, default=None
            Random number generator used to seed the two conditional samplers.

        Returns
        -------
        X_gen : np.ndarray
            One generated sample from the fitted conditional law ``X | Z``.

        Y_gen : np.ndarray
            One generated sample from the fitted conditional law ``Y | Z``.

        Raises
        ------
        ValueError
            If the requested space type does not match the fitted one.

        TypeError
            If ``Z`` is not a NumPy array.
        """
        if space_type != self.space_type:
            raise ValueError(
                f"This fitted generator was trained for space_type={self.space_type}, "
                f"but got space_type={space_type}."
            )

        if not isinstance(Z, np.ndarray):
            raise TypeError("Z must be np.ndarray.")

        if rng is None:
            rng = np.random.default_rng()

        # ------------------------------------------------------------
        # Draw independent seeds for the X and Y samplers
        # ------------------------------------------------------------
        seed_x = int(rng.integers(0, 2**31 - 1))
        seed_y = int(rng.integers(0, 2**31 - 1))

        # ------------------------------------------------------------
        # Transform the conditioning value to the model space
        # ------------------------------------------------------------
        z_trf = self._transform_z(Z)

        # ------------------------------------------------------------
        # Generate one sample from each fitted conditional model
        # ------------------------------------------------------------
        x_gen_trf = self.x_model.sample_numpy(
            z=z_trf,
            n_samples=1,
            seed=seed_x,
        )[0]

        y_gen_trf = self.y_model.sample_numpy(
            z=z_trf,
            n_samples=1,
            seed=seed_y,
        )[0]

        # ------------------------------------------------------------
        # Map generated samples back to the original space
        # ------------------------------------------------------------
        x_gen = self._inverse_transform_x(np.asarray(x_gen_trf))
        y_gen = self._inverse_transform_x(np.asarray(y_gen_trf))

        return x_gen, y_gen

    def generate_batch(
        self,
        Z: np.ndarray,
        M: int,
        space_type: str = "euclidean",
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Generate batched conditional samples for a batch of conditioning values.

        For each conditioning value ``Z_i``, this method generates ``M``
        independent samples from the fitted conditional laws
        ``X | Z_i`` and ``Y | Z_i``.

        Parameters
        ----------
        Z : np.ndarray
            Batch of conditioning values.

            - Euclidean case: shape ``(n, z_dim)``
            - SPD case: shape ``(n, p, p)``

        M : int
            Number of conditional replications generated for each observation.

        space_type : {"euclidean", "spd"}, default="euclidean"
            Space type requested for sampling. Must match the type
            on which the generator was trained.

        rng : np.random.Generator or None, default=None
            Random number generator used to seed the two conditional samplers.

        Returns
        -------
        X_gen : np.ndarray
            Batched generated samples from the fitted conditional law of X.

            - Euclidean case: shape ``(n, M, x_dim)``
            - SPD case: shape ``(n, M, p, p)``

        Y_gen : np.ndarray
            Batched generated samples from the fitted conditional law of Y.

            - Euclidean case: shape ``(n, M, y_dim)``
            - SPD case: shape ``(n, M, p, p)``

        Raises
        ------
        ValueError
            If the requested space type does not match the fitted one.

        TypeError
            If ``Z`` is not a NumPy array.
        """
        if space_type != self.space_type:
            raise ValueError(
                f"This fitted generator was trained for space_type={self.space_type}, "
                f"but got space_type={space_type}."
            )

        if not isinstance(Z, np.ndarray):
            raise TypeError("Z must be np.ndarray.")

        if rng is None:
            rng = np.random.default_rng()

        n = Z.shape[0]

        # ------------------------------------------------------------
        # Transform Z to the internal Euclidean representation
        # ------------------------------------------------------------
        Z_trf = self._transform_z(Z)

        # ------------------------------------------------------------
        # Expand each conditioning value M times
        # Resulting shape:
        #   (n*M, z_dim)  for Euclidean / transformed SPD
        # ------------------------------------------------------------
        if isinstance(Z_trf, np.ndarray):
            Z_rep = np.repeat(Z_trf, repeats=M, axis=0)
        else:
            Z_rep = Z_trf.repeat_interleave(M, dim=0)

        # ------------------------------------------------------------
        # Use one seed for X samples and one seed for Y samples
        # ------------------------------------------------------------
        seed_x = int(rng.integers(0, 2**31 - 1))
        seed_y = int(rng.integers(0, 2**31 - 1))

        # ------------------------------------------------------------
        # Generate all samples in one shot
        # ------------------------------------------------------------
        X_gen_trf = self.x_model.sample_numpy(
            z=Z_rep,
            n_samples=None,
            seed=seed_x,
        )

        Y_gen_trf = self.y_model.sample_numpy(
            z=Z_rep,
            n_samples=None,
            seed=seed_y,
        )

        # ------------------------------------------------------------
        # Map back to the original space if needed
        # ------------------------------------------------------------
        X_gen = self._inverse_transform_x(X_gen_trf)
        Y_gen = self._inverse_transform_x(Y_gen_trf)

        # ------------------------------------------------------------
        # Reshape back to (n, M, ...)
        # ------------------------------------------------------------
        X_gen = X_gen.reshape((n, M) + X_gen.shape[1:])
        Y_gen = Y_gen.reshape((n, M) + Y_gen.shape[1:])

        return X_gen, Y_gen


# ============================================================
# Train / Validation Split for (X, Y, Z)
# ------------------------------------------------------------
# This utility function performs a randomized split of paired
# observations (X, Y, Z) into training and validation subsets.
#
# The split is performed by generating a shared random permutation
# of indices, ensuring that the correspondence between X, Y, and Z
# is preserved across both subsets.
#
# The validation size is determined by a user-specified ratio,
# with at least one sample guaranteed in the validation set.
#
# This function is typically used prior to training conditional
# generative models, enabling validation-based monitoring such as
# early stopping.
# ============================================================


def train_val_split_xyz(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    val_ratio: float = 0.25,
    seed: int = 2026,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    r"""
    Split paired samples (X, Y, Z) into training and validation sets.

    Parameters
    ----------
    X : np.ndarray
        Input samples of shape ``(n_samples, ...)``.

    Y : np.ndarray
        Paired samples corresponding to X, same first dimension.

    Z : np.ndarray
        Conditioning variables corresponding to X and Y,
        same first dimension.

    val_ratio : float, default=0.25
        Fraction of samples assigned to the validation set.

    seed : int, default=2026
        Random seed used for reproducible shuffling.

    Returns
    -------
    X_train : np.ndarray
        Training subset of X.

    Y_train : np.ndarray
        Training subset of Y.

    Z_train : np.ndarray
        Training subset of Z.

    X_val : np.ndarray
        Validation subset of X.

    Y_val : np.ndarray
        Validation subset of Y.

    Z_val : np.ndarray
        Validation subset of Z.

    Notes
    -----
    The split is performed by applying a shared random permutation
    to all three arrays, ensuring that the triplet correspondence
    (X_i, Y_i, Z_i) is preserved.

    At least one validation sample is always included, even when
    ``val_ratio`` is very small.
    """

    # ------------------------------------------------------------
    # Number of samples
    # ------------------------------------------------------------
    n: int = X.shape[0]

    # ------------------------------------------------------------
    # Determine validation set size (at least 1 sample)
    # ------------------------------------------------------------
    n_val: int = max(1, int(round(n * val_ratio)))

    # ------------------------------------------------------------
    # Generate shared random permutation
    # ------------------------------------------------------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    # ------------------------------------------------------------
    # Split indices
    # ------------------------------------------------------------
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # ------------------------------------------------------------
    # Return split datasets
    # ------------------------------------------------------------
    return (
        X[train_idx], Y[train_idx], Z[train_idx],
        X[val_idx],   Y[val_idx],   Z[val_idx],
    )