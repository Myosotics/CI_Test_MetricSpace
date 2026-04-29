import torch
from dataclasses import dataclass
import torch
from typing import Callable, Optional, Dict


# ============================================================
# Data generation
# ============================================================


def _normalize_tangent(
    W: torch.Tensor,
    Z: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    atol: float = 1e-14,
) -> torch.Tensor:
    """
    Normalize tangent vectors so that:
        - W ⟂ Z
        - ||W|| = 1

    Handles degenerate (near-zero) vectors via resampling.
    """
    d = W.shape[1]

    norm = torch.linalg.norm(W, dim=1, keepdim=True)
    mask = norm.squeeze(-1) < atol

    while bool(mask.any()):
        num_bad = int(mask.sum().item())

        W_new = torch.randn(num_bad, d, device=device, dtype=dtype)
        Z_bad = Z[mask]

        # project to tangent space
        W_new = W_new - (W_new * Z_bad).sum(dim=1, keepdim=True) * Z_bad

        W[mask] = W_new

        norm = torch.linalg.norm(W, dim=1, keepdim=True)
        mask = norm.squeeze(-1) < atol

    return W / norm


def generate_data_spherical(
    n: int,
    size: int = 2,
    rho: float = 0.0,
    seed: int | None = None,
    sigma_perm: float = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Generate simulated spherical data ``(X, Y, Z)`` using PyTorch.
    """

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        raise TypeError("device must be torch.device or None")

    # ------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------
    if n <= 0:
        raise ValueError("n must be positive")

    if size < 2:
        raise ValueError("size must be >= 2 for sphere")

    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    d = size
    rho_comp = max(0.0, 1.0 - rho**2) ** 0.5

    # ------------------------------------------------------------
    # Step 1: Z on sphere
    # ------------------------------------------------------------
    Z = torch.randn(n, d, device=device, dtype=dtype)
    Z = Z / torch.linalg.norm(Z, dim=1, keepdim=True)

    # ------------------------------------------------------------
    # Step 2: tangent directions
    # ------------------------------------------------------------
    U = torch.randn(n, d, device=device, dtype=dtype)
    U = U - (U * Z).sum(dim=1, keepdim=True) * Z

    V = torch.randn(n, d, device=device, dtype=dtype)
    V = V - (V * Z).sum(dim=1, keepdim=True) * Z

    U = _normalize_tangent(U, Z, device, dtype)
    V = _normalize_tangent(V, Z, device, dtype)

    # ------------------------------------------------------------
    # Step 3: correlated noise
    # ------------------------------------------------------------
    eps1 = torch.randn(n, 1, device=device, dtype=dtype)
    eps2 = torch.randn(n, 1, device=device, dtype=dtype)

    xi_x = sigma_perm * eps1
    xi_y = sigma_perm * (rho * eps1 + rho_comp * eps2)

    # ------------------------------------------------------------
    # Step 4: project back to sphere
    # ------------------------------------------------------------
    X = Z + xi_x * U
    Y = Z + xi_y * V

    X = X / torch.linalg.norm(X, dim=1, keepdim=True)
    Y = Y / torch.linalg.norm(Y, dim=1, keepdim=True)

    return X, Y, Z