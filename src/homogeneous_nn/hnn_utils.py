import torch
import torch.nn as nn

def is_positive_definite(matrix):
    """
    Checks if a matrix is positive definite by ensuring all eigenvalues are positive.
    
    Parameters:
    matrix (torch.Tensor): A square matrix of shape (n, n).
    
    Returns:
    bool: True if the matrix is positive definite, False otherwise.
    """
    eigenvalues = torch.linalg.eigvalsh(matrix)  # Compute eigenvalues
    return torch.all(eigenvalues > 0).item()

def check_Gd_P_conditions(Gd, P):
    """
    Checks if the matrices Gd and P satisfy the conditions:
    1. P is positive definite: P ≻ 0
    2. P Gd + Gd^T P is positive definite: (P Gd + Gd^T P) ≻ 0
    
    Parameters:
    Gd (torch.Tensor): A square matrix of shape (n, n).
    P (torch.Tensor): A square matrix of shape (n, n), expected to be positive definite.
    
    Returns:
    bool: True if both conditions are met, False otherwise.
    """
    if not is_positive_definite(P):
        print("P is not positive definite.")
        return False
    
    # Compute M = P Gd + Gd^T P
    M = torch.matmul(P, Gd) + torch.matmul(Gd.T, P)

    if not is_positive_definite(M):
        print("P Gd + Gd^T P is not positive definite.")
        return False

    return True

import torch, numpy as np

def norm_P(x, P):
    """
    Computes the norm induced by P: sqrt(x.T @ P @ x).

    Parameters
    ----------
    x : torch.Tensor
        Vector of shape (n,).
    P : torch.Tensor
        Positive definite matrix of shape (n, n).

    Returns
    -------
    torch.Tensor
        A scalar (0-dim tensor) representing the P-norm.
    """
    return torch.sqrt(torch.matmul(x.T, torch.matmul(P, x))).item()


def norm_P_batch(X, P):
    """
    Batch version of sqrt(x_i^T P x_i).

    X : (B, n)
    P : (n, n) Positive definite matrix

    Returns : (B,) norms in R^B
    """
    # (B, n) @ (n, n) -> (B, n)
    PX = X @ P
    # Elementwise product and sum across dim=1 -> (B,)
    quad_forms = (X * PX).sum(dim=1)
    return torch.sqrt(quad_forms)

def dilation(Gd, s, x):
    """
    Applies the dilation operator: exp(s*Gd) on the vector x.

    Parameters
    ----------
    Gd : torch.Tensor
        Matrix G_d of shape (n, n).
    s : float
        Scalar coefficient.
    x : torch.Tensor
        Vector of shape (n,).

    Returns
    -------
    torch.Tensor
        The resulting vector after applying the matrix exponential.
    """
    return torch.matmul(torch.matrix_exp(s * Gd), x)

def dilation_batch(Gd, S, X):
    """
    Applies the dilation operator: exp(-S[i] * Gd) on each vector X[i].

    Parameters
    ----------
    Gd : torch.Tensor
        Matrix G_d of shape (n, n).
    S : torch.Tensor
        Tensor of shape (batch_size,) containing scalar coefficients.
    X : torch.Tensor
        Batch of vectors of shape (batch_size, n).

    Returns
    -------
    torch.Tensor
        The resulting batch of vectors after applying the matrix exponential.
        Shape: (batch_size, n)
    """

    
    batch_size, n = X.shape

    # Compute exp(-S[i] * Gd) for each i using a loop, since torch.matrix_exp does not support batching
    exp_matrices = torch.stack([torch.matrix_exp(s * Gd) for s in S])  # Shape (batch_size, n, n)

    # Apply the matrix exponential to each vector in X (bmm = batch matrix multiplication)
    return torch.bmm(exp_matrices, X.unsqueeze(-1)).squeeze(-1)  # Shape (batch_size, n)

def dilation_batch_diag(lam_or_triplet, S, X):
    """
    Apply anisotropic dilation d(s) = exp(Gd * s) to batch of vectors X.
    Supports:
        - lam: if Gd is diagonal
        - [lam, V, V_inv]: if Gd is diagonalizable
    """
    if isinstance(lam_or_triplet, list) or isinstance(lam_or_triplet, tuple):
        lam, V, V_inv = lam_or_triplet
        exp_lam = torch.exp(lam[None, :] * S[:, None])  # (B, n)
        exp_diag = torch.diag_embed(exp_lam)            # (B, n, n)
        exp_matrices = (V[None, ...] @ exp_diag @ V_inv[None, ...]).real  # (B, n, n)
        return torch.bmm(exp_matrices, X.unsqueeze(-1)).squeeze(-1)
    else:
        lam = lam_or_triplet
        exp_lam = torch.exp(lam[None, :] * S[:, None])  # (B, n)
        return X * exp_lam


def compute_alpha_beta(Gd, P):
    """
    Computes alpha and beta from Gd and the symmetric matrix P.

    Parameters
    ----------
    Gd : torch.Tensor
        Square matrix G_d of shape (n, n).
    P : torch.Tensor
        Positive definite and symmetric matrix of shape (n, n).

    Returns
    -------
    alpha : float
        The maximum eigenvalue of the symmetrized matrix S divided by 2.
    beta : float
        The minimum eigenvalue of S divided by 2.
    """
    # Compute the symmetric square root of P via eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(P)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    sqrt_P = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    inv_sqrt_eigenvalues = 1.0 / sqrt_eigenvalues
    inv_sqrt_P = eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T

    # Construct the symmetrized matrix S
    S = torch.matmul(sqrt_P, torch.matmul(Gd, inv_sqrt_P)) + \
        torch.matmul(inv_sqrt_P, torch.matmul(Gd.T, sqrt_P))
    
    eigen_S = torch.linalg.eigvalsh(S)
    
    alpha = torch.max(eigen_S).item() / 2.0
    beta  = torch.min(eigen_S).item() / 2.0
    return alpha, beta


def batch_bisection_solve(Gd, P, X, alpha, beta, tol=1e-6, max_iter=100):
    """
    Parallel bisection to solve:
      norm_P( dilation(Gd, -s, x_i), P ) ≈ 1
    for all samples x_i in X, simultaneously on the GPU.

    Gd   : (n, n)
    P    : (n, n)
    X    : (B, n)
    alpha: scalar
    beta : scalar
    tol  : float
    max_iter : int

    Returns : (B,) solutions s_i for each x_i
    """
    device = Gd.device

    # (B,) - compute norm of each sample
    norms_X = norm_P_batch(X, P)

    # The original code: a = norm_x^(1/alpha), b = norm_x^(1/beta)
    a = norms_X ** (1.0 / alpha)
    b = norms_X ** (1.0 / beta)

    # s_min, s_max = log(min(a,b)), log(max(a,b)) for each element
    s_min = torch.log(torch.minimum(a, b)).to(device)
    s_max = torch.log(torch.maximum(a, b)).to(device)

    for _ in range(max_iter):
        # (B,)
        s_mid = 0.5 * (s_min + s_max)

        # Compute norm of d(-s_mid)(X)
        ds_mid = dilation_batch(Gd, -s_mid, X)  # (B, n)
        norm_ds_mid = norm_P_batch(ds_mid, P)   # (B,)

        # mask: True if norm_ds_mid < 1 -> we move s_max downward
        mask = norm_ds_mid < 1.0

        # in-place updates
        s_max[mask] = s_mid[mask]
        s_min[~mask] = s_mid[~mask]

        # Check convergence: if the difference is small for all
        if (s_max - s_min).max() < tol:
            break

    # Return final midpoint
    return 0.5 * (s_min + s_max)


def batch_bisection_solve_diag(lam_or_triplet, P, X, alpha, beta, tol=1e-6, max_iter=100):
    """
    Solve for s such that norm_P(d(-s)(x)) ≈ 1 using bisection.
    Supports:
        - lam: if Gd is diagonal
        - [lam, V, V_inv]: if Gd is diagonalizable

    Returns: (B,) tensor of s values
    """
    device = X.device

    # (B,) - norm of each sample
    norms_X = norm_P_batch(X, P)

    a = norms_X ** (1.0 / alpha)
    b = norms_X ** (1.0 / beta)

    s_min = torch.log(torch.minimum(a, b)).to(device)
    s_max = torch.log(torch.maximum(a, b)).to(device)

    for _ in range(max_iter):
        s_mid = 0.5 * (s_min + s_max)

        ds_mid = dilation_batch_diag(lam_or_triplet, -s_mid, X)

        norm_ds_mid = norm_P_batch(ds_mid, P)

        mask = norm_ds_mid < 1.0
        s_max[mask] = s_mid[mask]
        s_min[~mask] = s_mid[~mask]

        if (s_max - s_min).max() < tol:
            break

    return 0.5 * (s_min + s_max)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(42)  # Ensures same init for every run
        nn.init.xavier_uniform_(m.weight)  # Xavier Initialization
        if m.bias != None:
            nn.init.zeros_(m.bias)  # Bias initialized to zero

def is_diagonalizable(Gd, tol=1e-6):
    # 1) calcul des valeurs propres (complexes) et vecteurs propres
    eigvals, eigvecs = torch.linalg.eig(Gd)
    n = Gd.shape[0]
    # regrouper par valeur propre (avec tolérance)
    uniq_vals = []
    for λ in eigvals:
        if not any(torch.isclose(λ, μ, atol=tol) for μ in uniq_vals):
            uniq_vals.append(λ)

    # 2) pour chaque valeur propre, comparer multiplicités
    for λ in uniq_vals:
        # multiplicité algébrique
        alg_mult = int((torch.isclose(eigvals, λ, atol=tol)).sum().item())
        # multiplicité géométrique = dimension du noyau de (Gd - λ I)
        M = Gd - λ.real * torch.eye(n, device=Gd.device)
        # on utilise svd ou rank pour estimer dim ker
        rank_M = torch.linalg.matrix_rank(M, tol=tol)
        geo_mult = n - rank_M
        if geo_mult < alg_mult:
            return False
    return True