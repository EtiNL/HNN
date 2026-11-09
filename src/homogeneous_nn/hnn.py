import torch
import torch.nn as nn
from homogeneous_nn.hnn_utils import check_Gd_P_conditions, compute_alpha_beta, batch_bisection_solve, batch_bisection_solve_diag, dilation_batch, dilation_batch_diag, is_diagonalizable

class HomogeneousNN(nn.Module):
    """
    An anisotropic homogeneous NN with d(s) = exp(Gd s).
    We find s for each sample so that d(-s)(x) is on the unit 'd-sphere',
    then pass that to an MLP, and finally scale output by exp(nu * s).

    The homogeneous norm ||x||_d = e^S, with S defined by ||d(-S) x||_P = 1, is obtained
      * from `homogeneous_norm` when an explicit (closed-form) expression is given: it is
        then differentiated directly by autograd (exact, no bisection);
      * otherwise by bisection, with the exact gradient dS/dx attached by the implicit
        function theorem (see `_differentiable_S`).
    """
    def __init__(self, input_dim, hidden_dim, P, Gd, nu, hidden_layers = 1, is_field = False,
                 homogeneous_norm=None, check_norm: bool = True,
                 bisection_tol: float = 1e-10, bisection_max_iter: int = 200):
        """
        Args:
            homogeneous_norm: optional callable x (B, n) -> ||x||_d (B,), a closed-form
                canonical homogeneous norm for this (Gd, P), written with differentiable torch
                operations. Example: if Gd - I/2 is antisymmetric and P = I, then
                ||d(s)x||_2 = e^{s/2} ||x||_2, hence ||x||_d = ||x||_2^2:
                    homogeneous_norm = lambda x: (x * x).sum(dim=-1)
                If None, the norm is computed by bisection.
            check_norm: if True and homogeneous_norm is given, check on random samples that it
                satisfies the defining equation ||d(-ln ||x||_d) x||_P = 1 (raises otherwise).
            bisection_tol: width of the final bisection interval on S. It must be small: the
                field is only smooth up to this tolerance (a loose value, e.g. 1e-4, degrades the
                Jacobian and forces adaptive ODE solvers to tiny steps). The implicit-gradient
                Newton step then squares this error.
            bisection_max_iter: maximum number of bisection iterations.
        """
        super().__init__()
        self.name = 'HNN'
        self.nu = nu
        self.is_field = is_field
        self.bisection_tol = bisection_tol
        self.bisection_max_iter = bisection_max_iter
        # Buffers (not plain attributes) so that .to(device) / .double() also convert them
        self.register_buffer("Gd", Gd)
        self.register_buffer("P", P)

        if not check_Gd_P_conditions(Gd, P):
           raise Exception('Gd P conditions not satisfied')

        self.alpha, self.beta = compute_alpha_beta(Gd, P)

        if torch.allclose(Gd, torch.diag(torch.diagonal(Gd)), atol=1e-6):
            self.Gd_is_diagonal = True
            self.Gd_diagonalizable = True
            self.register_buffer("lam", torch.diagonal(Gd).clone())

        elif is_diagonalizable(Gd):
            device = Gd.device
            self.Gd_is_diagonal = False
            self.Gd_diagonalizable = True
            # computed in float64 (complex128) whatever the dtype of Gd; _eig() casts down if needed
            eigvals, eigvecs = torch.linalg.eig(Gd.to(torch.float64))
            self.register_buffer("V", eigvecs.to(device))
            self.register_buffer("V_inv", torch.linalg.inv(eigvecs).to(device))
            self.register_buffer("lam", eigvals.to(device))
        
        else:
            self.Gd_is_diagonal = False
            self.Gd_diagonalizable = False

        # MLP that acts on the "d-sphere"
        if is_field:
            output_dim = input_dim
        else:
            output_dim = 1

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))

        self.net_on_sphere = nn.Sequential(*layers)

        self.homogeneous_norm = homogeneous_norm
        if homogeneous_norm is not None and check_norm:
            self._check_homogeneous_norm()

    @torch.no_grad()
    def _check_homogeneous_norm(self, n_samples: int = 64, rtol: float = 1e-6):
        """Check that the explicit norm satisfies ||d(-ln N(x)) x||_P = 1 on random samples."""
        g = torch.Generator(device="cpu").manual_seed(0)
        n = self.Gd.shape[0]
        x = torch.randn(n_samples, n, generator=g, dtype=torch.float64)
        x = x * torch.exp(torch.randn(n_samples, 1, generator=g, dtype=torch.float64))  # several scales
        Gd, P = self.Gd.to(torch.float64).cpu(), self.P.to(torch.float64).cpu()
        S = torch.log(self.homogeneous_norm(x))
        z = dilation_batch(Gd, -S, x)
        err = ((z * (z @ P)).sum(dim=-1).sqrt() - 1.0).abs().max().item()
        if not err < rtol:
            raise ValueError(
                f"homogeneous_norm does not satisfy ||d(-ln N(x)) x||_P = 1 for this (Gd, P) "
                f"(max error {err:.2e}); it is not the canonical homogeneous norm."
            )

    def _eig(self, dtype):
        """Eigen-decomposition [lam, V, V_inv] of a diagonalizable Gd, in the complex dtype
        matching `dtype` (.to()/.double() do not convert complex buffers to complex128)."""
        ct = torch.complex128 if dtype == torch.float64 else torch.complex64
        return [self.lam.to(ct), self.V.to(ct), self.V_inv.to(ct)]

    def _dilate(self, s, x):
        """d(s) x = exp(s Gd) x for a batch, using the cheapest available form of Gd."""
        if self.Gd_is_diagonal:
            return dilation_batch_diag(self.lam, s, x)
        if self.Gd_diagonalizable:
            return dilation_batch_diag(self._eig(x.dtype), s, x)
        return dilation_batch(self.Gd, s, x)

    def _differentiable_S(self, x, S0):
        r"""
        Attach the exact gradient dS/dx to the bisection solution S0, without changing its value.

        S = ln ||x||_d is defined implicitly by
            F(S, x) := ||d(-S) x||_P^2 - 1 = 0.
        The bisection gives the value S0 but no gradient (it runs under no_grad, and an
        iterative search has no useful derivative anyway). Autograd would then treat S as a
        constant and drop every term of the Jacobian df/dx that goes through S (the whole
        radial variation of the field: Euler's identity df/dx . Gd x = nu f + Gd f is lost).

        Implicit function theorem: F(S(x), x) = 0 for all x, hence
            dF/dS * dS/dx + dF/dx = 0   =>   dS/dx = - (dF/dx) / (dF/dS).

        We return
            S = S0 - F(S0, x) / kappa,      kappa = dF/dS (S0, x),
        where S0 and kappa are detached (constants for autograd):
          * value:    F(S0, x) ~ 0, so S ~ S0. This is one Newton step, so the value is
                      even more accurate (error ~ (S0 - S*)^2).
          * gradient: autograd only sees x inside F(S0, x), so
                      dS/dx = - (dF/dx)(S0, x) / kappa,  exactly the implicit-function formula.
        kappa is detached: its own x-dependence would only contribute terms multiplied by
        F(S0, x) ~ 0.

        With z = d(-S0) x:  F = z^T P z - 1  and  d z / dS = -Gd z,  so
            kappa = -2 z^T P Gd z.
        """
        S0 = S0.detach()
        z = self._dilate(-S0, x)                       # gradient flows through x only
        F = (z * (z @ self.P)).sum(dim=-1) - 1.0       # z^T P z - 1  (P symmetric)
        Gz = z @ self.Gd.T                             # (Gd z) for each sample
        kappa = (-2.0 * (z * (Gz @ self.P)).sum(dim=-1)).detach()   # -2 z^T P Gd z
        return S0 - F / kappa

    def log_norm(self, x):
        """
        S = ln ||x||_d with its exact gradient dS/dx.
          * explicit norm given:  S = ln homogeneous_norm(x), differentiated by autograd;
          * otherwise:            bisection for the value (no_grad) + implicit-function gradient.
        """
        if self.homogeneous_norm is not None:
            return torch.log(self.homogeneous_norm(x))
        with torch.no_grad():
            if self.Gd_is_diagonal:
                S0 = batch_bisection_solve_diag(self.lam, self.P, x, self.alpha, self.beta, tol=self.bisection_tol, max_iter=self.bisection_max_iter)
            elif self.Gd_diagonalizable:
                S0 = batch_bisection_solve_diag(self._eig(x.dtype), self.P, x, self.alpha, self.beta, tol=self.bisection_tol, max_iter=self.bisection_max_iter)
            else:
                S0 = batch_bisection_solve(self.Gd, self.P, x, self.alpha, self.beta, tol=self.bisection_tol, max_iter=self.bisection_max_iter)
        return self._differentiable_S(x, S0)

    def forward(self, x):
        """
        1)-2) S = ln ||x||_d with its exact gradient (see log_norm: explicit norm if given,
              otherwise bisection + implicit-function gradient, see _differentiable_S)
        3) map x to the sphere: x_sphere = d(-S) x
        4) MLP_out = net_on_sphere(x_sphere)
        5) scale back: exp(nu S) d(S) MLP_out (field) or exp(nu S) MLP_out (function)
        Steps 3-5 use the differentiable S, so autograd sees every dependence of f on x.
        """
        # x = 0 has no direction (S = -inf): evaluate on a dummy point, output 0 there
        # (a homogeneous field/function of degree nu > 0 vanishes at the origin)
        zero = (x == 0).all(dim=-1)
        if zero.any():
            x = torch.where(zero.unsqueeze(-1), torch.ones_like(x), x)

        # 1)-2) S = ln ||x||_d, with exact gradient dS/dx
        S = self.log_norm(x)

        # 3)-4) projection on the sphere and MLP
        x_sphere = self._dilate(-S, x)
        mlp_out = self.net_on_sphere(x_sphere)

        # 5) scale back (outside no_grad: d/dx of exp(nu S) and d(S) are needed in the Jacobian)
        scale_up = torch.exp(self.nu * S).unsqueeze(-1)  # shape=(batch_size,1)
        if self.is_field:
            out = scale_up * self._dilate(S, mlp_out)   # cheapest form of exp(S Gd)
        else:
            out = scale_up * mlp_out

        if zero.any():
            out = torch.where(zero.unsqueeze(-1), torch.zeros_like(out), out)
        return out

if __name__ == '__main__':
    import torch
    Gd = torch.diag(torch.tensor([1.0, 1.0]))
    P  = torch.eye(2)
    model_hnn = HomogeneousNN(input_dim=2, hidden_dim=2, output_dim=1,
                           P=P, Gd=Gd, nu=2)