import torch
import torch.nn as nn
import torch.optim as optim

def F_and_dF(s, x, r):
    """
    Computes F(s) = sum_{i=1..n} [ e^{-2 r_i s} * x_i^2 ] - 1
    and its derivative dF/ds for a single sample x in R^n.
    """
    # e^{-2 r_i s} = exp(-2*r_i*s)
    exp_terms = torch.exp(-2.0 * r * s)
    # Weighted sum of x_i^2
    val = torch.sum(exp_terms * (x**2)) - 1.0

    # derivative dF/ds = sum_{i=1..n} [ -2 r_i e^{-2 r_i s} x_i^2 ]
    dval = torch.sum(-2.0 * r * exp_terms * (x**2))
    return val, dval

# use ln instead
# bisection methods

def solve_s_single(x, r, max_iter=30, tol=1e-7):
    """
    Solves F(s)=0 for a single sample x in R^n via Newton's method.
    That is, we find s such that:
        sum_i e^{-2 r_i s} x_i^2 = 1
    Returns the scalar s.
    """
    # Corner case: if x=0, define s=0 as a convention.
    if torch.allclose(x, torch.zeros_like(x), atol=1e-12):
        return 0.0

    s = torch.tensor(0.0, dtype=x.dtype, device=x.device, requires_grad=False)
    for _ in range(max_iter):
        val, dval = F_and_dF(s, x, r)
        if abs(val.item()) < tol:
            break
        # Newton step
        s_next = s - val/dval
        s = s_next
    return s.item()

def batch_solve_s(X, r):
    """
    Applies 'solve_s_single' to each sample in X (batch_size x n).
    Returns a tensor S of shape (batch_size,).
    """
    S = []
    for i in range(X.shape[0]):
        s_val = solve_s_single(X[i], r)
        S.append(s_val)
    return torch.tensor(S, dtype=X.dtype, device=X.device)

class AnisotropicHomogeneousNN(nn.Module):
    """
    An anisotropic homogeneous NN with d(s) = diag(e^{r_1 s},...,e^{r_n s}).
    We find s for each sample so that d(-s)(x) is on the unit 'd-sphere',
    then pass that to an MLP, and finally scale output by exp(nu * s).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, r_values, nu=1.0):
        """
        r_values: a 1D tensor of shape (input_dim,) specifying (r_1,...,r_n).
        nu: homogeneity degree
        """
        super().__init__()
        self.register_buffer('r_values', r_values)  # store on the same device
        self.nu = nu

        # MLP that acts on the "d-sphere"
        self.net_on_sphere = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        1) For each sample, solve F(s)=0 => sum_i e^{-2 r_i s} x_i^2 = 1
           to get s (the "log-scale").
        2) Map x to sphere: x_sphere = d(-s)(x) = diag(e^{-r_i s}) x
        3) MLP_out = net_on_sphere(x_sphere)
        4) Scale by exp(nu * s)
        """
        # Solve for s in batch
        S = batch_solve_s(x, self.r_values)  # shape = (batch_size,)
        S = S.view(-1, 1)  # make it shape=(batch_size,1)

        # x_sphere = d(-s)(x)
        # => x_sphere[i] = ( e^{-r_1 s_i} * x_i1, ..., e^{-r_n s_i} * x_in )
        # We'll broadcast elementwise
        scale_down = torch.exp(- self.r_values * S)  # shape=(batch_size, n)
        x_sphere = x * scale_down

        # Pass x_sphere into the MLP
        mlp_out = self.net_on_sphere(x_sphere)

        # Finally, scale the output by exp(nu * s)
        scale_up = torch.exp(self.nu * S)  # shape=(batch_size,1)
        out = mlp_out * scale_up
        return out


