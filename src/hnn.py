import torch
import torch.nn as nn
from hnn_utils import check_Gd_P_conditions, compute_alpha_beta, batch_bisection_solve, batch_bisection_solve_diag, dilation_batch, dilation_batch_diag, is_diagonalizable

class HomogeneousNN(nn.Module):
    """
    An anisotropic homogeneous NN with d(s) = exp(Gd s).
    We find s for each sample so that d(-s)(x) is on the unit 'd-sphere',
    then pass that to an MLP, and finally scale output by exp(nu * s).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, P, Gd, nu, hidden_layers = 1):
        """
        """
        super().__init__()
        self.name = 'HNN'
        self.nu = nu
        self.Gd = Gd
        self.P = P

        if not check_Gd_P_conditions(Gd, P):
           raise Exception('Gd P conditions not satisfied')

        self.alpha, self.beta = compute_alpha_beta(Gd, P)

        if torch.allclose(Gd, torch.diag(torch.diagonal(Gd)), atol=1e-6):
            self.Gd_is_diagonal = True
            self.lam = torch.diagonal(Gd)

        elif is_diagonalizable(Gd):
            device = Gd.device
            self.Gd_diagonalizable = True
            eigvals, eigvecs = torch.linalg.eig(Gd)
            self.V = eigvecs.to(device)
            self.V_inv = torch.linalg.inv(self.V).to(device)
            self.lam = eigvals.to(device)

        # MLP that acts on the "d-sphere"
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))

        self.net_on_sphere = nn.Sequential(*layers)

    def forward(self, x):
        """
        1) solves for the batch ||d(-s)x||d = 1
        2) Map x to sphere: x_sphere = d(-s)(x) x
        3) MLP_out = net_on_sphere(x_sphere)
        4) Scale by exp(nu * s)
        """
        # Solve for s in batch
        #S = batch_bisection_solve(self.Gd, self.P, x, self.alpha, self.beta)#.detach()  # shape = (batch_size,)
        if self.Gd_is_diagonal:
            with torch.no_grad():
                S = batch_bisection_solve_diag(self.lam, self.P, x, self.alpha, self.beta, tol=1e-4, max_iter=1000)

            x_sphere = dilation_batch_diag(self.lam, -S, x)
            
        elif self.Gd_diagonalizable:
            with torch.no_grad():
                S = batch_bisection_solve_diag([self.lam, self.V, self.V_inv], self.P, x, self.alpha, self.beta, tol=1e-4, max_iter=1000)

            x_sphere = dilation_batch_diag([self.lam, self.V, self.V_inv], -S, x)

        else:
            with torch.no_grad():
                S = batch_bisection_solve(self.Gd, self.P, x, self.alpha, self.beta, tol=1e-4, max_iter=1000)

            x_sphere = dilation_batch(self.Gd, -S, x)
        
        # Pass x_sphere into the MLP
        mlp_out = self.net_on_sphere(x_sphere)

        with torch.no_grad():
            # Finally, scale the output by exp(nu * s)
            scale_up = torch.exp(self.nu * S).unsqueeze(-1)  # shape=(batch_size,1)
            
        out = mlp_out * scale_up

        return out

if __name__ == '__main__':
    import torch
    Gd = torch.diag(torch.tensor([1.0, 1.0]))
    P  = torch.eye(2)
    model_hnn = HomogeneousNN(input_dim=2, hidden_dim=2, output_dim=1,
                           P=P, Gd=Gd, nu=2)