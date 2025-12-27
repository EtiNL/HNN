import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from hnn_utils import batch_bisection_solve, batch_bisection_solve_diag, is_diagonalizable, compute_alpha_beta, dilation_batch_diag


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, Normalize

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, Normalize

def plot_model_heatmaps_triptych(
    model_hnn,
    model_mlp,
    objective_function,
    Gd,
    P=None, nu=None,
    bound=5.0,
    grid_size=300,
    logscale=True,                    # log color scale when data > 0
    linthresh=1e-3,                   # symlog linear half-width (if needed)
    cmap_pos='viridis',               # sequential map for positive data
    cmap_div='seismic',               # diverging map if symlog is used
    # --- Dilation overlay (no s_max) ---
    overlay_dilation=True,
    n_start=12,                       # number of starting points on the circle
    s_min=0.1,                        # starting dilation
    ds=0.02,                          # step in s
    max_steps=2000,                   # safety cap
    unit='P',                         # 'P' for P-unit circle, 'euclidean' for ||x||_2=1
    traj_alpha=0.9, traj_lw=1.5,
    traj_color='k',                   # <-- single color for all trajectories
    legend_loc='upper left'           # where to place the legend on the first panel
):
    """
    Plots Ground truth | hNN prediction | MLP prediction with a single colorbar.
    Overlays dilation trajectories gamma_x0(s) = exp(s Gd) x0 for s >= s_min
    with step ds, stopping each trajectory once it exits [-bound, bound]^2.

    All trajectories share the same color and a single legend label.
    """
    device = Gd.device
    x_range = np.linspace(-bound, bound, grid_size)
    y_range = np.linspace(-bound, bound, grid_size)
    Xg, Yg = np.meshgrid(x_range, y_range)
    grid_points = torch.tensor(
        np.column_stack([Xg.ravel(), Yg.ravel()]),
        dtype=torch.float32, device=device
    )

    # --- Ground truth ---
    y_true = objective_function(grid_points)
    if isinstance(y_true, tuple): y_true = y_true[0]
    y_true = y_true.reshape(grid_size, grid_size).detach().cpu().numpy()

    # --- hNN prediction ---
    y_hnn = model_hnn(grid_points)
    if isinstance(y_hnn, tuple): y_hnn = y_hnn[0]
    y_hnn = y_hnn.reshape(grid_size, grid_size).detach().cpu().numpy()

    # --- MLP prediction ---
    y_mlp = model_mlp(grid_points)
    if isinstance(y_mlp, tuple): y_mlp = y_mlp[0]
    y_mlp = y_mlp.reshape(grid_size, grid_size).detach().cpu().numpy()

    # --- Unified normalization ---
    arrays = [y_true, y_hnn, y_mlp]
    vmin_all = min(a.min() for a in arrays)
    vmax_all = max(a.max() for a in arrays)

    if logscale and vmin_all > 0:
        norm = LogNorm(vmin=max(vmin_all, 1e-12), vmax=vmax_all)
        cmap = cmap_pos
    elif logscale:
        lim = max(abs(vmin_all), abs(vmax_all))
        norm = SymLogNorm(linthresh=max(linthresh, 1e-12), vmin=-lim, vmax=lim)
        cmap = cmap_div
    else:
        norm = Normalize(vmin=vmin_all, vmax=vmax_all)
        cmap = cmap_pos

    # --- Plot: 3 panels, one colorbar ---
    titles = [
        "Ground Truth",
        f"{getattr(model_hnn, 'name', 'hNN')} Prediction",
        f"{getattr(model_mlp, 'name', 'MLP')} Prediction"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    ims = []
    for ax, data, title in zip(axes, [y_true, y_hnn, y_mlp], titles):
        im = ax.imshow(
            data, extent=(-bound, bound, -bound, bound), origin='lower',
            cmap=cmap, norm=norm
        )
        ims.append(im)
        ax.set_title(title)
        ax.set_xlabel('x1'); ax.set_ylabel('x2')

    # Single unified colorbar
    cbar = fig.colorbar(ims[0], ax=axes, location='right', fraction=0.046, pad=0.04)
    cbar.set_label('Output (log scale)' if logscale else 'Output')

    # ---------- Dilation trajectories (grow until out of bounds) ----------
    if overlay_dilation:
        # choose exp(s Gd)
        def _expG(s):
            if torch.allclose(Gd, torch.diag(torch.diag(Gd))):
                lam = torch.diag(Gd)
                return torch.diag(torch.exp(lam * s))
            try:
                eigvals, eigvecs = torch.linalg.eig(Gd)
                V, Vinv = eigvecs, torch.linalg.inv(eigvecs)
                D = torch.diag(torch.exp(eigvals.real * s))
                return (V @ D @ Vinv).real
            except RuntimeError:
                return torch.matrix_exp(s * Gd)

        # starting points on chosen unit circle
        thetas = torch.linspace(0, 2*np.pi, steps=n_start+1, device=device)[:-1]
        if unit.lower() == 'p' and P is not None:
            evals, evecs = torch.linalg.eigh(P)
            Pinvsqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
            X0 = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)
            X0 = (Pinvsqrt @ X0.T).T
        else:
            X0 = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)

        # plot with a single color and a single legend entry
        label_set = False
        for i in range(n_start):
            x0 = X0[i].to(device)
            traj = []
            s = float(s_min)
            steps = 0
            while steps < max_steps:
                Es = _expG(s)
                xs = (Es @ x0)
                xs_np = xs.detach().real.cpu().numpy()
                if np.any(np.abs(xs_np) > bound):
                    break
                traj.append(xs_np)
                s += float(ds)
                steps += 1

            if len(traj) > 1:
                traj = np.array(traj)
                # label only once on the first panel
                axes[0].plot(
                    traj[:, 0], traj[:, 1],
                    color=traj_color, lw=traj_lw, alpha=traj_alpha,
                    label=None if label_set else 'Dilation trajectories'
                )
                # other panels without label
                for ax in axes[1:]:
                    ax.plot(traj[:, 0], traj[:, 1],
                            color=traj_color, lw=traj_lw, alpha=traj_alpha)
                label_set = True

        # show legend on the first panel only
        axes[0]



def plot_mse_vs_dilation_on_test_loader(
    model_hnn,
    model_mlp,
    objective_function,
    test_loader,
    Gd,
    *,
    nu,
    s_min=0.0,
    s_max=3.0,
    num_points=20,
    use_fast_diag=True,
    ylog=True,
    log_eps=1e-12,
    plot_theory=True,
    normalize_theory=None
):
    """
    Computes MSE(model(d(s)X_test), f(d(s)X_test)) vs s for hNN and MLP,
    where d(s) = exp(s Gd). Plots on a log y-axis and (optionally) overlays exp(2*nu*s).
    Returns (s_values, hnn_mse, mlp_mse, theory) as numpy arrays (theory=None if disabled).
    """

    device = Gd.device

    # ---------- pick dilation routine once ----------
    def _make_dilate():
        if use_fast_diag:
            # diagonal
            if torch.allclose(Gd, torch.diag(torch.diag(Gd))):
                lam = torch.diag(Gd)
                return lambda S, X: dilation_batch_diag(lam, S, X)
            # diagonalizable
            if is_diagonalizable(Gd):
                eigvals, eigvecs = torch.linalg.eig(Gd)
                lam = eigvals.real
                V = eigvecs
                V_inv = torch.linalg.inv(V)
                return lambda S, X: dilation_batch_diag((lam, V, V_inv), S, X)
        # generic fallback
        return lambda S, X: dilation_batch(Gd, S, X)

    dilate = _make_dilate()

    # ---------- eval mode ----------
    model_hnn.eval()
    model_mlp.eval()

    s_grid = torch.linspace(s_min, s_max, steps=num_points, device=device)
    mse_hnn, mse_mlp = [], []

    with torch.no_grad():
        for s in s_grid:
            sum_sq_h = 0.0
            sum_sq_m = 0.0
            n_total  = 0

            for bx, _by in test_loader:
                B = bx.size(0)
                S = torch.full((B,), float(s), device=device)

                Xs = dilate(S, bx)                     # d(s) * X
                y_true = objective_function(Xs)
                if isinstance(y_true, tuple):
                    y_true = y_true[0]
                y_true = y_true.view(B, -1)

                # hNN predictions
                yh = model_hnn(Xs);  yh = yh[0] if isinstance(yh, tuple) else yh
                yh = yh.view(B, -1)

                # MLP predictions
                ym = model_mlp(Xs);  ym = ym[0] if isinstance(ym, tuple) else ym
                ym = ym.view(B, -1)

                sum_sq_h += ((yh - y_true) ** 2).sum().item()
                sum_sq_m += ((ym - y_true) ** 2).sum().item()
                n_total  += B

            mse_hnn.append(sum_sq_h / n_total)
            mse_mlp.append(sum_sq_m / n_total)

    # ---------- theory curve ----------
    s_np   = s_grid.detach().cpu().numpy()
    h_mse  = np.array(mse_hnn)
    m_mse  = np.array(mse_mlp)
    theory = None
    if plot_theory:
        theory = np.exp(2.0 * float(nu) * s_np)  # e^{2 nu s}
        if normalize_theory == 'hnn' and len(h_mse) > 0:
            theory = theory * (h_mse[0] / max(theory[0], 1e-30))
        elif normalize_theory == 'mlp' and len(m_mse) > 0:
            theory = theory * (m_mse[0] / max(theory[0], 1e-30))

    # ---------- plot ----------
    plt.figure(figsize=(7, 4.5))
    if ylog:
        plt.semilogy(s_np, np.maximum(h_mse, log_eps), label='hNN MSE')
        plt.semilogy(s_np, np.maximum(m_mse, log_eps), label='MLP MSE')
        if plot_theory:
            plt.semilogy(s_np, np.maximum(theory, log_eps), linestyle='--', label=r'$\exp(2\nu s)$')
        plt.ylabel('Test MSE (log scale)')
        plt.grid(True, which='both', alpha=0.5)
    else:
        plt.plot(s_np, h_mse, label='hNN MSE')
        plt.plot(s_np, m_mse, label='MLP MSE')
        if plot_theory:
            plt.plot(s_np, theory, linestyle='--', label=r'$\exp(2\nu s)$')
        plt.ylabel('Test MSE')

    plt.xlabel(r'Dilation $s$ in $d(s)=\exp(s G_d)$')
    plt.title('Generalization under anisotropic dilation (test set)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return s_np, h_mse, m_mse, (None if not plot_theory else theory)
