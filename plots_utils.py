import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from hnn_utils import batch_bisection_solve, batch_bisection_solve_diag, is_diagonalizable, compute_alpha_beta

def plot_model_heatmaps(model, objective_function, Gd, P, nu, bound, grid_size=300):
    # Fine grid for visualization (-10 to 10)
    device = Gd.device
    x_range = np.linspace(-bound, bound, grid_size)
    y_range = np.linspace(-bound, bound, grid_size)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    grid_points = torch.tensor(np.column_stack([X_grid.ravel(), Y_grid.ravel()]), dtype=torch.float32, device = device)

    # Ground truth values
    y_true_grid = objective_function(grid_points)
    if isinstance(y_true_grid, tuple):
        y_true_grid = y_true_grid[0]
    y_true_grid = y_true_grid.reshape(grid_size, grid_size).detach().cpu().numpy()

    # Model predictions
    y_pred_grid = model(grid_points)
    if isinstance(y_pred_grid, tuple):
        y_pred_grid = y_pred_grid[0]
    y_pred_grid = y_pred_grid.reshape(grid_size, grid_size).detach().cpu().numpy()

    
    diff_grid = (y_true_grid - y_pred_grid)
    
    max_abs_diff = abs(np.max(diff_grid))
    
    # Plot heatmaps
    plt.figure(figsize=(8, 8))

    # Grid difference heatmap
    plt.title(f'{model.name} (True - Predicted) Heatmap')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    heatmap1 = plt.imshow(diff_grid, extent=(-bound, bound, -bound, bound), origin='lower',
                          cmap='seismic', norm=SymLogNorm(linthresh=0.01, linscale=0.5, vmin=-max_abs_diff, vmax=max_abs_diff))
    plt.colorbar(heatmap1)

    plt.tight_layout()
    plt.show()

    mse = np.mean((y_true_grid - y_pred_grid)**2)
    print(f'Grid MSE: {mse:.4e}')