# HomogeneousNN (anisotropic dilation)

Anisotropic homogeneous neural network with dilation $d(s)=\exp(G_d s)$. For each input $x$, the model solves $||d(-s)x||_P=1$, evaluates an MLP on the resulting "d-sphere," then rescales the output by $\exp(\nu s)$.

---

## Contents

* [Features](#features)
* [Project layout](#project-layout)
* [Install](#install)
* [Quick start](#quick-start)
* [Mathematical notes](#mathematical-notes)
* [API](#api)

  * [`HomogeneousNN`](#homogeneousnn)
  * `hnn_utils.py` helpers
  * `plots_utils.py` utilities
  * Tests in `test_homogeneity.py`
* [Performance and stability](#performance-and-stability)
* [Reproducibility](#reproducibility)

---

## Features

* Enforces anisotropic homogeneity of degree $\nu$ via explicit rescaling.
* Batch bisection solver for $s$ in generic, diagonal, and diagonalizable cases.
* Device-aware matrix exponential paths.
* Plotting helpers for heatmaps and MSE vs dilation.
* Simple homogeneity tests for a model or a target function.

## Project layout

```
.
├── __init__.py
├── hnn.py                 # HomogeneousNN implementation
├── hnn_utils.py           # math utilities and batch solvers
├── plots_utils.py         # visualization utilities
└── test_homogeneity.py    # homogeneity tests
```

## Install

Requirements:

* Python ≥ 3.9
* PyTorch ≥ 2.1
* NumPy ≥ 1.24
* Matplotlib ≥ 3.7

Install with pip:

```bash
pip install -r requirements.txt
```

## Quick start

Minimal usage with diagonal $G_d$ and $P=I$:

```python
import torch
from hnn import HomogeneousNN

# Dimensions
n_in, n_hidden, n_out = 2, 64, 1

# Geometry
Gd = torch.diag(torch.tensor([1.0, 2.0]))  # anisotropic rates
P  = torch.eye(2)
nu = 1.0

model = HomogeneousNN(n_in, n_hidden, n_out, P=P, Gd=Gd, nu=nu, hidden_layers=2)

x = torch.randn(32, n_in)
y = model(x)  # shape (32, 1)
```

Training loop sketch:

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for step in range(1000):
    x = torch.randn(128, n_in)
    target = some_objective(x).detach()
    pred = model(x)
    loss = loss_fn(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
```

Homogeneity test on a trained model:

```python
from test_homogeneity import test_homogeneity_model

test_homogeneity_model(P=P, Gd=Gd, nu=nu, num_tests=10, s_range=(-2, 2))
```

Heatmap triptych and MSE vs dilation:

```python
from plots_utils import plot_model_heatmaps_triptych, plot_mse_vs_dilation_on_test_loader

# objective_function: callable X -> y with X shape (B, n)
plot_model_heatmaps_triptych(model_hnn=model,
                             model_mlp=mlp_baseline,
                             objective_function=objective_function,
                             Gd=Gd,
                             P=P,
                             bound=3.0,
                             grid_size=200)

s, mse_h, mse_m, theory = plot_mse_vs_dilation_on_test_loader(
    model_hnn=model,
    model_mlp=mlp_baseline,
    objective_function=objective_function,
    test_loader=test_loader,
    Gd=Gd,
    nu=nu,
    s_min=0.0,
    s_max=3.0,
    num_points=25,
)
```

## Mathematical notes

* Dilation: $d(s) = \exp(G_d s) \in \mathbb{R}^{n\times n}$.
* P-norm: $||x||_P = \sqrt{x^\top P x}$ with $P \succ 0$.
* Normalization step: find $s$ s.t. $||d(-s) x||_P = 1$. Solve by batch bisection.
* Model output: $f(x) = g(d(-s)x) \exp(\nu s)$, where $g$ is an MLP on the d-sphere.
* Conditions enforced at init: $P \succ 0$ and $P G_d + G_d^\top P \succ 0$. These imply positive bounds for the bisection interval via `compute_alpha_beta`.
* `compute_alpha_beta`: constructs $S = P^{1/2} G_d P^{-1/2} + P^{-1/2} G_d^\top P^{1/2}$, sets $\alpha = \tfrac{\lambda_{\max}(S)}{2}$, $\beta = \tfrac{\lambda_{\min}(S)}{2}$.

## API

### `HomogeneousNN`

```python
HomogeneousNN(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    P: torch.Tensor,
    Gd: torch.Tensor,
    nu: float,
    hidden_layers: int = 1,
)
```

Behavior:

1. Check `check_Gd_P_conditions(Gd, P)`.
2. Compute `(alpha, beta) = compute_alpha_beta(Gd, P)`.
3. Choose fast path:

   * If `Gd` is diagonal: use elementwise exponentiation.
   * Else if `is_diagonalizable(Gd)`: precompute eigenbasis once and use (V \exp(\Lambda s) V^{-1}).
   * Else: generic `torch.matrix_exp` per sample.
4. Forward pass:

   * Solve (s) by batch bisection with `torch.no_grad()`.
   * Map (x) to d-sphere: `x_sphere = d(-s) x`.
   * Evaluate `net_on_sphere(x_sphere)`.
   * Rescale by `exp(nu * s)` with `torch.no_grad()`.

Notes:

* Gradients flow through the MLP only. The scalar (s) is treated as constant in backprop by design. This stabilizes training but disables learning through the normalization step.
* Output layer bias is disabled to keep strict homogeneity.

Attributes:

* `self.Gd_is_diagonal: bool` when (G_d) diagonal.
* `self.Gd_diagonalizable: bool` when diagonalizable.
* `self.lam, self.V, self.V_inv` as available.
* `self.alpha, self.beta` for bisection bracketing.

Shapes:

* Input: `(B, input_dim)`.
* Output: `(B, output_dim)`.

#### `hnn_utils.py`

Key functions:

* `is_positive_definite(P) -> bool`
* `check_Gd_P_conditions(Gd, P) -> bool`
* `norm_P(x, P) -> float`, `norm_P_batch(X, P) -> Tensor[(B,)]`
* `dilation(Gd, s, x) -> Tensor[(n,)]`
* `dilation_batch(Gd, S, X) -> Tensor[(B, n)]`
* `dilation_batch_diag(lam_or_triplet, S, X) -> Tensor[(B, n)]`

  * `lam_or_triplet` is either `lam` for diagonal (G_d) or `(lam, V, V_inv)` for diagonalizable (G_d).
* `compute_alpha_beta(Gd, P) -> (alpha: float, beta: float)`
* `batch_bisection_solve(Gd, P, X, alpha, beta, tol=1e-6, max_iter=100)`
* `batch_bisection_solve_diag(lam_or_triplet, P, X, alpha, beta, tol=1e-6, max_iter=100)`
* `initialize_weights(m)` Xavier init for `nn.Linear`.
* `is_diagonalizable(Gd, tol=1e-6) -> bool` using eigenvalue clustering and rank tests.

#### `plots_utils.py`

* `plot_model_heatmaps_triptych(model_hnn, model_mlp, objective_function, Gd, P=None, nu=None, bound=5.0, grid_size=300, ...)`

  * Renders ground truth, hNN prediction, MLP prediction on a common color scale.
  * Optionally overlays dilation trajectories (\gamma_{x_0}(s)=\exp(sG_d)x_0) grown until they exit the plotting box.
  * Works with diagonal or diagonalizable (G_d); falls back to `matrix_exp`.

* `plot_mse_vs_dilation_on_test_loader(model_hnn, model_mlp, objective_function, test_loader, Gd, *, nu, s_min=0.0, s_max=3.0, num_points=20, use_fast_diag=True, ylog=True, plot_theory=True, normalize_theory=None)`

  * Computes (\operatorname{MSE}(\text{model}(d(s)X), f(d(s)X))) versus (s) on the test set.
  * Optionally overlays the theory curve (\exp(2\nu s)).
  * Returns `(s_values, hnn_mse, mlp_mse, theory)` as `numpy` arrays.

#### `test_homogeneity.py`

* `test_homogeneity_model(P, Gd, nu, num_tests=10, s_range=(-2, 2))`

  * Builds a small `HomogeneousNN`, samples `x` and `s`, checks ( f(d(s)x) \approx e^{\nu s} f(x) ).
* `test_f_anisotropic_homogeneity(f_anisotropic, Gd, nu, num_tests=10, s_range=(-2, 2))`

  * Same test for a target function `f_anisotropic`.

## Performance and stability

* Prefer diagonal or diagonalizable (G_d) to avoid per-sample `matrix_exp`.
* Complexity per forward:

  * Diagonal: `O(B n)` for dilation and `O(n)` storage.
  * Diagonalizable: one-time eigendecomposition then `O(B n^2)` multiplications.
  * Generic: `O(B n^3)` due to `matrix_exp` per sample.
* Bisection is numerically stable under the provided PD conditions. `tol` and `max_iter` control precision and runtime.
* `torch.no_grad()` around (s) prevents gradient explosions through the normalization path.

## Reproducibility

* `initialize_weights` seeds each `nn.Linear` with `torch.manual_seed(42)` and uses Xavier uniform. Biases are zeroed.
* Set a global seed in your scripts for full determinism.

