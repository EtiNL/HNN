def test_homogeneity_model(P, Gd, nu, num_tests=10, s_range=(-2, 2)):
    """
    Tests the homogeneity property of a HomogeneousNN model.

    Parameters
    ----------
    model : HomogeneousNN
        The neural network model to be tested.
    Gd : torch.Tensor
        The matrix Gd used in the dilation operation.
    nu : float
        The degree of homogeneity.
    num_tests : int, optional
        Number of test cases to evaluate. Default is 10.
    s_range : tuple, optional
        Range of values for s, sampled uniformly. Default is (-2, 2).

    Returns
    -------
    None. Prints the relative error of the homogeneity test.
    """

    input_dim = Gd.shape[0]
    hidden_dim = np.random.choice([4,8,16])

    model = HomogeneousNN(input_dim, hidden_dim, output_dim, P, Gd, nu)
    # Generate random test points
    x = torch.randn(num_tests, Gd.shape[0])
    
    # Generate random values for s in the given range
    s_values = (s_range[1] - s_range[0]) * torch.rand(num_tests) + s_range[0]
    
    # Compute d(s) * x for each s
    exp_matrices = torch.stack([torch.matrix_exp(s * Gd) for s in s_values])  # Shape (num_tests, n, n)
    x_dilated = torch.bmm(exp_matrices, x.unsqueeze(-1)).squeeze(-1)  # Shape (num_tests, n)
    
    # Compute model outputs
    f_x = model(x).detach().squeeze()  # Shape (num_tests,)
    f_x_dilated = model(x_dilated).detach().squeeze()  # Shape (num_tests,)
    
    # Compute expected homogeneity-scaled values
    expected_f_x_dilated = torch.exp(nu * s_values) * f_x
    
    # Compute relative error
    relative_error = torch.abs((f_x_dilated - expected_f_x_dilated) / (expected_f_x_dilated + 1e-8))
    
    # Print results
    print("Homogeneity test results for HomogeneousNN:")
    for i in range(num_tests):
        print(f"Test {i+1}: s = {s_values[i]:.4f}, Relative error = {relative_error[i].item():.6f}")
    
    print(f"\nAverage relative error: {relative_error.mean().item():.6f}")


def test_f_anisotropic_homogeneity(f_anisotropic, Gd, nu, num_tests=10, s_range=(-2, 2)):
    """
    Tests the homogeneity property of f_anisotropic.

    Parameters
    ----------
    f_anisotropic : function
        The function to be tested.
    Gd : torch.Tensor
        The matrix Gd used in the dilation operation.
    nu : float
        The degree of homogeneity.
    num_tests : int, optional
        Number of test cases to evaluate. Default is 10.
    s_range : tuple, optional
        Range of values for s, sampled uniformly. Default is (-2, 2).

    Returns
    -------
    None. Prints the relative error of the homogeneity test.
    """
    # Generate random test points
    x = torch.randn(num_tests, Gd.shape[0])
    
    # Generate random values for s in the given range
    s_values = (s_range[1] - s_range[0]) * torch.rand(num_tests) + s_range[0]
    
    # Compute d(s) * x for each s
    exp_matrices = torch.stack([torch.matrix_exp(s * Gd) for s in s_values])  # Shape (num_tests, n, n)
    x_dilated = torch.bmm(exp_matrices, x.unsqueeze(-1)).squeeze(-1)  # Shape (num_tests, n)
    
    # Compute function values
    f_x = f_anisotropic(x)  # Shape (num_tests,)
    f_x_dilated = f_anisotropic(x_dilated)  # Shape (num_tests,)
    
    # Compute expected homogeneity-scaled values
    expected_f_x_dilated = torch.exp(nu * s_values) * f_x
    
    # Compute relative error
    relative_error = torch.abs((f_x_dilated - expected_f_x_dilated) / (expected_f_x_dilated + 1e-8))
    
    # Print results
    print("Homogeneity test results:")
    for i in range(num_tests):
        print(f"Test {i+1}: s = {s_values[i]:.4f}, Relative error = {relative_error[i].item():.6f}")
    
    print(f"\nAverage relative error: {relative_error.mean().item():.6f}")