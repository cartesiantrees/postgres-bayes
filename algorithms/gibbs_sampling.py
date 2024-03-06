# Implementing the Gibbs Sampling algorithm for a bivariate normal distribution.
# This example demonstrates how to sample from a joint distribution of two variables
# (X, Y) assuming a bi-variate normal distribution with known means, variances, and
# correlation. The Gibbs Sampling method iteratively samples from the conditional
# distributions of each variable, given the other.

import numpy as np


def gibbs_sampling(mu_x: float, mu_y: float, sigma_x: float, sigma_y: float,
                   rho: float, n_iterations: int) -> np.ndarray:
    """
    Gibbs Sampling for a bivariate normal distribution.

    Parameters:
        mu_x (float): Mean of X.
        mu_y (float): Mean of Y.
        sigma_x (float): Standard deviation of X.
        sigma_y (float): Standard deviation of Y.
        rho (float): Correlation between X and Y.
        n_iterations (int): Number of iterations to perform.

    Returns:
        np.ndarray: Samples from the joint distribution, shape (n_iterations, 2).
    """
    samples = np.zeros((n_iterations, 2))
    x, y = 0.0, 0.0  # Initial values

    for i in range(n_iterations):
        # Sample X given Y
        sigma_x_given_y = np.sqrt(sigma_x ** 2 * (1 - rho ** 2))
        mu_x_given_y = mu_x + rho * (y - mu_y) * sigma_x / sigma_y
        x = np.random.normal(mu_x_given_y, sigma_x_given_y)

        # Sample Y given X
        sigma_y_given_x = np.sqrt(sigma_y ** 2 * (1 - rho ** 2))
        mu_y_given_x = mu_y + rho * (x - mu_x) * sigma_y / sigma_x
        y = np.random.normal(mu_y_given_x, sigma_y_given_x)

        samples[i, :] = [x, y]

    return samples


# Example usage:
if __name__ == "__main__":
    mu_x, mu_y = 0, 0  # Means
    sigma_x, sigma_y = 1, 1  # Standard deviations
    rho = 0.5  # Correlation coefficient
    n_iterations = 10000

    samples = gibbs_sampling(mu_x, mu_y, sigma_x, sigma_y, rho, n_iterations)

    # The user can analyze the samples, for example, by plotting to visualize the joint distribution.
    print(f"First few samples:\n{samples[:5]}")

    # Additional analysis like plotting the samples or computing summary statistics could be added here.

# This implementation of the Gibbs Sampling algorithm demonstrates how to iteratively sample
# from conditional distributions within a bivariate normal distribution. The method is
# particularly useful in higher dimensions where direct sampling from the joint distribution
# is challenging. The choice of initial values, number of iterations, and handling of
# potential autocorrelation between samples are important considerations for practical applications.
# The user can analyze the samples, for example, by plotting to visualize the joint distribution.
