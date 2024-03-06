import numpy as np


def bayesian_update(mu_prior, sigma_prior, data, sigma_data):
    """
    Perform a Bayesian update on a prior belief given new data.

    Parameters:
    - mu_prior: Mean of the prior distribution.
    - sigma_prior: Standard deviation of the prior distribution.
    - data: New data points (numpy array).
    - sigma_data: Standard deviation of the data.

    Returns:
    - mu_post: Mean of the posterior distribution.
    - sigma_post: Standard deviation of the posterior distribution.
    """
    sigma_prior_squared = sigma_prior ** 2
    sigma_data_squared = sigma_data ** 2

    mu_post = (mu_prior / sigma_prior_squared + data.sum() / sigma_data_squared) / \
              (1 / sigma_prior_squared + len(data) / sigma_data_squared)
    sigma_post = 1 / (1 / sigma_prior_squared + len(data) / sigma_data_squared)

    return mu_post, np.sqrt(sigma_post)


# Example usage
if __name__ == "__main__":
    # Prior belief: mean = 0, standard deviation = 1
    mu_prior, sigma_prior = 0, 1
    # New data simulated from a normal distribution with mean = 0.5
    data = np.random.normal(0.5, 0.5, 100)
    # Standard deviation of the data
    sigma_data = 0.5

    mu_post, sigma_post = bayesian_update(mu_prior, sigma_prior, data, sigma_data)
    print(f"Posterior mean: {mu_post}, Posterior standard deviation: {sigma_post}")
