# To provide an extensive implementation of the Approximate Bayesian Computation (ABC)
# rejection sampling algorithm, we'll simulate a scenario where we aim to estimate
# parameters of a distribution based on observed data. Let's consider a simple example
# where we want to estimate the mean of a Gaussian distribution, given observed data
# that we assume comes from a Gaussian distribution with a known variance but unknown
# mean.

# We'll implement the following steps in the `abc_rejection_sampling` function:
# 1. Define a prior distribution for the parameter (mean) we want to estimate.
# 2. Define a simulator function that generates data given a parameter value.
# 3. Define a distance function to measure how close the simulated data is to the observed data.
# 4. Repeatedly sample parameter values from the prior, simulate data with these parameter
#    values, and accept the parameter values if the distance between the simulated and
#    observed data is below a threshold.

import numpy as np


def prior_sampler() -> float:
    """
    Samples from the prior distribution of the parameter (mean of the Gaussian).
    For simplicity, we assume the prior is a Gaussian centered at 0 with a std of 10.
    """
    return np.random.normal(0, 10)


def simulator(param: float, sample_size: int = 100) -> np.ndarray:
    """
    Simulates data given the parameter (mean of the Gaussian) and a fixed variance.
    Here, we assume a variance of 1 for the Gaussian distribution.
    """
    return np.random.normal(param, 1, sample_size)


def distance(simulated_data: np.ndarray, observed_data: np.ndarray) -> float:
    """
    Computes the distance between the simulated data and the observed data.
    We use the mean absolute difference as a simple distance metric.
    """
    return np.abs(simulated_data - observed_data).mean()


def abc_rejection_sampling(observed_data: np.ndarray, epsilon: float, n_samples: int) -> np.ndarray:
    """
    Approximate Bayesian Computation via rejection sampling to estimate the parameter
    posterior distribution given observed data and a distance threshold (epsilon).

    Parameters:
        observed_data (np.ndarray): The observed data to compare against.
        epsilon (float): Acceptance threshold for the distance.
        n_samples (int): Number of accepted samples to generate.

    Returns:
        np.ndarray: Parameters accepted from the prior that result in simulated data
        close to the observed data.
    """
    accepted_params = []
    attempts = 0
    while len(accepted_params) < n_samples and attempts < 10000:
        param = prior_sampler()
        simulated_data = simulator(param, sample_size=len(observed_data))
        if distance(simulated_data, observed_data) < epsilon:
            accepted_params.append(param)
        attempts += 1

    return np.array(accepted_params)


# Example usage of the ABC rejection sampling algorithm:
if __name__ == "__main__":
    # Generate some observed data with a true mean of 5 and known variance of 1
    true_mean = 5
    observed_data = np.random.normal(true_mean, 1, 100)

    # Set the distance threshold and number of samples
    epsilon = 0.5  # Acceptance threshold
    n_samples = 50  # Number of parameter samples to accept

    # Perform ABC rejection sampling
    estimated_means = abc_rejection_sampling(observed_data, epsilon, n_samples)

    # The user can analyze the estimated_means to understand the posterior distribution
    # of the parameter of interest (mean of the Gaussian in this case).
    print(f"Estimated Means: {estimated_means}")
    print(f"Mean of Estimated Means: {np.mean(estimated_means)}")
    print(f"Standard Deviation of Estimated Means: {np.std(estimated_means)}")

# This extensive example demonstrates how to implement and use the ABC rejection
# sampling algorithm for parameter estimation in a simple scenario. The choice of prior,
# simulator, and distance function can be adjusted based on the specific application and
# model under consideration.
