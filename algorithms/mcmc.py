# Implementing a Markov Chain Monte Carlo (MCMC) method using the Metropolis-Hastings algorithm.
# This example will demonstrate how to estimate the parameters of a Gaussian distribution,
# specifically its mean, given a set of observations. The Metropolis-Hastings algorithm
# allows us to sample from the posterior distribution of the parameter of interest
# without knowing the full Bayesian posterior.

import numpy as np


def metropolis_hastings(data: np.ndarray, prior_mu: float, prior_sigma: float,
                        proposal_width: float, n_iterations: int) -> np.ndarray:
    """
    Metropolis-Hastings algorithm to sample from the posterior distribution of a
    parameter of interest, here the mean of a Gaussian distribution, given observed data.

    Parameters:
        data (np.ndarray): Observed data assumed to come from a Gaussian distribution.
        prior_mu (float): Mean of the prior distribution of the parameter (mean of the Gaussian).
        prior_sigma (float): Standard deviation of the prior distribution.
        proposal_width (float): Width of the proposal distribution. Controls the step size.
        n_iterations (int): Number of iterations to perform.

    Returns:
        np.ndarray: Samples from the posterior distribution of the parameter.
    """
    # Initialize the chain with a random value
    mu_current = np.random.rand()
    posterior_samples = [mu_current]

    for i in range(n_iterations):
        # Propose a new value (mu_proposal) for the mean from a normal distribution
        mu_proposal = np.random.normal(mu_current, proposal_width)

        # Calculate the likelihood by assuming a Gaussian model
        likelihood_current = np.sum(np.log(np.random.normal(mu_current, 1, len(data))))
        likelihood_proposal = np.sum(np.log(np.random.normal(mu_proposal, 1, len(data))))

        # Compute prior probability of current and proposed mu
        prior_current = np.log(np.random.normal(prior_mu, prior_sigma, 1))
        prior_proposal = np.log(np.random.normal(prior_mu, prior_sigma, 1))

        # Calculate the acceptance probability
        p_accept = np.exp((likelihood_proposal + prior_proposal) -
                          (likelihood_current + prior_current))

        # Accept proposal with probability p_accept
        if np.random.rand() < p_accept:
            mu_current = mu_proposal
        posterior_samples.append(mu_current)

    return np.array(posterior_samples)


# Example usage:
if __name__ == "__main__":
    # Simulating some data: 1000 observations from a Gaussian with a true mean of 10
    true_mu = 10
    data = np.random.normal(true_mu, 1, 1000)

    # Running the Metropolis-Hastings algorithm
    n_iterations = 5000
    samples = metropolis_hastings(data, prior_mu=0, prior_sigma=1,
                                  proposal_width=0.5, n_iterations=n_iterations)

    print(f"Mean of sampled posterior distribution: {np.mean(samples)}")
    # The user can further analyze the `samples` array, for example, by plotting
    # the histogram of samples to visualize the posterior distribution.

# Note: This implementation is a simplified example intended to demonstrate the
# Metropolis-Hastings algorithm. In practice, considerations like the choice of the
# prior, proposal distribution, and convergence diagnostics are crucial for robust
# Bayesian inference.
