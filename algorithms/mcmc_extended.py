# Correcting and enhancing the Metropolis-Hastings algorithm implementation with considerations
# for the likelihood and prior calculations, and including basic convergence diagnostics.

import numpy as np


def gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """
    Probability density function of a Gaussian distribution.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def metropolis_hastings_corrected(data: np.ndarray, prior_mu: float, prior_sigma: float,
                                  proposal_width: float, n_iterations: int) -> np.ndarray:
    """
    Corrected Metropolis-Hastings algorithm to sample from the posterior distribution
    of the Gaussian mean parameter.
    """
    mu_current = np.random.rand()  # Initial guess
    posterior_samples = []

    for i in range(n_iterations):
        # Propose a new mu from a normal distribution centered at the current mu
        mu_proposal = np.random.normal(mu_current, proposal_width)

        # Calculate likelihood by assuming data is from a Gaussian with proposed mu
        likelihood_current = np.sum(np.log(gaussian_pdf(data, mu_current, 1)))
        likelihood_proposal = np.sum(np.log(gaussian_pdf(data, mu_proposal, 1)))

        # Compute prior probability of current and proposed mu
        prior_current = np.log(gaussian_pdf(mu_current, prior_mu, prior_sigma))
        prior_proposal = np.log(gaussian_pdf(mu_proposal, prior_mu, prior_sigma))

        # Calculate acceptance probability
        p_accept = np.exp((likelihood_proposal + prior_proposal) -
                          (likelihood_current + prior_current))

        # Accept proposal?
        if np.random.rand() < p_accept:
            mu_current = mu_proposal

        posterior_samples.append(mu_current)

    return np.array(posterior_samples)


def basic_convergence_diagnostics(samples: np.ndarray):
    """
    A very basic convergence diagnostic, calculating and returning the mean and
    standard deviation of the second half of a sample set to assess its stability.
    """
    halfway = len(samples) // 2
    mean_estimate = np.mean(samples[halfway:])
    std_estimate = np.std(samples[halfway:])
    return mean_estimate, std_estimate


# Example usage with corrected implementation:
if __name__ == "__main__":
    data = np.random.normal(10, 1, 1000)  # Observed data
    n_iterations = 10000
    samples = metropolis_hastings_corrected(data, prior_mu=0, prior_sigma=10,
                                            proposal_width=0.5, n_iterations=n_iterations)

    mean_estimate, std_estimate = basic_convergence_diagnostics(samples)

    print(f"Mean of sampled posterior distribution: {mean_estimate}")
    print(f"Standard deviation of sampled posterior distribution: {std_estimate}")
    # TODO: Additional analysis like plotting the distribution of samples could be added here.


