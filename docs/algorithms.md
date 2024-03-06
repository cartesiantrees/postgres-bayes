# Gibbs Sampling Algorithm

## Overview
The Gibbs Sampling algorithm is a Markov Chain Monte Carlo (MCMC) method used to sample from the joint distribution of multiple variables when direct sampling is difficult. It is particularly useful for high-dimensional distributions and in Bayesian inference where the posterior distributions are complex.

## Implementation
This implementation demonstrates Gibbs Sampling for a bivariate normal distribution with known means, variances, and correlation. The algorithm iteratively samples from the conditional distributions of each variable, given the other.

### Key Concepts
- **Conditional Sampling**: At each step, one variable is sampled given the current value of the other, leveraging the conditional distributions derived from the joint distribution.
- **Bivariate Normal Distribution**: The example assumes a bivariate normal distribution for simplicity, but Gibbs Sampling can be applied to more complex and higher-dimensional distributions.

### Example Output:

The first few samples are shown, demonstrating the algorithm's ability to generate samples from the joint distribution of XX and YY. The output provides a glimpse into the nature of the sampled values, reflecting the specified means, variances, and correlation.
Further Analysis:

For practical applications or deeper analysis, one might:
- **Plot the Samples**: Visualizing the samples in a scatter plot can help illustrate the joint distribution and the correlation between XX and YY. 
- **Compute Summary Statistics**: Calculating the empirical means, variances, and correlation from the samples can offer insights into the accuracy and efficiency of the Gibbs Sampling process.
- **Assess Convergence**: Analyzing the trace plots for each variable or employing more sophisticated convergence diagnostics can ensure that the sampling process has stabilized and is providing reliable estimates from the desired distribution.

This implementation serves as a foundational example of Gibbs Sampling, illustrating the method's utility in sampling from complex joint distributions through conditional sampling strategies


## Usage
The `gibbs_sampling` function takes the means (`mu_x`, `mu_y`), standard deviations (`sigma_x`, `sigma_y`), correlation coefficient (`rho`), and the number of iterations (`n_iterations`) as inputs to generate samples from the joint distribution.

```python
samples = gibbs_sampling(mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, rho=0.5, n_iterations=10000)
```