from typing import Tuple
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt


def generate_synthetic_data(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic linear data with noise.

    Parameters:
    - n_points: Number of data points to generate.

    Returns:
    - Tuple of X (features) and Y (target) numpy arrays.
    """
    np.random.seed(42)
    X = np.linspace(0, 10, n_points)[:, None]  # 100 data points in the range [0, 10]
    true_slope = 2.5
    true_intercept = 0.5
    noise = np.random.randn(n_points) * 0.5
    Y = true_slope * X.squeeze() + true_intercept + noise  # Linear relationship with noise
    return X, Y


def build_bayesian_linear_model(X: np.ndarray, Y: np.ndarray) -> pm.Model:
    """Builds a Bayesian linear regression model with PyMC3.

    Parameters:
    - X: Feature matrix.
    - Y: Target vector.

    Returns:
    - A PyMC3 model object.
    """
    with pm.Model() as model:
        # Priors for unknown model parameters
        slope = pm.Normal('slope', mu=0, sigma=10)
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        mu = intercept + slope * X.squeeze()

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
    return model


def perform_variational_inference(model: pm.Model) -> pm.approximations.MeanField:
    """Performs variational inference on a given PyMC3 model.

    Parameters:
    - model: The PyMC3 model to perform inference on.

    Returns:
    - An approximation object from PyMC3.
    """
    with model:
        approx = pm.fit(method='advi')
    return approx


def plot_results(X: np.ndarray, Y: np.ndarray, trace: pm.backends.base.MultiTrace, n_samples: int = 100) -> None:
    """Plots the posterior predictive regression lines and original data.

    Parameters:
    - X: Feature matrix.
    - Y: Target vector.
    - trace: The trace object containing samples from the posterior.
    - n_samples: Number of samples to draw for plotting.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(X, Y, 'x', label='Data')
    pm.plot_posterior_predictive_glm(trace, samples=n_samples, eval=np.linspace(0, 10, 100),
                                     label='Posterior predictive regression lines',
                                     lm=lambda x, sample: sample['intercept'] + sample['slope'] * x)
    plt.title('Bayesian Linear Regression with Variational Inference')
    plt.legend()
    plt.show()


def main():
    X, Y = generate_synthetic_data()
    model = build_bayesian_linear_model(X, Y)
    approx = perform_variational_inference(model)
    trace = approx.sample(draws=1000)
    plot_results(X, Y, trace)


if __name__ == "__main__":
    main()
