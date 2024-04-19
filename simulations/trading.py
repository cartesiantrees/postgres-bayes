import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data_size = 100  # Number of months
data = pd.DataFrame({
    'market_index_return': np.random.normal(0.05, 0.1, data_size),  # Monthly returns (mean 5%, sd 10%)
    'interest_rate': np.random.normal(0.01, 0.01, data_size),  # Interest rates (mean 1%, sd 1%)
    'stock_return': np.random.normal(0.06, 0.15, data_size)  # Stock returns to predict (mean 6%, sd 15%)
})

# Bayesian Linear Regression Model with Variational Inference in PyMC3
with pm.Model() as model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta_market = pm.Normal('beta_market', mu=0, sigma=1)
    beta_interest = pm.Normal('beta_interest', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected outcome formula
    expected_return = alpha + beta_market * data['market_index_return'] + beta_interest * data['interest_rate']

    # Likelihood (sampling distribution) of observations
    stock_return_obs = pm.Normal('stock_return_obs', mu=expected_return, sigma=sigma, observed=data['stock_return'])

    # Approximate inference with ADVI (Automatic Differentiation Variational Inference)
    approx = pm.fit(n=30000, method='advi')

# Draw samples from the variational approximation
trace = approx.sample(draws=5000)

# Plotting the results
az.plot_trace(trace)
plt.show()

# Model summary
print(az.summary(trace, round_to=2))

# Posterior Predictive Checks
with model:
    ppc = pm.sample_posterior_predictive(trace, var_names=['alpha', 'beta_market', 'beta_interest', 'sigma'],
                                         samples=500)
    az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model))
    plt.show()
