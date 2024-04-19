import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Enhanced Data Generation
np.random.seed(42)
data_size = 300  # Increased dataset size for better model estimation
data = pd.DataFrame({
    'usage_frequency': np.random.poisson(10, data_size),
    'customer_support_calls': np.random.poisson(2, data_size),
    'marketing_engagement': np.random.binomial(5, 0.6, data_size) / 5,
    'will_renew': np.random.binomial(1, 0.75, data_size)
})

# Interaction term
data['interaction'] = data['usage_frequency'] * data['marketing_engagement']

# Bayesian Model in PyMC3
with pm.Model() as model:
    # More sophisticated priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    betas = pm.Normal('betas', mu=0, sigma=10, shape=4)  # Multiple predictors

    # Expected mean of outcome using logistic link function
    mu = alpha + betas[0] * data['usage_frequency'] + betas[1] * data['customer_support_calls'] \
         + betas[2] * data['marketing_engagement'] + betas[3] * data['interaction']

    theta = pm.Deterministic('theta', pm.math.sigmoid(mu))  # Probability of renewal

    # Likelihood
    Y_obs = pm.Bernoulli('Y_obs', p=theta, observed=data['will_renew'])

    # Sample from posterior
    trace = pm.sample(2000, return_inferencedata=False, tune=1000)

# Diagnostics and Visualization
az.plot_trace(trace)
plt.show()

# Summary of the model
model_summary = az.summary(trace, round_to=2)
print(model_summary)

# Posterior Predictive Checks
with model:
    ppc = pm.sample_posterior_predictive(trace, samples=500)
    az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model))
    plt.show()
