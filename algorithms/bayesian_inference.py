import pymc3 as pm
import numpy as np

class BayesianInference:
    def __init__(self):
        pass

    def perform_inference(self, data):
        with pm.Model() as model:
            # Prior distribution for mean and standard deviation
            mean = pm.Normal('mean', mu=0, sd=10)
            std_dev = pm.HalfNormal('std_dev', sd=10)
            
            # Likelihood of the data
            likelihood = pm.Normal('likelihood', mu=mean, sd=std_dev, observed=data)
            
            # Sample from the posterior distribution
            trace = pm.sample(1000, tune=1000)
        
        return trace

# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=100)
    
    # Perform Bayesian inference
    inference = BayesianInference()
    trace = inference.perform_inference(data)
    
    # Summary statistics of the posterior distribution
    print(pm.summary(trace))
