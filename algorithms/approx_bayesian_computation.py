import numpy as np


def generate_data(params):
    """Generate synthetic data given parameters."""
    mean, std = params
    return np.random.normal(mean, std, size=100)


def distance_metric(observed_data, simulated_data):
    """Calculate distance between observed and simulated data."""
    return np.abs(np.mean(observed_data) - np.mean(simulated_data))


def abc_inference(observed_data, prior_sampler, distance_metric, threshold, num_samples):
    """
    Perform ABC to infer parameters.
    
    Parameters:
    - observed_data: the observed data
    - prior_sampler: function to sample from the prior distribution of parameters
    - distance_metric: function to calculate the distance between observed and simulated data
    - threshold: acceptance threshold for the distance
    - num_samples: number of accepted samples to generate
    """
    accepted_params = []
    
    while len(accepted_params) < num_samples:
        # Sample parameters from the prior
        params = prior_sampler()
        
        # Generate simulated data using the sampled parameters
        simulated_data = generate_data(params)
        
        # Calculate the distance between observed and simulated data
        distance = distance_metric(observed_data, simulated_data)
        
        # Accept the parameters if the distance is below the threshold
        if distance < threshold:
            accepted_params.append(params)
    
    return np.array(accepted_params)


# Example usage
if __name__ == "__main__":
    # Observed data (for example purposes, generating synthetic observed data)
    true_params = [5.0, 2.0] # True mean and standard deviation
    observed_data = generate_data(true_params)
    
    # Define a prior sampler for the parameters
    def prior_sampler():
        """
        Sample from the prior distribution of the parameters.
        For simplicity, we assume uniform priors for the mean and standard deviation."""
        return [np.random.uniform(0, 10), # Prior for mean
                np.random.uniform(1, 5)]  # Prior for standard deviation

    # Perform ABC inference
    accepted_params = abc_inference(
        observed_data=observed_data,
        prior_sampler=prior_sampler,
        distance_metric=distance_metric,
        threshold=0.1, # Example threshold, adjust based on your data and model
        num_samples=100 # Number of accepted samples to generate
    )
    
    print(f"Accepted parameters (mean, std): {accepted_params.mean(axis=0)}")
