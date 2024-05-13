"""
    1. Synthetic Data Generation: We create synthetic stock price data that fluctuates between
        two regimes: stable and volatile. Each regime has its own mean and standard deviation for price changes.
    2. HMM Training: We define and train a Gaussian Hidden Markov Model with two states
        (stable and volatile) using the hmmlearn library.
    3. State Prediction: After training, we use the model to predict the hidden states of the
        stock prices, which correspond to the different regimes.
    4. Visualization: The stock prices and the predicted hidden states are plotted. The hidden states are
        scaled and shifted for clear visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic stock price data
def generate_stock_data(n_points, regimes):
    data = []
    current_price = 100
    for _ in range(n_points):
        regime = np.random.choice(regimes)
        price_change = np.random.normal(regime['mean'], regime['std'])
        current_price += price_change
        data.append(current_price)
    return np.array(data)

# Define regimes: stable and volatile
regimes = [
    {'mean': 0.5, 'std': 1.0},  # Stable regime
    {'mean': 0.0, 'std': 2.0},  # Volatile regime
]

# Generate synthetic data
n_points = 300
stock_prices = generate_stock_data(n_points, regimes)

# Prepare the data for HMM (reshape to a 2D array with one feature)
stock_prices = stock_prices.reshape(-1, 1)

# Define and train the Hidden Markov Model
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
model.fit(stock_prices)

# Predict the hidden states (regimes) for the stock prices
hidden_states = model.predict(stock_prices)

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(stock_prices, label='Stock Prices', lw=2)
plt.plot(hidden_states * 10 + 90, label='Hidden States (Regimes)', lw=2)
plt.title('Stock Prices and Hidden States')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Print out the means and variances of the hidden states
print("Means of each hidden state:", model.means_.flatten())
print("Variances of each hidden state:", model.covars_.flatten())
