from typing import Tuple
import numpy as np


class IncrementalNaiveBayes:
    """Incremental Naive Bayes classifier for binary classification with binary features."""

    def __init__(self, n_features: int) -> None:
        """
        Initializes the classifier with the given number of features.

        Parameters:
        - n_features: int, number of features in the dataset.
        """
        self.n_features = n_features
        self.class_counts = np.zeros(2, dtype=np.float64)
        self.feature_counts = np.zeros((2, n_features), dtype=np.float64)
        self.total_samples = 0

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Incrementally updates the model with a new batch of data.

        Parameters:
        - X: np.ndarray, 2D array of shape (n_samples, n_features), new data points.
        - y: np.ndarray, 1D array of shape (n_samples,), class labels for the new data.
        """
        for x, label in zip(X, y):
            self.class_counts[label] += 1
            self.feature_counts[label] += x
            self.total_samples += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given data.

        Parameters:
        - X: np.ndarray, 2D array of shape (n_samples, n_features), data to predict.

        Returns:
        - np.ndarray, 1D array of predicted class labels.
        """
        # Avoid division by zero and log(0) by adding 1 to counts (Laplace smoothing)
        smoothed_class_counts = self.class_counts + 1
        smoothed_feature_counts = self.feature_counts + 1
        smoothed_total = 2 * (self.n_features + 1)

        # Calculate log probabilities of features given class
        feature_log_prob = np.log(smoothed_feature_counts / smoothed_class_counts[:, None])
        # Calculate log probabilities of each class
        class_log_prior = np.log(smoothed_class_counts / (self.total_samples + 2))

        # Calculate log probability of each class for each sample
        log_prob = X.dot(feature_log_prob.T) + class_log_prior
        # Return the class with the highest log probability
        return np.argmax(log_prob, axis=1)


# Example usage
if __name__ == "__main__":
    n_features = 10  # Example number of features
    model = IncrementalNaiveBayes(n_features=n_features)

    # Example: Update model with new batches of data
    for _ in range(5):  # Assume 5 batches of new data
        X_new = np.random.randint(2, size=(10, n_features))
        y_new = np.random.randint(2, size=10)
        model.update(X_new, y_new)

    # Example: Predict class labels for new data
    X_test = np.random.randint(2, size=(5, n_features))
    predictions = model.predict(X_test)
    print("Predicted class labels:", predictions)
