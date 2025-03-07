import numpy as np

def linear_regression_cost_function(X, y, theta):
    """
    Compute the cost function for linear regression.

    Parameters:
    X : numpy array
        Input feature matrix of shape (m, n), where m is the number of samples and n is the number of features.
    y : numpy array
        Target values of shape (m, 1).
    theta : numpy array
        Parameters of the linear regression model of shape (n, 1).

    Returns:
    float
        The computed cost (MSE).
    """
    m = len(y)  # Number of training examples
    
    # Predictions
    predictions = X.dot(theta)
    
    # Calculate the squared errors
    squared_errors = (predictions - y) ** 2
    
    # Compute the cost (MSE)
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    
    return cost

# Example usage:
if __name__ == "__main__":
    # Example data
    X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])  # Feature matrix with a column of ones for the intercept term
    y = np.array([[7], [6], [5], [4]])  # Target values
    theta = np.array([[0.1], [0.2]])  # Initial parameters

    # Compute the cost
    cost = linear_regression_cost_function(X, y, theta)
    print(f"Cost: {cost}")