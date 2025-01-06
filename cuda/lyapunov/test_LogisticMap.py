import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, r):
    """Logistic map function."""
    return r * x * (1 - x)

def compute_lyapunov_exponent(r, x0, n_iterations, n_transient=100):
    """
    Compute the Lyapunov exponent for the logistic map.
    
    Parameters:
        r: The parameter of the logistic map.
        x0: Initial condition.
        n_iterations: Number of iterations for computing the exponent.
        n_transient: Number of transient iterations to discard.
    
    Returns:
        The Lyapunov exponent.
    """
    x = x0
    le_sum = 0.0  # Sum of the logarithms of the derivatives
    
    for i in range(n_transient + n_iterations):
        # Skip the transient phase
        if i >= n_transient:
            derivative = abs(r * (1 - 2 * x))  # Derivative of the logistic map
            if derivative == 0:
                derivative = 1e-8  # Avoid log(0)
            le_sum += np.log(derivative)
        
        # Update x using the logistic map
        x = logistic_map(x, r)
    
    # Average the sum of the logarithms to get the Lyapunov exponent
    return le_sum / n_iterations

# Parameters
r_values = np.linspace(2.5, 4.0, 500)  # Range of r values to explore
x0 = 0.5  # Initial condition
n_iterations = 1000  # Number of iterations to compute Lyapunov exponent
n_transient = 500  # Transient iterations to discard

# Compute Lyapunov exponents for the range of r values
lyapunov_exponents = [compute_lyapunov_exponent(r, x0, n_iterations, n_transient) for r in r_values]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(r_values, lyapunov_exponents, label="Lyapunov Exponent")
plt.axhline(0, color="red", linestyle="--", label="Zero Line")
plt.xlabel("r")
plt.ylabel("Lyapunov Exponent")
plt.title("Lyapunov Exponent for the Logistic Map")
plt.legend()
plt.grid()
plt.show()
