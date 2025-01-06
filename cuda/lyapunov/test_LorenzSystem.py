import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz_system(t, state, sigma, rho, beta):
    """Lorenz system equations."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def compute_lyapunov_lorenz_fixed(sigma, rho, beta, dt=0.01, t_max=50, n_transient=30):
    """Compute Lyapunov Exponent for the Lorenz system."""
    state1 = np.array([1.0, 1.0, 1.0])  # Initial condition for state1
    state2 = state1 + 1e-6  # Slightly perturbed initial condition for state2
    t = 0
    time_points = []
    lyapunov_exponents = []

    # To store the x, y, z values for both states
    x1_vals, y1_vals, z1_vals = [], [], []
    x2_vals, y2_vals, z2_vals = [], [], []

    total_lyapunov = 0
    num_steps = 0

    while t < t_max:
        # Integrate both trajectories
        sol1 = solve_ivp(lorenz_system, [t, t + dt], state1, args=(sigma, rho, beta), method='RK45')
        sol2 = solve_ivp(lorenz_system, [t, t + dt], state2, args=(sigma, rho, beta), method='RK45')
        
        # Update the states
        state1 = sol1.y[:, -1]
        state2 = sol2.y[:, -1]
        
        # Store the x, y, z values
        x1_vals.append(state1[0])
        y1_vals.append(state1[1])
        z1_vals.append(state1[2])
        
        x2_vals.append(state2[0])
        y2_vals.append(state2[1])
        z2_vals.append(state2[2])
        
        # Compute separation
        delta = np.linalg.norm(state2 - state1)
        
        if t > n_transient:
            # Calculate the logarithmic separation
            lyapunov_exponent = np.log(delta) / dt
            total_lyapunov += lyapunov_exponent
            num_steps += 1
            lyapunov_exponents.append(total_lyapunov / num_steps)
        
        # Renormalize the separation
        state2 = state1 + (state2 - state1) / delta
        
        # Store time
        time_points.append(t)
        t += dt
    
    return np.array(time_points), np.array(lyapunov_exponents), x1_vals, y1_vals, z1_vals, x2_vals, y2_vals, z2_vals

# Parameters for Lorenz System
sigma, rho, beta = 10, 28, 8 / 3
dt = 0.01
t_max = 50
n_transient = 30

# Compute Lyapunov Exponent and get state trajectories
time, lyapunov_exponents, x1_vals, y1_vals, z1_vals, x2_vals, y2_vals, z2_vals = compute_lyapunov_lorenz_fixed(sigma, rho, beta, dt, t_max, n_transient)

# Plot Lyapunov exponent convergence
plt.figure(figsize=(8, 5))
plt.plot(time[len(time) - len(lyapunov_exponents):], lyapunov_exponents, label="Lyapunov Exponent")
plt.axhline(0, color="red", linestyle="--", label="Zero Line")
plt.xlabel("Time")
plt.ylabel("Lyapunov Exponent")
plt.title("Lyapunov Exponent Convergence (Improved Stability)")
plt.legend()
plt.grid()
plt.show()

# Plot the trajectories of x, y, z for both states in the same graphs
plt.figure(figsize=(14, 8))

# Plot x1 and x2 on the same graph
plt.subplot(3, 1, 1)
plt.plot(time[n_transient:], x1_vals[n_transient:], label='x1')
plt.plot(time[n_transient:], x2_vals[n_transient:], label='x2')
plt.title("x1 vs x2")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Plot y1 and y2 on the same graph
plt.subplot(3, 1, 2)
plt.plot(time[n_transient:], y1_vals[n_transient:], label='y1')
plt.plot(time[n_transient:], y2_vals[n_transient:], label='y2')
plt.title("y1 vs y2")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Plot z1 and z2 on the same graph
plt.subplot(3, 1, 3)
plt.plot(time[n_transient:], z1_vals[n_transient:], label='z1')
plt.plot(time[n_transient:], z2_vals[n_transient:], label='z2')
plt.title("z1 vs z2")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
 