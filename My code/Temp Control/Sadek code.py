import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_x = 50
Q = pd.read_excel('optPolicy_50.xlsx', header=None)
Q_array = Q.to_numpy()

# Compute the state value function by taking the max across the action dimension (axis=1).
state_values = np.max(Q_array, axis=1).reshape((n_x, n_x, n_x))

# Create grid for visualization
x = np.linspace(0, n_x, n_x)
y = np.linspace(0, n_x, n_x)
X, Y = np.meshgrid(x, y)

# Number of subplots per row/column (adjust depending on the number of theta values)
subplot_cols = 6  # Number of columns in the subplot grid
subplot_rows = int(np.ceil(n_x / subplot_cols))  # Calculate the number of rows needed

# Create a figure for subplots
fig = plt.figure(figsize=(15, 15))  # Adjust the figure size as needed

# Plot each theta_idx in a separate subplot
for theta_idx in range(n_x-1):
    Z = state_values[:, :, theta_idx]  # Fixing theta at a certain index

    ax = fig.add_subplot(subplot_rows, subplot_cols, theta_idx + 1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('State Value')
    ax.set_title(f'Theta Index {theta_idx + 1}')

# Adjust layout for better display
plt.tight_layout()

# Show the figure with all subplots
plt.show()