from sklearn.datasets import load_iris
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X = iris.data[:200, :2]  # Use the first two features and the first 200 samples
y = iris.target[:200]     # Use the first 200 target values
y = np.where(y == 0, -1, 1)  # Convert labels to {-1, 1}

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Class -1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')
plt.title('Iris Dataset Visualization')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.grid(True)
plt.show()