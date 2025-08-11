import numpy as np
import matplotlib.pyplot as plt

# Set up synthetic sugnals
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Create 3 independent signals
s1 = np.sin(2 * time)                # Sinusoidal
s2 = np.sign(np.sin(3 * time))       # Square wave
s3 = np.random.normal(size=n_samples) # Gaussian noise

S = np.c_[s1, s2, s3]

# Mix signals with a random mixing matrix
A = np.random.rand(3, 3)
X = S @ A.T  # Mixed signals

# Plot mixed signals
plt.figure(figsize=(8, 5))
plt.title("Mixed Signals")
for i in range(3):
    plt.plot(X[:, i])
plt.show()

# center and whiten the data now
def center(X):
    # Subtract mean from each column
    return X - X.mean(axis=0)

def whiten(X):
    # Whiten the data so covariance is the identity
    cov = np.cov(X, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    D_inv = np.diag(1. / np.sqrt(eig_vals))
    X_whitened = (eig_vecs @ D_inv @ eig_vecs.T) @ X.T
    return X_whitened.T

X_centered = center(X)
X_whitened = whiten(X_centered)

# ICA using deflation & kurtosis maximization
def kurtosis(x):
    # Compute kurtosis for non-Gaussianity
    return np.mean(x**4) - 3*(np.mean(x**2)**2)

def fastica_one_component(X, tol=1e-5, max_iter=1000):
    # Extract one independent component
    n_features = X.shape[1]
    w = np.random.rand(n_features)
    w /= np.linalg.norm(w)
    
    for _ in range(max_iter):
        # Non-Gaussianity maximization (cubic nonlinearity)
        w_new = np.mean(X * ((X @ w) ** 3)[:, None], axis=0) - 3 * w
        w_new /= np.linalg.norm(w_new)
        
        # Convergence check
        if np.abs(np.abs(np.dot(w, w_new)) - 1) < tol:
            break
        w = w_new
    return w

def fastica(X, n_components):
    # FastICA using deflation
    W = np.zeros((n_components, X.shape[1]))
    for i in range(n_components):
        w = fastica_one_component(X)
        # Deflation: make orthogonal to previous components
        for j in range(i):
            w -= np.dot(w, W[j]) * W[j]
        w /= np.linalg.norm(w)
        W[i, :] = w
    return W

#Run ICA
W = fastica(X_whitened, n_components=3)
S_estimated = X_whitened @ W.T  # Recovered signals

plt.figure(figsize=(10, 7))

plt.subplot(3, 1, 1)
plt.title("True Source Signals")
for i in range(3):
    plt.plot(S[:, i])

plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
for i in range(3):
    plt.plot(X[:, i])

plt.subplot(3, 1, 3)
plt.title("Recovered Independent Components")
for i in range(3):
    plt.plot(S_estimated[:, i])

plt.tight_layout()
plt.show()
