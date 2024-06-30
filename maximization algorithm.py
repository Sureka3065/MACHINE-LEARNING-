import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mean, cov):
    return (1.0 / np.sqrt(2 * np.pi * cov)) * np.exp(-0.5 * (x - mean)**2 / cov)

def initialize_parameters(X, n_clusters):
    np.random.seed(0)
    mean = np.random.choice(X.flatten(), n_clusters)
    cov = np.random.random_sample(size=n_clusters)
    pi = np.ones(n_clusters) / n_clusters
    return mean, cov, pi

def expectation_step(X, mean, cov, pi, n_clusters):
    likelihood = np.zeros((X.shape[0], n_clusters))
    for k in range(n_clusters):
        likelihood[:, k] = pi[k] * gaussian(X.flatten(), mean[k], cov[k])
    total_likelihood = np.sum(likelihood, axis=1, keepdims=True)
    responsibility = likelihood / total_likelihood
    return responsibility

def maximization_step(X, responsibility, n_clusters):
    N_k = np.sum(responsibility, axis=0)
    mean = np.sum(X * responsibility, axis=0) / N_k
    cov = np.zeros(n_clusters)
    for k in range(n_clusters):
        cov[k] = np.sum(responsibility[:, k] * (X.flatten() - mean[k])**2) / N_k[k]
    pi = N_k / X.shape[0]
    return mean, cov, pi

def log_likelihood(X, mean, cov, pi, n_clusters):
    likelihood = np.zeros((X.shape[0], n_clusters))
    for k in range(n_clusters):
        likelihood[:, k] = pi[k] * gaussian(X.flatten(), mean[k], cov[k])
    return np.sum(np.log(np.sum(likelihood, axis=1)))

def em_algorithm(X, n_clusters, n_iter):
    mean, cov, pi = initialize_parameters(X, n_clusters)
    log_likelihoods = []

    for _ in range(n_iter):
        responsibility = expectation_step(X, mean, cov, pi, n_clusters)
        mean, cov, pi = maximization_step(X, responsibility, n_clusters)
        log_likelihoods.append(log_likelihood(X, mean, cov, pi, n_clusters))

    return mean, cov, pi, log_likelihoods

# Generate synthetic data
np.random.seed(42)
X1 = np.random.normal(0, 1, 100)
X2 = np.random.normal(5, 1, 100)
X = np.hstack((X1, X2)).reshape(-1, 1)

# Run the EM algorithm
n_clusters = 2
n_iter = 100
mean, cov, pi, log_likelihoods = em_algorithm(X, n_clusters, n_iter)

# Plot the data and the resulting Gaussian distributions
plt.hist(X, bins=30, density=True, alpha=0.5, color='gray')
x_axis = np.linspace(min(X), max(X), 1000)
for k in range(n_clusters):
    plt.plot(x_axis, pi[k] * gaussian(x_axis, mean[k], cov[k]), label=f'Cluster {k+1}')
plt.title('Expectation-Maximization Gaussian Mixture Model')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot log-likelihoods
plt.plot(log_likelihoods)
plt.title('Log-Likelihood over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.show()

print("Final means:", mean)
print("Final covariances:", cov)
print("Final mixing coefficients:", pi)
