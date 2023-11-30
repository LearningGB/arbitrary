import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
def initialize_centroids(X, K):
    """
    Randomly select K initial centroids from the data matrix X.

    This function solves Problem 1. It randomly selects K initial centroids from the data matrix X.
    """
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        idx = np.random.randint(0, X.shape[0])
        centroids[k, :] = X[idx, :]
    return centroids

def calculate_distances(X, mu):
    """
    Calculates the distances between each data point and each centroid.

    This function solves Problem 3. It calculates the distances between each data point and each centroid.
    """
    distances = np.linalg.norm(X[:, :, None] - mu[None, :, :], axis=2)
    return distances

def assign_clusters(distances):
    """
    Assigns each data point to the closest centroid.

    This function solves Problem 3. It assigns each data point to the closest centroid.
    """
    clusters = np.argmin(distances, axis=1)
    return clusters

def update_centroids(X, clusters, r):
    """
    Updates the centroids to the means of the data points assigned to each cluster.

    This function solves Problem 4. It updates the centroids to the means of the data points assigned to each cluster.
    """
    mu = np.zeros((r.shape[1], X.shape[1]))
    for k in range(r.shape[1]):
        Xk = X[clusters == k, :]
        if len(Xk) > 0:
            mu[k, :] = np.mean(Xk, axis=0)
    return mu
"problem 6"
def kmeans(X, K, max_iter=100, tol=1e-4, n_init=10):
    """
    Performs k-means clustering with multiple initializations.

    Args:
        X (np.ndarray): The data matrix of shape (N, D).
        K (int): The number of clusters.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for convergence.
        n_init (int): The number of initializations.

    Returns:
        np.ndarray: The cluster assignments of shape (N,).
        np.ndarray: The centroids of shape (K, D).
    """
    best_sse = np.inf
    best_mu = None
    best_clusters = None

    for _ in range(n_init):
        mu = initialize_centroids(X, K)
        for _ in range(max_iter):
            distances = calculate_distances(X, mu)
            clusters = assign_clusters(distances)
            r = np.zeros((X.shape[0], K))
            for i in range(X.shape[0]):
                r[i, clusters[i]] = 1
            mu = update_centroids(X, clusters, r)

            # Check for convergence
            if np.linalg.norm(mu - update_centroids(X, clusters, r)) < tol:
                break

        sse = calculate_sse(X, r, mu)
        if sse < best_sse:
            best_sse = sse
            best_mu = mu
            best_clusters = clusters

    return best_clusters, best_mu
"problem 7 For the data point $ X_n $ and the center point $ \ mu_k $ determined by learning, select $ r_ {nk} $ that minimizes $ SSE $."
"Assign the data point $ X_n $ to the nearest $ \ mu_k $."

def calculate_sse(X, r, mu):
    """
    Calculates the sum of squared errors (SSE).

    Args:
        X (np.ndarray): The data matrix of shape (N, D).
        r (np.ndarray): The cluster assignment matrix of shape (N, K).
        mu (np.ndarray): The centroids of shape (K, D).

    Returns:
        float: The sum of squared errors.
    """
    sse = 0
    for k in range(r.shape[1]):
        Xk = X[r[:, k] == 1, :]
        if len(Xk) > 0:
            sse += np.sum((Xk - mu[k, :])**2)
    return sse

def elbow_method(X, K_range):

    """
    Performs the elbow method to determine the optimal number of clusters.

    Args:
        X (np.ndarray): The data matrix of shape (N, D).
        K_range (list): The range of K values to consider.

    Returns:
        int: The optimal number of clusters.
    """
    sse_values = []
    for K in K_range:
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(X)
        r = kmeans.predict(X)
        sse = calculate_sse(X, r, kmeans.cluster_centers_)
        sse_values.append(sse)

    # Plot the elbow graph
    plt.plot(K_range, sse_values)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Sum of squared errors (SSE)')
    plt.title('Elbow method')
    plt.show()

    # Find the elbow point
    elbow_point = 0
    min_sse = np.inf
    for i, sse in enumerate(sse_values):
        if sse < min_sse:
            min_sse = sse
            elbow_point = i

    return K_range[elbow_point]

def main():
    # Load the data
    data = np.loadtxt('Wholesale customers data.csv', delimiter=',', skiprows=1)

    # Perform the elbow method
    K_range = range(1, 11)
    optimal_K = elbow_method(data[:, 4:], K_range)
    print('Optimal number of clusters:', optimal_K)

