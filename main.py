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
    Randomly select K data points from X as initial centroids.

    Args:
        X (np.ndarray): Input data.
        K (int): Number of clusters.

    Returns:
        np.ndarray: Initial centroids.
    """
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    return centroids

def sse(X, labels, centroids):
    """
    Calculate the sum of squared errors (SSE) for a given clustering.

    Args:
        X (np.ndarray): Input data.
        labels (np.ndarray): Cluster labels.
        centroids (np.ndarray): Cluster centroids.

    Returns:
        float: SSE.
    """
    sse = 0
    for k in range(centroids.shape[0]):
        sse += np.sum((X[labels == k] - centroids[k])**2)

    return sse

def cluster_allocation(X, centroids):
    """
    Assign each data point in X to the nearest centroid in centroids.

    Args:
        X (np.ndarray): Input data.
        centroids (np.ndarray): Cluster centroids.

    Returns:
        np.ndarray: Cluster labels.
    """
    labels = np.argmin(np.linalg.norm(X[:, None] - centroids[None, :], axis=2), axis=1)
    return labels
def move_centroids(X, labels, K):
    """
    Update the centroids based on the current cluster assignments.

    Args:
        X (np.ndarray): Input data.
        labels (np.ndarray): Cluster labels.
        K (int): Number of clusters.

    Returns:
        np.ndarray: Updated centroids.
    """
    centroids = np.zeros_like(centroids)
    for k in range(K):
        centroids[k] = np.mean(X[labels == k], axis=0)

    return centroids
def kmeans(X, K, max_iter=100, tol=1e-4):
    """
    K-means clustering algorithm.

    Args:
        X (np.ndarray): Input data.
        K (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        np.ndarray: Cluster labels.
        np.ndarray: Final centroids.
        float: SSE.
    """
    centroids = initialize_centroids(X, K)
    for _ in range(max_iter):
        labels = cluster_allocation(X, centroids)
        new_centroids = move_centroids(X, labels, K)

        if np.allclose(centroids, new_centroids, tol=tol):
            break

        centroids = new_centroids

    sse = sse(X, labels, centroids)
    return labels, centroids, sse

def kmeans_multiple_runs(X, K, n_init=10):
    """
    Run k-means clustering multiple times with different initial values.

    Args:
        X (np.ndarray): Input data.
        K (int): Number of clusters.
        n_init (int): Number of initializations.

    Returns:
        np.ndarray: Best cluster labels.
        np.ndarray: Best centroids.
        float: Best SSE.
    """
    best_labels = None
    best_centroids = None
    best_sse = np.inf

    for _ in range(n_init):
        labels, centroids, sse = kmeans(X, K)

        if sse < best_sse:
            best_labels = labels
            best_centroids = centroids
            best_sse = sse

    return best_labels, best_centroids, best_sse

def estimate_cluster(X, cent
