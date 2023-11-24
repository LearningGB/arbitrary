import numpy as np

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

def kmeans(X, K, max_iter=100):
    """
    K-means clustering algorithm.

    Args:
        X (np.ndarray): Input data.
        K (int): Number of clusters.
        max_iter (int): Maximum number of iterations.

    Returns:
        np.ndarray: Cluster labels.
    """
    centroids = initialize_centroids(X, K)
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            new_centroids[k] = np.mean(X[labels == k], axis=0)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels

X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
K = 2
labels = kmeans(X, K)
print(labels)
