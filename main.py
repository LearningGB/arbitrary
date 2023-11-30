import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class ScratchKMeans:
    def __init__(self, n_clusters, n_init, max_iter, tol, verbose=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centers = None
        self.labels = None

    def initialize_centers(self, X):
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def calculate_sse(self, X, centers, labels):
        sse = 0
        for k in range(len(centers)):
            cluster_points = X[labels == k]
            sse += np.sum(np.linalg.norm(cluster_points - centers[k]) ** 2)
        return sse

    def allocate_to_cluster(self, X, centers):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def move_centers(self, X, labels):
        new_centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centers[k] = np.mean(cluster_points, axis=0)
        return new_centers

    def kmeans(self, X):
        best_centers = None
        best_labels = None
        best_sse = np.inf

        for _ in range(self.n_init):
            centers = self.initialize_centers(X)
            for _ in range(self.max_iter):
                labels = self.allocate_to_cluster(X, centers)
                new_centers = self.move_centers(X, labels)

                if np.linalg.norm(new_centers - centers) < self.tol:
                    break

                centers = new_centers

            sse = self.calculate_sse(X, centers, labels)

            if self.verbose:
                print(f"Iteration {_ + 1}, SSE: {sse}")

            if sse < best_sse:
                best_sse = sse
                best_centers = centers
                best_labels = labels

        self.centers = best_centers
        self.labels = best_labels

    def fit(self, X):
        self.kmeans(X)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

# Artificial dataset for clustering
X, _ = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=0.5, shuffle=True, random_state=0)

# Create and fit the KMeans model
kmeans_model = ScratchKMeans(n_clusters=4, n_init=10, max_iter=100, tol=1e-4, verbose=True)
kmeans_model.fit(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans_model.labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans_model.centers[:, 0], kmeans_model.centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering')
plt.legend()
plt.show()
