import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.metrics import pairwise_distances, davies_bouldin_score, calinski_harabasz_score
from pyswarm import pso

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# K-means clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
kmeans_labels = kmeans.fit_predict(X)

# Calculate centroid distances
centroid_distances = pairwise_distances(kmeans.cluster_centers_)

# Calculate average cluster density
def calculate_density(cluster_center, cluster_points, radius=0.5):
    num_points = len(cluster_points)
    num_points_within_radius = sum([1 for point in cluster_points if euclidean(point, cluster_center) <= radius])
    return num_points_within_radius / num_points

cluster_densities = [calculate_density(center, X[kmeans_labels == i]) for i, center in enumerate(kmeans.cluster_centers_)]

# Calculate deviation index (standard deviation of distances from points to cluster center)
deviation_index = np.mean([np.std(X[kmeans_labels == i], axis=0) for i in range(n_clusters)])

# Calculate cluster separation index
average_inter_cluster_distance = np.mean(centroid_distances)
average_intra_cluster_distance = np.mean([np.mean([centroid_distances[i][j] for j in range(n_clusters) if j != i]) for i in range(n_clusters)])
cluster_separation_index = average_inter_cluster_distance / average_intra_cluster_distance

# Define the optimization function
def optimization_function(weights):
    weight_density, weight_deviation, weight_separation = weights
    combined_validity_index = (weight_density * np.mean(cluster_densities) +
                               weight_deviation * deviation_index +
                               weight_separation * cluster_separation_index)
    return -combined_validity_index  # PSO minimizes the objective function, so we negate it

# Set bounds for the weights
lower_bounds = [0, 0, 0]
upper_bounds = [1, 1, 1]

# Perform PSO optimization
best_weights, _ = pso(optimization_function, lower_bounds, upper_bounds, swarmsize=10, maxiter=50)

# Calculate Davies-Bouldin Index and Calinski-Harabaz Index for KMeans results
db_index = davies_bouldin_score(X, kmeans_labels)
ch_index = calinski_harabasz_score(X, kmeans_labels)

print("Davies-Bouldin Index:", db_index)
print("Calinski-Harabaz Index:", ch_index)
print("Best Weights:", best_weights)
best_combined_validity_index = -optimization_function(best_weights)
print("Best Combined Validity Index:", best_combined_validity_index)


# Stopping search: maximum iterations reached --> 50
# Davies-Bouldin Index: 0.2905354431615246
# Calinski-Harabaz Index: 3386.414706995127
# Best Weights: [1. 1. 1.]
# Best Combined Validity Index: 1.8257422481486885







