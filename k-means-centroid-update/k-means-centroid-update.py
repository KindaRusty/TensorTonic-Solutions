import numpy as np
def k_means_centroid_update(points, assignments, k):
    points = np.array(points, dtype=float)
    assignments = np.array(assignments)
    centroids = np.zeros((k, points.shape[1]))
    
    for i in range(k):
        points_in_cluster = points[assignments == i]
        if len(points_in_cluster) > 0:
            centroids[i] = np.mean(points_in_cluster, axis=0)
        else:
            pass
    return centroids.tolist()