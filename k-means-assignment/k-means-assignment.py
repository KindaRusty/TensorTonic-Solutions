def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    assignments = []
    for p in points:
        best_dist = float('inf')
        best_idx = 0 
        
        for idx, c in enumerate(centroids):
            current_dist = 0
            for d in range(len(p)):
                current_dist += (p[d] - c[d]) ** 2
            if current_dist < best_dist:
                best_dist = current_dist
                best_idx = idx
        
        assignments.append(best_idx)
        
    return assignments