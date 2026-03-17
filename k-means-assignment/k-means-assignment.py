def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    assignments = []
    #duyệt từng điểm 
    for p in points:
        best_dist = float('inf')
        best_idx = 0 
        #so sánh p với từng centroid
        for idx, c in enumerate(centroids):
            #tính bình phương khoảng cách euclidean
            current_dist = 0 
            for d in range(len(p)):
                current_dist += (p[d] - c[d]) ** 2
            #cập nhật nếu gần centroid hơn
            #dùng dấu < để ưu tiên index nhỏ hơn nếu cùng khoảng cách
            if current_dist < best_dist:
                best_dist = current_dist
                best_idx = idx
        
        assignments.append(best_idx)
        
    return assignments