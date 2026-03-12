import numpy as np
def k_means_centroid_update(points, assignments, k):
    points = np.array(points) #Có thể để dtype = float tùy yêu cầu chấm của từng bài khó hay dễ
    assignments = np.array(assignments)
    
    centroids = np.zeros((k, points.shape[1]))
    
    for i in range(k):
        points_in_cluster = points[assignments == i] #có điều kiện là phải có 1 mới pick up đến bước tiếp theo
        if len(points_in_cluster) > 0: #tránh chia cho 0
            centroids[i] = np.mean(points_in_cluster, axis=0) #mean là chia trung bình cộng các dãy đã lấy phía trên
        else:
            pass
    return centroids.tolist()