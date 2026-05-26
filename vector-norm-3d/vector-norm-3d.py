import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    #dùng asarray sẽ tiết kiệm bộ nhớ 1 chút nếu v đã là array sẵn rồi
    v = np.asarray(v)
    #np.linalg.norm với axis=-1 sẽ tự động trả về một số thực (float scalar) nếu đầu vào là vector 1D, và trả về mảng shape (N,) nếu đầu vào là 2D
    magnitude = np.linalg.norm(v, axis=-1) #axis=-1 sẽ tính toán chiều cuối cùng phụ thuộc vào input là vector đơn lẻ hay là 1 batch
    return magnitude