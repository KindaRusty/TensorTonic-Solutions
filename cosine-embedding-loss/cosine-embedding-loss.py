import math
#ko dùng numpy
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    #tính tích vô hướng (dot product)
    dot_product = sum(a*b for a,b in zip(x1, x2))
    #tích độ dài vector (norms)
    norm_x1 = math.sqrt(sum(a * a for a in x1))
    norm_x2 = math.sqrt(sum(b * b for b in x2))
    #tính cosine 
    cos = dot_product / (norm_x1 * norm_x2)
    if label == 1:
        return float(1-cos)
    else:
        return float(max(0, cos - margin))