import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    def get_node_gini(y):
        y = np.array(y)
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.size
        return 1.0 - np.sum(probs**2)

    n_l, n_r = len(y_left), len(y_right)
    n_total = n_l + n_r
    
    if n_total == 0:
        return 0.0
    return (n_l / n_total) * get_node_gini(y_left) + (n_r / n_total) * get_node_gini(y_right)