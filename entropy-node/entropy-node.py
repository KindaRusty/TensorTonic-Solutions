import numpy as np

def entropy_node(y):
    y = np.array(y)
    if len(y) == 0:
        return 0.0
    
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    
    nonzero = probs[probs > 0]
    
    h = -np.sum(nonzero * np.log2(nonzero))
    
    return float(h)