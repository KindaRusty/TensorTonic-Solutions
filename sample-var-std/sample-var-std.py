import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.array(x)
    mean_of_x = np.mean(x)
    n = len(x) 
    
    variance = np.sum((x - mean_of_x) ** 2) / (n - 1)

    std = np.sqrt(variance)
    return variance, std