import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    mean = np.mean(x)
    median = np.median(x)
    
    freq = Counter(x)
    max_freq = max(freq.values())
    modes = [key for key, value in freq.items() if value == max_freq]
    
    mode_value = min(modes) 
    
    return float(mean), float(median), float(mode_value)