import numpy as np
from collections import Counter

def mean_median_mode(x): #x là dãy số
    """
    Compute mean, median, and mode.
    """
    mean = np.mean(x) #tính trung bình 
    median = np.median(x) #tính trung vị 
    #tính mốt
    freq = Counter(x) #tính tần suất 
    max_freq = max(freq.values()) #tìm kiếm giá trị có tần suất xuất hiện nhiều nhất
    modes = [key for key, value in freq.items() if value == max_freq]
    
    mode_value = min(modes) 
    
    return float(mean), float(median), float(mode_value)