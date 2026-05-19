import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x = np.array(x)
    mean_of_x = np.mean(x) #tính giá trị trung bình
    n = len(x) #Tính độ dài dãy số
    
    variance = np.sum((x - mean_of_x) ** 2) / (n - 1) #tính phương sai

    std = np.sqrt(variance) #tính sai lệch
    return variance, std #in kết quả