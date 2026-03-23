import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Tính tổng số lần dự đoán đúng (True Positives)
    tp_sum = np.sum(y_true == y_pred)
    
    # Tổng số lượng dự đoán
    total_samples = len(y_true)
    
    #Precision = Recall = Accuracy
    precision = tp_sum / total_samples
    recall = tp_sum / total_samples
    
    if (precision + recall) == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return float(f1)