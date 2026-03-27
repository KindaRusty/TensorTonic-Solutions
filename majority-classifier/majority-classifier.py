import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    if len(y_train) == 0:
        return np.array([], dtype=int)
    # 2. tìm các class duy nhất và return true
    values, counts = np.unique(y_train, return_counts=True)
    # 3. tìm index của số lần xuất hiện lớn nhất
    majority_index = np.argmax(counts)
    majority_class = values[majority_index]
    # 4. tạo mảng dự đoán
    num_predictions = len(X_test)
    predictions = np.full(num_predictions, majority_class, dtype=int)
    
    return predictions