import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    n_samples = X_train.shape[0]
    classes = np.unique(y_train)
    model = {}
    
    # Huấn luyện: Tìm prior, mean và variance cho từng class
    for c in classes:
        X_c = X_train[y_train == c]
        prior = len(X_c) / n_samples
        mean = np.mean(X_c, axis=0)
        # Tính phương sai quần thể và cộng epsilon 1e-9
        var = np.var(X_c, axis=0) + 1e-9 
        model[c] = (np.log(prior), mean, var)
    predictions = []
    # Dự đoán: Tính log posterior cho từng điểm dữ liệu test
    for x in X_test:
        posteriors = {}
        for c in classes:
            log_prior, mean, var = model[c]
            # Tính tổng log-likelihood của các đặc trưng (features)
            log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var))
            # Tính log posterior
            posteriors[c] = log_prior + log_likelihood
        # Chọn class có xác suất cao nhất
        predictions.append(max(posteriors, key=posteriors.get))
        
    return predictions