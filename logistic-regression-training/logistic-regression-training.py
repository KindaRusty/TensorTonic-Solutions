import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples, n_features = X.shape
    # 1. Initialization
    # w=[0,0...], b=0
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        # 2. Forward pass
        # z = Xw + b
        z = np.dot(X, w) + b
        # p = σ(z)
        p = _sigmoid(z)
        # 3. Compute gradients
        error = p - y
        # L/w = (1/n) * X^T * (p - y)
        dw = (1 / n_samples) * np.dot(X.T, error)
        # L/b = (1/n) * Σ(p - y)
        db = (1 / n_samples) * np.sum(error)        
        # 4. Update rules
        w -= lr * dw
        b -= lr * db
        
    return w, b