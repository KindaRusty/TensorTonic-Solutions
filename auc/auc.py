import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    auc = 0.0
    for i in range(len(fpr) - 1):
        # Calculate width and height of each trapezoid
        width = fpr[i + 1] - fpr[i]
        height = (tpr[i] + tpr[i + 1]) / 2.0
        auc += width * height
    return auc
