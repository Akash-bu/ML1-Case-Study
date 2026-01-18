import numpy as np 
import matplotlib.pyplot as plt

def zscore_fit(X):
    """
    X: shape (n_samples, n_features)
    Returns: mean, std   
    """

    mean = np.mean(X, axis=0) 
    std = np.std(X, axis=0)
    std[std == 0] = 1.0 
    return mean, std 

def zscore_transform(X, mean, std, clip = None):

    Z = (X - mean) / std 
    if clip is not None:
        Z = np.clip(Z, -clip, clip)
    return Z

def zscore_fit_transform(X):

    mean, std = zscore_fit(X) 
    return zscore_transform(X, mean, std), mean, std      

def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

import matplotlib.pyplot as plt

def plot_loss_curve(losses, title="Training Loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(losses, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
