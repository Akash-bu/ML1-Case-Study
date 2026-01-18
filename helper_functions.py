import numpy as np 

def zscore_fit(X):
    """
    X: shape (n_samples, n_features)
    Returns: mean, std   
    """

    mean = np.mean(X, axis=0) 
    std = np.std(X, axis=0)
    std[std == 0] = 1.0 
    return mean, std 

def zscore_transform(X, mean, std):

    return (X - mean) / std 

def zscore_fit_transform(X):

    mean, std = zscore_fit(X) 
    return zscore_transform(X, mean, std), mean, std      
