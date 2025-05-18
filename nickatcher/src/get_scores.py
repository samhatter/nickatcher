import numpy as np
from sklearn.metrics import pairwise_kernels

def get_scores(user_embeddings_1, user_embeddings_2):
    X = user_embeddings_1.detach().cpu().numpy()
    Y = user_embeddings_2.detach().cpu().numpy()
    return mmd_rbf(X, Y)

def mmd_rbf(X, Y, gamma=1.0):
    """
    Calculates the squared MMD using an RBF (Gaussian) kernel.
    
    Args:
      X: A numpy array representing the first sample (n_samples1, n_features).
      Y: A numpy array representing the second sample (n_samples2, n_features).
      gamma: The kernel coefficient for RBF kernel.
    
    Returns:
      The squared MMD value.
    """

    k_xx = pairwise_kernels(X, X, metric='rbf', gamma=gamma)
    k_yy = pairwise_kernels(Y, Y, metric='rbf', gamma=gamma)
    k_xy = pairwise_kernels(X, Y, metric='rbf', gamma=gamma)
    
    return np.mean(k_xx) + np.mean(k_yy) - 2 * np.mean(k_xy)