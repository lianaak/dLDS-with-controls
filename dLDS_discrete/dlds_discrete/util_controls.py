import numpy as np
from sklearn.decomposition import NMF


def init_U(X1, X2, D_control):
    """Initialize the control input matrix U by applying NMF on the first iteration without control and take the residuals as U

    """

    err = X2 - (X2 @ np.linalg.pinv(X1)) @ X1
    nnmf = NMF(n_components=D_control, init='random', random_state=0, )
    W = nnmf.fit_transform(np.abs(err.T))
    W_max = W.max(axis=0, keepdims=True)
    U_0 = np.divide(W, W_max, where=W != 0, out=W)
    return U_0.T
