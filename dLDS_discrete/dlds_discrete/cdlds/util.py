import numpy as np
from sklearn.decomposition import NMF
import plotly.express as px
import plotly.graph_objects as go
import torch


def init_U(X1, X2, D_control, err=None):
    """Initialize the control input matrix U by applying NMF on the first iteration without control and take the residuals as U

    """
    if err is None:
        err = X2 - (X2 @ np.linalg.pinv(X1)) @ X1
    nnmf = NMF(n_components=D_control, init='random', random_state=0)
    W = nnmf.fit_transform(np.abs(err.T))
    W_max = W.max(axis=0, keepdims=True)
    U_0 = np.divide(W, W_max, where=W != 0, out=W)

    return U_0.T


def single_step(X, model):
    # predict the next time step
    X_hat = []
    with torch.no_grad():
        for i in range(X.shape[0]):
            y = model(X[i, :], i)
            X_hat.append(y)
    return X_hat


def multi_step(X, model):
    # predict all time steps from initial state
    X_hat = []
    X_hat.append(X[0, :].unsqueeze(0))
    with torch.no_grad():
        for i in range(X.shape[0]):
            y = model(X_hat[-1].squeeze(), i)
            X_hat.append(y)
    return X_hat


def plotting(X, plot_states=False, states=None, title=None, show=True):

    fig = go.Figure()

    lines = px.line(X)

    for line in lines.data:
        fig.add_trace(line)

    if plot_states:
        scaled_states = np.array(
            list(map(lambda x: (x * (X.max() - X.min()) + X.min()), states)))

        fig.add_trace(go.Scatter(
            x=np.arange(X.shape[0]),
            y=scaled_states,
            mode='lines',
            line=dict(color='black', width=2),
            name='state'
        ))
    fig.update_layout(title=title)
    if show:
        fig.show()
    return fig
