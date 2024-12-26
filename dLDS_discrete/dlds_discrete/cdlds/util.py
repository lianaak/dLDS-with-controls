from os import name
import select
import numpy as np
from sklearn.decomposition import NMF
import plotly.express as px
import plotly.graph_objects as go
import torch

def frames_to_time(df):
    df['time'] = df.index/3.26
    df.index = df['time']
    df.drop(columns=['time'], inplace=True)
    return df


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


def find_highlight_regions(states):
    regions = []
    start = None
    for i in range(len(states)):
        if states[i] == 1 and start is None:
            start = i
        elif states[i] == 0 and start is not None:
            regions.append((start, i - 1))
            start = None
    # If the states array ends with a 1, capture that region as well
    if start is not None:
        regions.append((start, len(states) - 1))
    return regions


def single_step(X, model):
    # predict the next time step
    X_hat = []
    with torch.no_grad():
        for i in range(X.shape[0]):
            y = model(X[i, :].unsqueeze(0), i)
            X_hat.append(y)
    return X_hat


def multi_step(X, model, lookback=1):
    # predict all time steps from initial state
    X_hat = []
    X_hat.append(X[:lookback, :])
    with torch.no_grad():
        for i in range(X.shape[0]):
            # y = model(X_hat[-1].squeeze(), i)
            # print(X_hat[-lookback:][1].shape)
            # print(torch.tensor(X_hat[-lookback:]).shape)
            y = model(
                X_hat[-lookback:][0], np.arange(i+lookback))
            X_hat.append(y)
    return X_hat


def plotting(X, plot_states=False, states=None, title=None, show=False, stack_plots=False):

    fig = go.Figure()

    # if we want to stack multiple data, make sure that additional data is plotted in the same color but with different line style
    if stack_plots:
        for i, data in enumerate(X):
            if i == 0:
                name = 'true'
            else:
                name = f'predicted'
            lines = px.line(data)
            for line in lines.data:
                no = int(line.name)+1
                line.name = f'{no}_{name}'
                fig.add_trace(line)
                fig.update_traces(
                    line=dict(dash=('dash' if i > 0 else 'solid')), selector=dict(name=line.name))

    else:
        lines = px.line(X)
        for line in lines.data:
            fig.add_trace(line)

    if plot_states:
        if stack_plots:
            X = np.hstack(X)
        """ scaled_states = np.array(
            list(map(lambda x: (x * (X.max() - X.min()) + X.min()), states)))

        fig.add_trace(go.Scatter(
            x=np.arange(X.shape[0]),
            y=scaled_states,
            mode='lines',
            line=dict(color='black', width=2),
            name='state'
        )) """

        # Identify the regions where states are 1
        highlight_regions = find_highlight_regions(states)

        # Add shaded regions (rectangles) to the plot for each highlight region
        for region in highlight_regions:
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=region[0], x1=region[1],
                y0=0, y1=1,
                fillcolor="lightyellow", opacity=1, line_width=0,
                layer="below"  # Place the shapes in the background
            )
    fig.update_layout(title=title)
    if show:
        fig.show()
    return fig
