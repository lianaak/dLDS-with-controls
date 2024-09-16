
import argparse
from re import U
import stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import wandb
from datasets import CdLDSDataGenerator
from models import DLDSwithControl
import util
import slim
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
import seaborn as sns



class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor


class DLDSModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_points, num_subdyn, lookback=50, control_size=1, sparsity_threshold=0.1):
        super().__init__()

        self.control_size = control_size

        self.coeffs = torch.nn.Parameter(torch.tensor(
            np.random.rand(num_subdyn, time_points), requires_grad=True))

        self.F = torch.nn.ParameterList()

        # Control matrix B (learnable)
        # self.B = torch.nn.Parameter(
        #    torch.randn(input_size, control_size, requires_grad=True) * 0.01)

        self.B = slim.linear.SpectralLinear(
            control_size, input_size, bias=False, sigma_min=0.0, sigma_max=1.0)

        # Learnable control input U (initialize with small values)
        # U will have the shape: (batch_size, seq_len, control_size)
        self.U = torch.nn.Parameter(torch.randn(
            control_size, time_points, requires_grad=True))

        # Store sparsity weight for L1 regularization
        self.sparsity_threshold = sparsity_threshold

        self.sparsity_pattern = torch.zeros_like(self.U, dtype=torch.bool)

        for _ in range(num_subdyn):
            f_i = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True, bias=True, bidirectional=True)
            linear = nn.Linear(hidden_size*2, output_size)
            self.F.append(nn.Sequential(
                f_i, extract_tensor(), linear, nn.LayerNorm(output_size)))

    def forward(self, x, idx):
        # x = self.F[0](x)
        x = torch.stack([self.coeffs[i, idx].view(-1, 1, 1)*f_i(x) + (self.B(self.effective_U[:, idx].T)).unsqueeze(1)
                         for i, f_i in enumerate(self.F)]).sum(dim=0)
        
        

        return x

    @ property
    def effective_U(self):
        return torch.relu(self.U)

    def control_sparsity_loss(self):
        # L1 regularization for sparsity on the control matrix U
        return torch.sum(torch.abs(self.effective_U))

    def sparsity_mask(self):

        # self.sparsity_threshold

        for j in range(self.control_size):
            tmp = torch.relu(self.U)[j, :]
            print(tmp)
            threshold = max(torch.quantile(tmp[tmp > 0], 0.3),
                            torch.min(tmp[tmp > 0]))  # threshold for sparsity pattern, if control_density is 0.1, then we take the 0.1 quantile of the positive values of the control input as the threshold unless it is smaller than the smallest positive value

            above_threshold = tmp > threshold

            # print of tmp those values that are above the threshold
            # print(
            #    f'Values under threshold {threshold} in control input {j}: {tmp[~above_threshold]}')

            # update sparsity pattern, if control input is below threshold
            self.sparsity_pattern[j, :] = ~(above_threshold)

        # set control inputs below threshold to 0
        with torch.no_grad():
            self.U[self.sparsity_pattern] = 0

        # if self.U.sum() == 0:
        #    print('All control inputs are 0')
        #    return


def multi_step_loss(X, y, model, loss_fn, lookback, teacher_forcing_ratio=0.5):
    recon = torch.zeros_like(y)
    recon[:lookback] = X[:lookback]  # Set initial inputs from X

    for i in range(len(X) - lookback):
        # Predict next step based on the past 'lookback' steps

        # Update the reconstruction with the new prediction
        if torch.rand(1) < teacher_forcing_ratio:
            # Use actual next value (ground truth)
            y_pred = model(y[i:i + lookback].float(),
                           torch.arange(i, i + lookback))
        else:
            y_pred = model(recon[i:i + lookback].float(),
                           torch.arange(i, i + lookback))
        recon[i + lookback] = y_pred[-1]
    # Compute the loss over the entire reconstructed sequence
    return loss_fn(recon, y)


def init_U(D_control, X1=None, X2=None, err=None, lookback=50):
    """Initialize the control input matrix U by applying NMF on the first iteration without control and take the residuals as U

    """

    U_0_init = np.zeros((D_control, lookback))
    if err is None:
        err = X2 - (X2 @ np.linalg.pinv(X1)) @ X1

    nnmf = NMF(n_components=D_control, init='random', random_state=0)
    W = nnmf.fit_transform(np.abs(err))
    W_max = W.max(axis=0, keepdims=True)
    U_0 = np.divide(W, W_max, where=W != 0, out=W).T

    return np.concatenate([U_0_init, U_0], axis=1)


class TimeSeriesDataset(torch.utils.data.Dataset):
    # TODO: make more explicit
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)

    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

def calculate_best_correlation(ground_truth, learned, num_subdyn):
    # Calculate pairwise correlations between ground truth and learned coefficients
    corr_matrix = torch.zeros((num_subdyn, num_subdyn))

    for i in range(num_subdyn):
        for j in range(num_subdyn):
            gt_flat = ground_truth[:,i].flatten()
            learned_flat = learned[:,j].flatten()
            # Compute correlation between ground truth[i] and learned[j]
            corr_matrix[i, j] = torch.corrcoef(torch.stack((gt_flat, learned_flat)))[0, 1]

    # Convert correlation matrix to negative for minimization
    cost_matrix = -corr_matrix.numpy()

    # Use the Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get the optimal permutation
    best_permutation = col_ind

    # Compute the average correlation for the best permutation
    best_corr_sum = corr_matrix[row_ind, col_ind].sum()

    # Return the correlation matrix and the best permutation
    return corr_matrix.numpy(), best_corr_sum / num_subdyn, best_permutation


def main(args):
    
    fix_point_change = args.fix_point_change
    eigenvalue_radius = args.eigenvalue_radius

    num_true_subdyn = args.num_subdyn

    generator = DLDSwithControl(CdLDSDataGenerator(
        K=num_true_subdyn, D_control=args.control_size, fix_point_change=fix_point_change, eigenvalue_radius=float(eigenvalue_radius), set_seed=num_true_subdyn))

    time_points = 1000

    # generate toy data
    timeseries = generator.datasets.generate_data(time_points, sigma=0.01).T
    states = generator.datasets.z_
    controls = generator.datasets.U_[:args.control_size,:]
    
    
    #timeseries = np.load(args.data_path).T
    #states = np.load(args.state_path)
    #controls = np.load(args.control_path)

    # train-test split for time series
    train_size = int(len(timeseries) * 0.8)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    lookback = args.lookback
    X_train_idx = np.arange(len(train)-lookback)
    y_train_idx = np.arange(lookback, len(train))

    X_test_idx = np.arange(len(train), len(timeseries)-lookback)
    y_test_idx = np.arange(len(train)+lookback, len(timeseries))

    test_idx = np.arange(len(train), len(timeseries))
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    
    K = np.unique(states).shape[0]
    
    # one hot encode states
    one_hot_states = np.zeros((K,len(states)))
    one_hot_states[states,np.arange(len(states))] = 1
    
    
    # shift states for plotting according to lookback
    states = states[lookback:]

    wandb.login(key='a79ac9d4509caa0d5e477c939a41d790e7711171')

    if args.eigenvalue_radius < 0.999:
        project_name = f"Oscillatory_FastDecay"
    else:
        project_name = f"Oscillatory_SlowDecay"

    run = wandb.init(
        # Set the project where this run will be logged
        project=f"TRUE_Control_Coeff_{project_name}_{str(args.num_subdyn)}_State_Bias_Init_Rand_LSTM",
        # dir=f'/state_{str(args.num_subdyn)}/fixpoint_change_{str(args.fix_point_change)}', # This is not a wandb feature yet, see issue: https://github.com/wandb/wandb/issues/6392
        # name of the run is a combination of the model name and a timestamp
        # reg{str(round(args.reg, 3))}_
        name=f"smooth{str(round(args.smooth, 3))}_fixpoint_change_{str(args.fix_point_change)}_batch_size_{str(args.batch_size)}_lookback_{str(args.lookback)}_epochs_{str(args.epochs)}",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_subdyn": args.num_subdyn,
            "lookback": args.lookback,
            "smooth": args.smooth,
            "control_sparsity_reg": args.control_sparsity_reg
        },
    )
    
    
    hidden_size = 4
    input_size = train.shape[1]

    model = DLDSModel(input_size=input_size, hidden_size=hidden_size,
                      output_size=input_size, time_points=len(timeseries), num_subdyn=args.num_subdyn, lookback=args.lookback).float()

    with torch.no_grad():
        
        # Initialize coefficients with one-hot encoded states
        model.coeffs = torch.nn.Parameter(torch.tensor(one_hot_states, requires_grad=True, dtype=torch.float32))
        
        # Initialize control input matrix U and add noise
        model.U = torch.nn.Parameter(torch.tensor(controls, requires_grad=True, dtype=torch.float32)) # + torch.randn_like(model.U) * args.sigma)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # loader = TimeSeriesDataset(data.TensorDataset(X_train.float(), y_train.float()), shuffle=True, batch_size=8)
    data = TimeSeriesDataset(list(zip(X_train.float(), y_train.float())))

    # create an iterable over our data, no shuffling because we want to keep the temporal information
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    initial_teacher_forcing_ratio = 0.5
    final_teacher_forcing_ratio = 0.0
    n_epochs = args.epochs
    for epoch in range(n_epochs):
        model.train()
        for (X_batch, y_batch), idx in loader:

            teacher_forcing_ratio = initial_teacher_forcing_ratio - \
                (initial_teacher_forcing_ratio -
                 final_teacher_forcing_ratio) * (epoch / n_epochs)
            # indices of the batch
            y_pred = model(X_batch.float(), X_train_idx[idx])

            # sparsifying control inputs with mask
            # model.sparsity_mask()
            # with torch.no_grad():
            #    model.U = torch.relu(model.U)

            coeff_delta = model.coeffs[:, 1:] - model.coeffs[:, :-1]
            # L2 norm of the difference between consecutive coefficients
            smooth_reg = torch.norm(coeff_delta, p=2)
            
            smooth_reg_loss = args.smooth * smooth_reg * input_size

            control_sparsity = model.control_sparsity_loss()
            single_loss = loss_fn(y_pred, y_batch)
            multi_loss = multi_step_loss(
                X_batch, y_batch, model, loss_fn, lookback, teacher_forcing_ratio)
            loss = smooth_reg_loss + args.loss_reg * \
                multi_loss + single_loss + args.control_sparsity_reg * control_sparsity
            wandb.log({'loss': loss.item()})
            wandb.log({'single_reconstruction_loss': single_loss.item()})
            wandb.log({'control_sparsity_loss': control_sparsity.item()})
            # wandb.log({'sparsity_loss': sparsity_loss.item()})
            wandb.log({'smooth_reg_loss': smooth_reg_loss.item()})
            wandb.log({'multi_reconstruction_loss': multi_loss.item()})
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 7)
            optimizer.step()

        # Validation
        if epoch % 10 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train.float(), X_train_idx)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test.float(), X_test_idx)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" %
              (epoch, train_rmse, test_rmse))

    print("Training finished")
    print('Storing visualizations..')
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = model(X_train.float(), X_train_idx)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(
            X_train.float(), X_train_idx)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size+lookback:len(timeseries)
                  ] = model(X_test.float(), X_test_idx)[:, -1, :]
        # plot
        plt.plot(timeseries)
        plt.plot(train_plot, c='r')
        plt.plot(test_plot, c='g')
        wandb.log({"train": plt})
        
        
        corr_mat, best_corr, _ = calculate_best_correlation(
            torch.tensor(one_hot_states).T, model.coeffs.T, K)
        wandb.log({"coeff_correlation": best_corr})
        
        
        _, control_corr, _ = calculate_best_correlation(
            torch.tensor(controls).T, model.U.T, controls.shape[0])
        wandb.log({"control_correlation": control_corr})
    
        # Create the Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_mat,
        x=[f'Learned {i+1}' for i in range(K)],
        y=[f'True {i+1}' for i in range(K)],
        colorscale='Viridis',
        zmin=-1, zmax=1,
        hoverongaps=False,
        text=corr_mat.round(2),
        hoverinfo="text",
    ))

    

    wandb.log({"correlation_matrix": fig})

    fig = util.plotting(model.coeffs.detach().numpy()[
                        :, :train_size].T, title='coeffs', plot_states=args.plot_states, states=states)
    wandb.log({"coeffs": fig})

    time_series, _ = create_dataset(timeseries, lookback=lookback)
    # multi-step reconstruction by always using the last prediction as input
    recon = torch.zeros_like(time_series)
    recon[:lookback] = time_series[:lookback]
    with torch.no_grad():
        for i in range(len(time_series)-lookback):
            y_pred = model(recon[i:i+lookback].float(),
                           torch.arange(i, i+lookback))
            recon[i+lookback] = y_pred[-1]
    
    

    fig = util.plotting(recon.detach().numpy()[
                        :, -1, :], title='reconstruction', stack_plots=False, plot_states=args.plot_states, states=states)
    wandb.log({"multi-step reconstruction": fig})

    result = model(time_series.float(), torch.arange(len(timeseries)-lookback))
    fig = util.plotting([time_series[:, -1, :], result.detach().numpy()
                        [:, -1, :]], title='result', stack_plots=True, plot_states=args.plot_states, states=states)
    wandb.log({"single-step + ground truth reconstruction": fig})

    # print(model.U.detach().numpy())

    # control plot
    fig = util.plotting(model.effective_U.detach().numpy()[
                        :, :train_size].T, title='Control Signals')

    fig.add_trace(go.Scatter(
        x=np.arange(train_size),
        y=controls[0],
        mode='lines',
        line=dict(color='black', width=2),
        name='true control signals'
    ))

    wandb.log({"Control Matrix": fig})

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--data_path', type=str, default='data.npy')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_subdyn', type=int, default=2)
    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--control_sparsity_reg', type=float, default=0.0001)
    parser.add_argument('--smooth', type=float, default=0.0001)
    parser.add_argument('--dynamics_path', type=str, default='As.npy')
    parser.add_argument('--state_path', type=str, default='states.npy')
    parser.add_argument('--control_path', type=str, default='controls.npy')
    parser.add_argument('--control_size', type=int, default=1)
    parser.add_argument('--lookback', type=int, default=50)
    parser.add_argument('--fix_point_change', type=bool, default=False),
    parser.add_argument('--eigenvalue_radius', type=float, default=0.995),
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--loss_reg', type=float, default=0.1)
    parser.add_argument('--plot_states', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    main(args)
