
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import wandb
import util


class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor


class AirModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_points, num_subdyn, lookback):
        super().__init__()

        self.coeffs = torch.nn.Parameter(torch.tensor(
            np.random.rand(num_subdyn, time_points), requires_grad=True))

        self.F = torch.nn.ParameterList()

        for _ in range(num_subdyn):
            f_i = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True, bias=True, bidirectional=True)
            linear = nn.Linear(hidden_size*2, output_size)
            self.F.append(nn.Sequential(
                f_i, extract_tensor(), linear, nn.BatchNorm1d(lookback)))

    def forward(self, x, idx):
        # x = self.F[0](x)

        x = torch.stack([self.coeffs[i, idx].view(-1, 1, 1)*f_i(x)
                         for i, f_i in enumerate(self.F)]).sum(dim=0)

        return x


def multi_step_loss(X, y, model, loss_fn, lookback, teacher_forcing_ratio=0.5):
    recon = torch.zeros_like(y)
    recon[:lookback] = X[:lookback]  # Set initial inputs from X

    for i in range(len(X) - lookback):
        # Predict next step based on the past 'lookback' steps
        y_pred = model(recon[i:i + lookback].float(),
                       torch.arange(i, i + lookback))

        # Update the reconstruction with the new prediction
        if torch.rand(1) < teacher_forcing_ratio:
            # Use actual next value (ground truth)
            recon[i + lookback] = y[i + lookback]
        else:
            recon[i + lookback] = y_pred[-1]
    # Compute the loss over the entire reconstructed sequence
    return loss_fn(recon, y)


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


def main(args):

    timeseries = np.load('data.npy').T

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

    wandb.login(key='a79ac9d4509caa0d5e477c939a41d790e7711171')

    if args.eigenvalue_radius < 0.999:
        project_name = f"Oscillatory_FastDecay"
    else:
        project_name = f"Oscillatory_SlowDecay"

    run = wandb.init(
        # Set the project where this run will be logged
        project=f"{project_name}_{str(args.num_subdyn)}_State_Bias_C-Init_Rand_A-Init_Rand_SpectralLinear",
        # dir=f'/state_{str(args.num_subdyn)}/fixpoint_change_{str(args.fix_point_change)}', # This is not a wandb feature yet, see issue: https://github.com/wandb/wandb/issues/6392
        # name of the run is a combination of the model name and a timestamp
        # reg{str(round(args.reg, 3))}_
        name=f"smooth{str(round(args.smooth, 3))}_fixpoint_change_{str(args.fix_point_change)}_batch_size_{str(args.batch_size)}_random_coeffs",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    )

    hidden_size = 4
    input_size = train.shape[1]

    model = AirModel(input_size=input_size, hidden_size=hidden_size,
                     output_size=input_size, time_points=len(timeseries), num_subdyn=args.num_subdyn, lookback=args.lookback).float()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # loader = TimeSeriesDataset(data.TensorDataset(X_train.float(), y_train.float()), shuffle=True, batch_size=8)
    data = TimeSeriesDataset(list(zip(X_train.float(), y_train.float())))

    # create an iterable over our data, no shuffling because we want to keep the temporal information
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)

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

            coeff_delta = model.coeffs[:, 1:] - model.coeffs[:, :-1]
            # L1 norm of the difference
            smooth_reg = coeff_delta.abs().sum()
            smooth_reg_loss = args.smooth * smooth_reg * input_size

            multi_loss = multi_step_loss(
                X_batch, y_batch, model, loss_fn, lookback, teacher_forcing_ratio)
            loss = smooth_reg_loss + args.loss_reg * \
                multi_loss + loss_fn(y_pred, y_batch)
            wandb.log({'loss': loss.item()})
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

    fig = util.plotting(model.coeffs.detach().numpy()[
                        :, :train_size].T, title='coeffs')
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
                        :, -1, :], title='reconstruction', stack_plots=False)
    wandb.log({"multi-step reconstruction": fig})

    result = model(time_series.float(), torch.arange(len(timeseries)-lookback))
    fig = util.plotting([time_series[:, -1, :], result.detach().numpy()
                        [:, -1, :]], title='result', stack_plots=True)
    wandb.log({"single-step reconstruction": fig})

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--data_path', type=str, default='data.npy')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_subdyn', type=int, default=2)
    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--smooth', type=float, default=0.0001)
    parser.add_argument('--dynamics_path', type=str, default='As.npy')
    parser.add_argument('--state_path', type=str, default='states.npy')
    parser.add_argument('--lookback', type=int, default=50)
    parser.add_argument('--fix_point_change', type=bool, default=False),
    parser.add_argument('--eigenvalue_radius', type=float, default=0.995),
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--loss_reg', type=float, default=0.1)
    args = parser.parse_args()
    print(args)
    main(args)
