from audioop import bias
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import numpy as np
from models import DeepDLDS
import wandb
import submitit
import plotly.express as px
import util


class TimeSeriesDataset(Dataset):
    # TODO: make more explicit
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx


def main(args):

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load the data
    data = np.load(args.data_path)

    # Split data into input and target
    X = torch.tensor(data[:, :-1].T, dtype=torch.float32)
    # target for single-step prediction
    y = torch.tensor(data[:, 1:].T, dtype=torch.float32)

    data = TimeSeriesDataset(list(zip(X, y)))

    # create an iterable over our data, no shuffling because we want to keep the temporal information
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    # Build the model
    input_size = X.shape[1]
    num_subdyn = args.num_subdyn
    output_size = y.shape[1]
    model = DeepDLDS(input_size, output_size, num_subdyn,
                     X.shape[0], softmax_temperature=0.0001)

    A = np.load('As.npy')

    # initialize F with A
    with torch.no_grad():
        for idx, f in enumerate(model.F):
            f.w = torch.tensor(A[idx], dtype=torch.float32)

    # ransac = RANSACRegressor()
    # ransac.fit(X, y)

    # weights = ransac.estimator_.coef_
    # bias = ransac.estimator_.intercept_

    # for f in model.F:
    #    f.w = torch.tensor(weights.T, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)  # Adam optimizer

    losses = []
    # wandb.login()

    # run = wandb.init(
    # Set the project where this run will be logged
    #    project="DeepDLDS",
    # Track hyperparameters and run metadata
    #    config={
    #        "learning_rate": args.lr,
    #        "epochs": args.epochs,
    #    },
    # )

    # training the model
    for epoch in range(args.epochs):

        # by wrapping the data loader with tqdm, we can see a pretty progress bar
        with tqdm(train_loader, unit='batch') as tepoch:

            # iterate over the data loader which will return a batch of data and target
            for (batch, target), idx in tepoch:
                # print(idx)
                tepoch.set_description(f'Epoch {epoch}')
                for i in range(len(batch)):
                    # if we call the model with the input, it will call the forward function

                    y_pred = model(batch[i], idx[i])  # forward pass

                    regularization_term = model.coeffs.abs().sum()

                    # l2_reg = sum(p.pow(2).sum() for p in model.parameters())

                    # Difference between t and t-1 of the coefficients
                    coeff_delta = model.coeffs[:,
                                               1:] - model.coeffs[:, :-1]
                    # L1 norm of the difference
                    smooth_reg = coeff_delta.abs().sum()

                    # add a penalty for when the eigenvalues are not in the unit circle
                    f_sum = np.sum([model.coeffs[fidx, idx[i]].detach().numpy(
                    )*f_i.effective_W().detach().numpy() for fidx, f_i in enumerate(model.F)], axis=0)
                    eigenvalues = torch.linalg.eig(
                        torch.tensor(f_sum, dtype=torch.float32))[0]

                    loss = dlds_loss(
                        y_pred, target[i].unsqueeze(0)) + args.reg * regularization_term + args.smooth * smooth_reg * input_size
                    # reset the gradients to zero before backpropagation to avoid accumulation
                    optimizer.zero_grad()
                    # backpropagation of the loss to compute the gradients
                    loss.backward(retain_graph=True)
                    optimizer.step()  # update the weights

                    tepoch.set_postfix(
                        loss=loss.item())
                    # wandb.log({'loss': loss.item()})

            losses.append(loss.item())

            # save the model
            torch.save(model.state_dict(), os.path.join(
                args.model_path, 'model.pth'))

    # plot the losses
    # plt.plot(losses)
    # plt.show()
    # save the loss

    print(f'Finished training with reg={args.reg}, smooth={args.smooth}')

    # predict the next time step
    X2_hat = util.single_step(X, model)
    X2_hat_multi = util.multi_step(X, model)

    reg_string = str(args.reg).replace('.', '_')
    smooth_string = str(args.smooth).replace('.', '_')

    print(f'Storing visualizations...')

    fig = px.line(
        torch.stack(X2_hat).squeeze(), title=f'Prediction with reg_term={args.reg}, smooth={args.smooth}')
    fig.write_image(
        f'visualizations/hyperparameters/reg_{reg_string}_smooth_{smooth_string}_RECON.png', engine="orca", width=900, height=450, scale=3)

    fig = px.line(
        torch.stack(X2_hat_multi).squeeze(), title=f'Prediction with reg_term={args.reg}, smooth={args.smooth}')
    fig.write_image(
        f'visualizations/hyperparameters/reg_{reg_string}_smooth_{smooth_string}_RECON_MULTI.png', engine="orca", width=900, height=450, scale=3)

    # coefficients
    coefficients = np.array([c.detach().numpy()
                            for c in model.coeffs])
    fig = px.line(
        coefficients.T, title=f'Coefficients with reg_term={args.reg}, smooth={args.smooth}')
    fig.write_image(
        f'visualizations/hyperparameters/reg_{reg_string}_smooth_{smooth_string}_COEFFS.png', engine="orca", width=900, height=450, scale=3)

    np.save('loss.npy', loss.item())
    return loss.item()


def dlds_loss(y_pred, y_true):
    return torch.nn.MSELoss()(y_pred, y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--data_path', type=str, default='data.npy')
    parser.add_argument('--epochs', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_subdyn', type=int, default=3)
    parser.add_argument('--reg', type=float, default=0.0001)
    parser.add_argument('--smooth', type=float, default=0.0001)
    args = parser.parse_args()
    print(args)
    main(args)
