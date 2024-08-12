from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import numpy as np
from models import DeepDLDS


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
    train_loader = DataLoader(data, batch_size=32, shuffle=True)

    # Build the model
    input_size = X.shape[1]
    num_subdyn = 3
    output_size = y.shape[1]
    model = DeepDLDS(input_size, output_size, num_subdyn,
                     X.shape[0], softmax_temperature=0.0001)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)  # Adam optimizer

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
                    coefficients = torch.cat(
                        [c.view(-1) for c in model.coeffs])
                    # regularization_term = sum(torch.linalg.norm(
                    #    p, 1) for p in model.soft_coeffs)
                    regularization_term = model.coeffs.abs().sum()

                    # l2_reg = sum(p.pow(2).sum() for p in model.parameters())

                    # Difference between t and t-1 of the coefficients
                    coeff_delta = model.coeffs[:,
                                               1:] - model.coeffs[:, :-1]
                    # L1 norm of the difference
                    smooth_reg = coeff_delta.abs().sum()

                    loss = dlds_loss(
                        y_pred, target[i]) + args.reg * regularization_term + args.smooth * smooth_reg  # compute the loss
                    # reset the gradients to zero before backpropagation to avoid accumulation
                    optimizer.zero_grad()
                    loss.backward()  # backpropagation of the loss to compute the gradients
                    optimizer.step()  # update the weights

                    # tepoch.set_postfix(
                    #    loss=loss.item(), accuracy=accuracy.item())

                # save the model
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'model.pth'))


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
