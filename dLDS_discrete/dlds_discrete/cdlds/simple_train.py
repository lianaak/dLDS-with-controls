from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from models import SimpleNN


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

    # create an iterable over our data, no shuffling because we want to keep the temporal information
    train_loader = DataLoader(list(zip(X, y)), batch_size=32, shuffle=False)

    # Build the model
    input_size = X.shape[1]
    hidden_sizes = [8, 4, 4, 4]
    output_size = y.shape[1]
    model = SimpleNN(input_size, hidden_sizes, output_size)

    loss_fn = torch.nn.MSELoss()  # mean squared error loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)  # Adam optimizer

    # training the model
    for epoch in range(args.epochs):

        # by wrapping the data loader with tqdm, we can see a pretty progress bar
        with tqdm(train_loader, unit='batch') as tepoch:

            # iterate over the data loader which will return a batch of data and target
            for data, target in tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                # if we call the model with the input, it will call the forward function
                y_pred = model(data)
                loss = loss_fn(y_pred, target)  # compute the loss
                # reset the gradients to zero before backpropagation to avoid accumulation
                optimizer.zero_grad()
                loss.backward()  # backpropagation of the loss to compute the gradients
                optimizer.step()  # update the weights
                accuracy = torch.mean(torch.abs(target - y_pred))
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item())

                # save the model
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--data_path', type=str, default='data.npy')
    parser.add_argument('--epochs', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
