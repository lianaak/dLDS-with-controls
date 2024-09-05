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


def reg_error(layer, p=2):
    return torch.max(torch.norm(layer.effective_W(), p) - 1.0, torch.zeros(1)) + \
        torch.max(0.0 -
                  torch.norm(layer.effective_W(), p), torch.zeros(1))


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

    A = np.load(args.dynamics_path)
    states = np.load(args.state_path)
    # bias = np.load(args.bias_path)

    # one-hot encoding of the states
    one_hot_states = np.zeros((num_subdyn, len(states)))
    one_hot_states[states, np.arange(len(states))] = 1

    # initialize F with A
    with torch.no_grad():

        """         
        for idx, f in enumerate(model.F):
            if args.fix_point_change:
                print(f.bias.shape)
                pass
                # f.linear.bias = torch.nn.Parameter(
                #    torch.tensor(bias[idx], dtype=torch.float32)) """

        """
        # decompose A with SVD
        f.linear.weight = torch.nn.Parameter(
            torch.tensor(A[idx], dtype=torch.float32))

        # adding a bit of noise
        f.linear.weight += 1 * torch.randn_like(f.linear.weight) """

        # if args.fix_point_change:
        #    model.linear.bias = torch.nn.Parameter(
        #        torch.tensor(one_hot_states, dtype=torch.float32))

        # model.coeffs = torch.nn.Parameter(
        #     torch.tensor(one_hot_states, dtype=torch.float32))

        # sigma = args.sigma
        # adding a bit of noise
        # model.coeffs += sigma * torch.randn_like(model.coeffs)

        # model.coeffs = torch.nn.Parameter(torch.tensor(
        #    np.random.rand(num_subdyn, X.shape[0]), requires_grad=True))

    # ransac = RANSACRegressor()
    # ransac.fit(X, y)

    # weights = ransac.estimator_.coef_
    # bias = ransac.estimator_.intercept_

    # for f in model.F:
    #    f.w = torch.tensor(weights.T, dtype=torch.float32)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)  # Adam optimizer

    losses = []

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

    # Smooth Loss: Difference between t and t-1 of the coefficients
    coeff_delta = model.coeffs[:,
                               1:] - model.coeffs[:, :-1]
    # L1 norm of the difference
    smooth_reg = coeff_delta.abs().sum()
    smooth_reg_loss = args.smooth * smooth_reg * input_size

    y_pred = torch.zeros(len(X), input_size)

    for i in range(len(X)):
        y_pred[i] = model(X[i], i)  # forward pass

    # Reconstruction Loss
    reconstruction_loss = dlds_loss(y_pred, y)

    loss = reconstruction_loss + \
        smooth_reg_loss

    wandb.log({'loss': loss.item()})
    # wandb.log({'sparsity_loss': sparsity_loss.item()})
    wandb.log({'smooth_reg_loss': smooth_reg_loss.item()})
    wandb.log({'reconstruction_loss': reconstruction_loss.item()})

    # training the model
    for epoch in range(args.epochs):

        # by wrapping the data loader with tqdm, we can see a pretty progress bar
        with tqdm(train_loader, unit='batch') as tepoch:

            # iterate over the data loader which will return a batch of data and target
            for (batch, target), idx in tepoch:
                y_pred = torch.zeros(len(batch), input_size)

                tepoch.set_description(f'Epoch {epoch}')

                for i in range(len(batch)):
                    y_pred[i] = model(batch[i], idx[i])  # forward pass

                # Sparsity loss: L1 regularization term
                regularization_term = model.coeffs.abs().sum()
                sparsity_loss = args.reg * regularization_term

                # Smooth Loss: Difference between t and t-1 of the coefficients
                coeff_delta = model.coeffs[:,
                                           1:] - model.coeffs[:, :-1]
                # L1 norm of the difference
                smooth_reg = coeff_delta.abs().sum()
                smooth_reg_loss = args.smooth * smooth_reg * input_size

                # Reconstruction Loss
                reconstruction_loss = dlds_loss(y_pred, target)

                # Eigenvalue regularization loss
                eigenvalue_reg_loss = 0.01 * torch.stack(
                    [reg_error(f_i) for f_i in model.F], axis=0).sum()

                # entropy based sparsity loss for coefficients
                # sparsity_loss_2 = 0.01*torch.nn.functional.gumbel_softmax(
                #    model.coeffs, tau=0.1, hard=True).sum()

                # sparsity_loss_2 = -torch.sum(model.coeffs * torch.log(c + epsilon), dim=1)  # Sum over the systems
                # entropy_loss = torch.mean(sparsity_loss_2)
                sparsity = - \
                    torch.sum(model.coeffs *
                              torch.log(model.coeffs + 1e-10), dim=0).mean()

                sparsity_loss = args.reg * sparsity

                # print(f"Sparsity Loss: {sparsity_loss_2}")

                loss = reconstruction_loss + \
                    smooth_reg_loss + sparsity_loss

                # log loss before backpropagation
                # if epoch == 0:
                #    wandb.log({'loss': loss.item()})
                # wandb.log({'sparsity_loss': sparsity_loss.item()})
                #    wandb.log({'smooth_reg_loss': smooth_reg_loss.item()})
                #    wandb.log(
                #        {'reconstruction_loss': reconstruction_loss.item()})
                # wandb.log(
                #    {'eigenvalue_reg_loss': eigenvalue_reg_loss.item()})

                # reset the gradients to zero before backpropagation to avoid accumulation
                optimizer.zero_grad()
                # backpropagation of the loss to compute the gradients
                loss.backward(retain_graph=True)
                optimizer.step()  # update the weights

                tepoch.set_postfix(
                    loss=loss.item())

                # log the losses
            losses.append(loss.item())

            wandb.log({'loss': loss.item()})
            wandb.log({'sparsity_loss': sparsity_loss.item()})
            wandb.log({'smooth_reg_loss': smooth_reg_loss.item()})
            wandb.log({'reconstruction_loss': reconstruction_loss.item()})
            # wandb.log({'eigenvalue_reg_loss': eigenvalue_reg_loss.item()})

            # log learning rate

    # save the model
    torch.save(model.state_dict(), os.path.join(
        args.model_path, 'model.pth'))

    print(f'Finished training with smooth={args.smooth} and reg={args.reg}')

    # add a penalty for when the eigenvalues are not in the unit circle
    # f_sum = np.sum([model.coeffs[fidx, idx[i]].detach().numpy(
    # )*f_i.effective_W().detach().numpy() for fidx, f_i in enumerate(model.F)], axis=0)
    # eigenvalues = torch.linalg.eig(
    #    torch.tensor(f_sum, dtype=torch.float32))[0]

    # create correlation matrix plot of coefficients and states
    # heat = go.Heatmap()

    # predict the next time step
    X2_hat = util.single_step(X, model)
    X2_hat_multi = util.multi_step(X, model)

    reg_string = str(args.reg).replace('.', '_')
    smooth_string = str(args.smooth).replace('.', '_')

    # if folder does not exist, create it
    visualizations_path = 'visualizations/hyperparameters'
    if not os.path.exists(visualizations_path):
        os.makedirs(visualizations_path)

    saving_path = visualizations_path + \
        '/state'+str(args.num_subdyn)+'_nofixpoint'

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    print(f'Storing visualizations...')

    X2_hat_residuals = torch.stack(X2_hat).squeeze() - y
    X2_hat_multi_residuals = torch.stack(X2_hat_multi[1:]).squeeze() - y

    fig = util.plotting(
        [y, torch.stack(X2_hat).squeeze()], title=f'Ground Truth and single-step reconstruction with smooth={args.smooth} and coefficient entropy={args.reg}', plot_states=True, states=states, show=False, stack_plots=True)
    wandb.log({'single-step + ground truth': fig})

    fig = util.plotting(
        torch.stack(X2_hat_multi[1:]).squeeze(), title=f'Multi-step reconstruction with smooth={args.smooth} and coefficient entropy={args.reg}', plot_states=True, states=states, show=False)
    wandb.log({'multi-step': fig})

    fig = util.plotting(
        X2_hat_residuals, title=f'Residuals of single-step reconstruction with smooth={args.smooth} and coefficient entropy={args.reg}', plot_states=True, states=states, show=False)
    fig.write_image(
        f'{saving_path}/smooth_{smooth_string}_RECON.png', width=900, height=450, scale=3)

    wandb.log({'single-step residuals': fig})

    fig = util.plotting(
        X2_hat_multi_residuals, title=f'Residuals of multi-step reconstruction with smooth={args.smooth} and coefficient entropy={args.reg}', plot_states=True, states=states, show=False)
    fig.write_image(
        f'{saving_path}/smooth_{smooth_string}_RECON_MULTI.png', width=900, height=450, scale=3)

    wandb.log({'multi-step residuals': fig})

    # coefficients
    coefficients = np.array([c.detach().numpy()
                            for c in model.coeffs])
    fig = util.plotting(
        coefficients.T, title=f'Coefficients with smooth={args.smooth} and coefficient entropy={args.reg}', plot_states=True, states=states, show=False)
    fig.write_image(
        f'{saving_path}/smooth_{smooth_string}_COEFFS.png', width=900, height=450, scale=3)

    # log the plots to wandb
    wandb.log({'coefficients': fig})

    # np.save(
    #    f'losses/reg_{reg_string}_smooth_{smooth_string}_loss.npy', loss.item())
    # np.save('loss.npy', loss.item())

    # return loss.item()
    return losses


def dlds_loss(y_pred, y_true):
    return torch.nn.MSELoss()(y_pred, y_true)


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
    # parser.add_argument('--bias_path', type=str, default='Bias.npy')
    parser.add_argument('--fix_point_change', type=bool, default=False),
    parser.add_argument('--eigenvalue_radius', type=float, default=0.995),
    parser.add_argument('--sigma', type=float, default=0.01)
    args = parser.parse_args()
    print(args)
    main(args)
