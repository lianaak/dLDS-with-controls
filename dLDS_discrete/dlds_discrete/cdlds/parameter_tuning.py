import argparse
import comm
import numpy as np
from models import DLDSwithControl, DeepDLDS, SimpleNN
from datasets import CdLDSDataGenerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import plotly.express as px
import os


def main(args):
    # load data from file
    X = np.load(args.data_path)

    input_size = X.shape[0]
    X1 = torch.tensor(X[:, :-1], dtype=torch.float32)
    t = X1.shape[1]

    # create a dictionary with different hyperparameters like reg and smoothness
    hyperparameters = {
        'reg': [0.0, 0.0001, 0.001, 0.01, 0.1, 1, 10],
        'smooth': [0.0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    }

    num_subdyn = args.num_subdyn
    # loop over all hyperparameters

    for reg in hyperparameters['reg']:
        for smooth in hyperparameters['smooth']:

            command = 'python train.py --data_path ' + args.data_path + ' --reg ' + \
                str(reg) + ' --smooth ' + str(smooth) + ' --epochs ' + \
                str(args.epochs) + ' --lr ' + str(args.lr) + \
                ' --num_subdyn ' + str(num_subdyn)
            os.system(command)
            print(f'Finished training with reg={reg}, smooth={smooth}')
            simple_model = DeepDLDS(input_size, input_size,
                                    num_subdyn, t)
            model = torch.load('models/model.pth', weights_only=False)
            simple_model.load_state_dict(model)

            # predict the next time step
            X2_hat = []
            with torch.no_grad():
                for i in range(t):
                    y = simple_model(X1[:, i], i)
                    X2_hat.append(y)

            reg_string = str(reg).replace('.', '_')
            smooth_string = str(smooth).replace('.', '_')

            print(f'Storing visualizations...')

            fig = px.line(
                torch.stack(X2_hat).squeeze(), title=f'Prediction with reg_term={reg}, smooth={smooth}')
            fig.write_image(
                f'visualizations/hyperparameters/reg_{reg_string}_smooth_{smooth_string}_RECON.png', engine="orca", width=900, height=450, scale=3)

            # coefficients
            coefficients = np.array([c.detach().numpy()
                                    for c in simple_model.coeffs])
            fig = px.line(
                coefficients.T, title=f'Coefficients with reg_term={reg}, smooth={smooth}')
            fig.write_image(
                f'visualizations/hyperparameters/reg_{reg_string}_smooth_{smooth_string}_COEFFS.png', engine="orca", width=900, height=450, scale=3)

            print(f'Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/data.npy')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_subdyn', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    print(args)
    main(args)
