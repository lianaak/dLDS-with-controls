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
import time
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import exp_to_df
from submitit import AutoExecutor, LocalJob, DebugJob

os.environ.pop("SUBMITIT_PREEMPT_SIGNAL", None)


def main(args):
    # load data from file
    X = np.load(args.data_path)

    input_size = X.shape[0]
    X1 = torch.tensor(X[:, :-1], dtype=torch.float32)
    t = X1.shape[1]

    ax_client = AxClient()
    ax_client.create_experiment(
        name="dLDS_hyperparameter_tuning",
        parameters=[
            {"name": "reg", "type": "range", "bounds": [
                0.00001, 10.00001], "log_scale": True},
            {"name": "smooth", "type": "range", "bounds": [
                0.00001, 10.00001], "log_scale": True},
            {"name": "num_subdyn", "type": "range", "bounds": [1, 5]}
        ],
        objectives={'loss': ObjectiveProperties(minimize=True)}
    )

    num_subdyn = args.num_subdyn
    # loop over all hyperparameters

    for _ in range(10):  # Number of trials or iterations
        # Get next parameters to evaluate
        parameters, trial_index = ax_client.get_next_trial()
        try:
            # Submit the job and wait for result
            accuracy = evaluate_parameterization(parameters)
            # Complete the trial with obtained result
            ax_client.complete_trial(trial_index=trial_index, raw_data={
                                     "accuracy": (accuracy, 0.0)})  # 0.0 as standard error
        except Exception as e:
            print(f"Failed trial with error: {e}")
            ax_client.log_trial_failure(
                trial_index=trial_index)  # Log failed trial

    # Save the experiment
    ax_client.save_to_json_file("experiment.json")

    # Get best parameters
    best_parameters, values = ax_client.get_best_parameters()
    print(f"Best parameters: {best_parameters}")
    print(f"Best values: {values}")


def train_model(reg, smooth, num_subdyn):

    command = 'python train.py --data_path ' + args.data_path + ' --reg ' + \
        str(reg) + ' --smooth ' + str(smooth) + ' --epochs ' + \
        str(args.epochs) + ' --lr ' + str(args.lr) + \
        ' --num_subdyn ' + str(num_subdyn)
    os.system(command)
    # get the loss
    loss = np.load('loss.npy')
    print(f'Finished')
    return loss


def submitit_job(reg, smooth):
    # Directory for Submitit logs
    executor = AutoExecutor(folder="submitit_jobs")
    executor.update_parameters(
        timeout_min=60  # ,  # Timeout for each job in minutes
        # slurm_partition="dev",  # Replace with your SLURM partition name
        # gpus_per_node=1,  # Number of GPUs if required
    )

    job = executor.submit(train_model, reg, smooth)
    return job

# Assuming ax_client is already initialized as shown in step 2


def evaluate_parameterization(parameterization):
    reg = parameterization["reg"]
    smooth = parameterization["smooth"]

    job = submitit_job(reg, smooth)
    return job.result()  # Wait for job to complete and get result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/data.npy')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_subdyn', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    print(args)
    main(args)
