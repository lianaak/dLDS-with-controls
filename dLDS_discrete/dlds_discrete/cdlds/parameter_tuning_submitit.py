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
            {"name": "num_subdyn", "type": "range",
                "bounds": [1, 3], "value_type": "int"},
            {"name": "learning_rate", "type": "range", "bounds": [
                0.00001, 0.1], "log_scale": True},
            {"name": "epochs", "type": "range", "bounds": [
                1, 50], "value_type": "int"}
        ],
        objectives={'loss': ObjectiveProperties(minimize=True)}
    )

    total_budget = 10
    num_parallel_jobs = 3

    jobs = []
    submitted_jobs = 0

    # Run until all the jobs have finished and our budget is used up.
    while submitted_jobs < total_budget or jobs:
        for job, trial_index in jobs[:]:
            # Poll if any jobs completed
            # Local and debug jobs don't run until .result() is called.
            if job.done() or type(job) in [LocalJob, DebugJob]:
                result = job.result()
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data={"loss": (result, 0.0)})
                jobs.remove((job, trial_index))
            # Get next parameters to evaluate

            # Schedule new jobs if there is availablity
        trial_index_to_param, _ = ax_client.get_next_trials(
            max_trials=min(num_parallel_jobs - len(jobs), total_budget - submitted_jobs))
        for trial_index, parameters in trial_index_to_param.items():

            # Submit the job and wait for result
            job = submitit_job(parameters)
            submitted_jobs += 1
            jobs.append((job, trial_index))
            time.sleep(1)

        # Sleep for a bit before checking the jobs again to avoid overloading the cluster.
        # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
        time.sleep(30)

    # Save the experiment
    ax_client.save_to_json_file("experiment.json")

    # Get best parameters
    best_parameters, values = ax_client.get_best_parameters()

    # Store best parameters
    with open('best_parameters.txt', 'w') as f:
        f.write(f"Best parameters: {best_parameters}")
        f.write(f"Best values: {values}")


def train_model(parameters):

    fix_point_change = False
    eigenvalue_radius = 0.999

    generator = DLDSwithControl(CdLDSDataGenerator(
        K=parameters['num_subdyn'], D_control=0, fix_point_change=fix_point_change, eigenvalue_radius=eigenvalue_radius))

    time_points = 1000

    # generate toy data
    X = generator.datasets.generate_data(time_points, sigma=0.01)

    # z-score normalization
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    np.save(args.data_path, X)
    # save the As of the model
    np.save(args.dynamics_path, np.array(generator.datasets.A))

    np.save(args.state_path, generator.datasets.z_)

    command = 'python train.py --data_path ' + args.data_path + ' --reg ' + \
        str(parameters['reg']) + ' --smooth ' + str(parameters['smooth']) + ' --epochs ' + \
        str(args.epochs) + ' --lr ' + str(args.lr) + \
        ' --num_subdyn ' + str(parameters['num_subdyn']) + ' --dynamics_path ' + \
        args.dynamics_path + ' --state_path ' + args.state_path + \
        ' --fix_point_change ' + \
        str(fix_point_change) + ' --eigenvalue_radius ' + str(eigenvalue_radius)
    os.system(command)
    # get the loss
    loss = np.load('loss.npy')
    print(f'Finished')
    return loss


def submitit_job(parameters):
    # Directory for Submitit logs
    executor = AutoExecutor(folder="submitit_jobs")
    executor.update_parameters(
        timeout_min=60,
        cpus_per_task=4,  # Request 4 CPUs per job
        mem_gb=16,  # Request 16 GB of memory per job
        nodes=1,  # Number of nodes to request
        gpus_per_node=1,  # Number of GPUs if required
    )

    job = executor.submit(train_model, parameters)
    return job

# Assuming ax_client is already initialized as shown in step 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/data.npy')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dynamics_path', type=str, default='As.npy')
    parser.add_argument('--state_path', type=str, default='states.npy')
    args = parser.parse_args()
    print(args)
    main(args)
