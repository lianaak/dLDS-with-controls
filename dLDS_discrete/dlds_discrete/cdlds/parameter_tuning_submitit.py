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

    ax_client = AxClient()
    ax_client.create_experiment(
        name="dLDS_hyperparameter_tuning",
        parameters=[
            # {"name": "reg", "type": "range", "bounds": [
            #    0.00001, 10.00001], "log_scale": True},
            {"name": "smooth", "type": "range", "bounds": [
                0.00001, 10.00001], "log_scale": True},
            {"name": "num_subdyn", 'type': 'choice',
                'values': [2, 3, 4]},
            {"name": "learning_rate", "type": "range", "bounds": [
                0.00001, 0.1], "log_scale": True},
            {"name": "epochs", "type": "range", "bounds": [
                50, 100], "value_type": "int"},
            {'name': 'batch_size', 'type': 'choice',
                'values': [16, 32, 64, 128]},
            {'name': 'sigma', 'type': 'range', 'bounds': [
                0.0001, 1.0], 'log_scale': True}
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
            job = submitit_job(parameters, args)
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


def train_model(parameters, args):

    fix_point_change = args.fix_point_change
    eigenvalue_radius = args.eigenvalue_radius

    num_true_subdyn = parameters['num_subdyn']

    generator = DLDSwithControl(CdLDSDataGenerator(
        K=num_true_subdyn, D_control=0, fix_point_change=fix_point_change, eigenvalue_radius=float(eigenvalue_radius)))

    time_points = 1000

    # generate toy data
    X = generator.datasets.generate_data(time_points, sigma=0.01)

    # z-score normalization
    # X = (X - X.mean(axis=0)) / X.std(axis=0)

    run_id = np.random.randint(0, 1000)

    # save the data for each job
    if args.data_path is None:
        data_path = f'data_{run_id}.npy'
    else:
        data_path = args.data_path
    np.save(data_path, X)

    if args.dynamics_path is None:
        dynamics_path = f'As_{run_id}.npy'
    else:
        dynamics_path = args.dynamics_path
    # save the As of the model
    np.save(dynamics_path, np.array(generator.datasets.A))

    if args.state_path is None:
        state_path = f'states_{run_id}.npy'
    else:
        state_path = args.state_path

    np.save(state_path, generator.datasets.z_)

    # ' --reg ' + str(parameters['reg']) + \

    command = 'python train.py --data_path ' + data_path + \
        ' --smooth ' + str(parameters['smooth']) + ' --epochs ' + \
        str(parameters['epochs']) + ' --lr ' + str(parameters['learning_rate']) + \
        ' --num_subdyn ' + str(parameters['num_subdyn']) + ' --dynamics_path ' + \
        dynamics_path + ' --state_path ' + state_path + \
        ' --fix_point_change ' + \
        str(fix_point_change) + ' --eigenvalue_radius ' + str(eigenvalue_radius) + \
        ' --batch_size ' + \
        str(parameters['batch_size']) + ' --sigma ' + str(parameters['sigma'])
    os.system(command)
    # get the loss
    loss = np.load('loss.npy')
    print(f'Finished')
    return loss


def submitit_job(parameters, fix_point_change=False):
    # Directory for Submitit logs
    executor = AutoExecutor(folder="submitit_jobs")
    executor.update_parameters(
        timeout_min=60,
        cpus_per_task=4,  # Request 4 CPUs per job
        mem_gb=32,  # Request 16 GB of memory per job
        nodes=1,  # Number of nodes to request
        gpus_per_node=1,  # Number of GPUs if required
    )

    job = executor.submit(train_model, parameters, fix_point_change)
    return job

# Assuming ax_client is already initialized as shown in step 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dynamics_path', type=str, default=None)
    parser.add_argument('--state_path', type=str, default=None)
    parser.add_argument('--fix_point_change', type=str, default=False)
    parser.add_argument('--eigenvalue_radius', type=str, default=0.99)
    args = parser.parse_args()
    print(args)
    main(args)
