import submitit

# Initialize the Submitit executor (using Slurm as an example)
executor = submitit.AutoExecutor(folder="submitit_logs")

# Set up multi-node, multi-task configurations
executor.update_parameters(
    nodes=1,  # Number of nodes
    tasks_per_node=4,  # Number of tasks per node (GPUs, etc.)
    timeout_min=300,  # Max duration of the job in minutes
    cpus_per_task=4,
    mem_gb=32,
    gpus_per_node=1,  # Number of GPUs per node
)

# Function to be executed on each node


def launch_training():
    import wandb
    import os
    os.system(
        "python -m wandb agent lianaakobian-university-of-vienna/PROTOTYPE_2_State_With_Control_All_True/zgyzaxbc")
    # Initialize W&B Sweep agent to pull hyperparameter configs
    # wandb.agent("4a0rzujk", function=main, count=4)  # count=4 for 4 tasks per node


# Submit the jobs
job = executor.submit(launch_training)
