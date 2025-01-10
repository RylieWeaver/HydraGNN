##################################################################################################################
# IMPORTS
##################################################################################################################
import os
import pdb
import json
import logging
import torch
import torch_geometric
import argparse

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn

# torch.backends.cudnn.enabled = False


##################################################################################################################
# PREPROCESS MD17
##################################################################################################################
# Update each sample prior to loading.
def md17_pre_transform(data, compute_edges):
    """Update each sample prior to loading."""
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    data.forces = data.force
    # Only predict energy (index 0 of 2 properties) for this run.
    data.y = torch.cat(
        [data.energy.view(-1, 1), data.forces.flatten().unsqueeze(-1)], dim=0
    )
    data.y_loc = torch.tensor([[0, 1, 1 + data.num_nodes * 3]])
    data = compute_edges(data)
    return data


# Randomly select ~1000 samples
def md17_pre_filter(data):
    return torch.rand(1) < 1.1


##################################################################################################################
# RUN DEEPHYPER TRIAL
##################################################################################################################
def run(trial):

    ########## INITIALIZATION ##########
    ## Config
    global config
    trial_config = config
    ## Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    trial_log_name = log_name + "_" + str(trial.id)
    hydragnn.utils.print.print_utils.setup_log(trial_log_name)
    writer = hydragnn.utils.model.model.get_summary_writer(trial_log_name)
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )
    # log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    ######### HYPERPARAMETER SPACE ##########
    ## Model-specific parameters
    trial_config["NeuralNetwork"]["Architecture"]["hidden_dim"] = trial.parameters[
        "hidden_dim"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = trial.parameters[
        "num_conv_layers"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["basis_emb_size"] = trial.parameters[
        "basis_emb_size"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["int_emb_size"] = trial.parameters[
        "int_emb_size"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["out_emb_size"] = trial.parameters[
        "out_emb_size"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["num_before_skip"] = trial.parameters[
        "num_before_skip"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["num_after_skip"] = trial.parameters[
        "num_after_skip"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["num_radial"] = trial.parameters[
        "num_radial"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["num_spherical"] = trial.parameters[
        "num_spherical"
    ]
    ## Head-specific parameters
    dim_headlayers = [
        trial.parameters["dim_headlayers"]
        for _ in range(trial.parameters["num_headlayers"])
    ]
    for head_type in trial_config["NeuralNetwork"]["Architecture"]["output_heads"]:
        trial_config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "num_headlayers"
        ] = trial.parameters["num_headlayers"]
        trial_config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "dim_headlayers"
        ] = [trial.parameters["dim_headlayers"]] * trial.parameters["num_headlayers"]

    ## Training-related parameters
    trial_config["NeuralNetwork"]["Training"]["learning_rate"] = trial.parameters[
        "learning_rate"
    ]
    trial_config["NeuralNetwork"]["Training"]["batch_size"] = trial.parameters[
        "batch_size"
    ]

    ######### SETUP ##########
    ## Dataset
    (
        train_loader,
        val_loader,
        test_loader,
    ) = hydragnn.preprocess.create_dataloaders(
        train, val, test, trial_config["NeuralNetwork"]["Training"]["batch_size"]
    )
    ## Config
    trial_config = hydragnn.utils.input_config_parsing.update_config(
        trial_config, train_loader, val_loader, test_loader
    )
    ## Model / Training
    hydragnn.utils.input_config_parsing.save_config(trial_config, trial_log_name)
    model = hydragnn.models.create_model_config(
        config=trial_config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)
    learning_rate = trial_config["NeuralNetwork"]["Training"]["Optimizer"][
        "learning_rate"
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=(learning_rate / 1e3)
    )
    hydragnn.utils.model.model.load_existing_model_config(
        model, trial_config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ######### TRAINING ##########
    ## Train
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        trial_config["NeuralNetwork"],
        trial_log_name,
        verbosity,
        create_plots=False,
        compute_grad_energy=config["NeuralNetwork"]["Training"],
    )
    ## Save
    hydragnn.utils.model.model.save_model(model, optimizer, trial_log_name)
    hydragnn.utils.print.print_distributed(verbosity)
    ## Update metric (val loss)
    validation_loss, tasks_loss = hydragnn.train.validate(
        val_loader, model, verbosity, reduce_ranks=True
    )
    ## Show
    validation_loss = validation_loss.cpu().detach().numpy()
    print("validation_loss.item()", validation_loss.item())
    return (
        -validation_loss.item()
    )  # By default, DeepHyper maximizes the objective function

    ########### CLEANUP ##########
    # config = hydragnn.utils.input_config_parsing.update_config(
    #     config, train_loader, val_loader, test_loader
    # )

    # model = hydragnn.models.create_model_config(
    #     config=config["NeuralNetwork"],
    #     verbosity=verbosity,
    # )
    # model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    # learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=5, min_lr=(learning_rate/1e3)
    # )


1
# # Run training with the given model and md17 dataset.
# writer = hydragnn.utils.model.model.get_summary_writer(log_name)
# hydragnn.utils.input_config_parsing.save_config(config, log_name)

# hydragnn.train.train_validate_test(
#     model,
#     optimizer,
#     train_loader,
#     val_loader,
#     test_loader,
#     writer,
#     scheduler,
#     config["NeuralNetwork"],
#     log_name,
#     verbosity,
#     compute_grad_energy=config["NeuralNetwork"]["Training"],
# )


##################################################################################################################
# MAIN
##################################################################################################################
if __name__ == "__main__":
    ########### SETUP ##########
    ## Log
    log_name = "md17_hpo_trials"

    ## Config
    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "md17_DimeNet.json"
    )
    with open(filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]

    ## Data
    arch_config = config["NeuralNetwork"]["Architecture"]
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)
    torch_geometric.datasets.MD17.file_names["uracil"] = (
        "md17_uracil.npz"  # Fix for MD17 datasets
    )
    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name="uracil",
        pre_transform=lambda data: md17_pre_transform(data, compute_edges),
        pre_filter=md17_pre_filter,
    )
    train, val, test = hydragnn.preprocess.split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )

    ########### DEEPHYPER HPO ##########
    # DeepHyper
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator

    # Search space
    problem = HpProblem()  # Define the variable you want to optimize
    ## Model
    problem.add_hyperparameter((10, 100), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((1, 4), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((4, 20), "basis_emb_size")  # discrete parameter
    problem.add_hyperparameter((8, 40), "int_emb_size")  # discrete parameter
    problem.add_hyperparameter((4, 20), "out_emb_size")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_before_skip")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_after_skip")  # discrete parameter
    problem.add_hyperparameter((3, 9), "num_radial")  # discrete parameter
    problem.add_hyperparameter((3, 6), "num_spherical")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((10, 100), "dim_headlayers")  # discrete parameter
    ## Training
    problem.add_hyperparameter((1e-5, 1e-2), "learning_rate")  # continuous parameter
    problem.add_hyperparameter((8, 128), "batch_size")  # discrete parameter
    ## Not Tuned
    ### envelope_exponent, optimizer

    # Algorithm
    parallel_evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 1,
        },
    )
    search = CBO(problem, parallel_evaluator, random_state=42, log_dir=log_name)

    # Run
    timeout = 1200
    results = search.search(max_evals=10, timeout=timeout)
    print(results)

    sys.exit(0)
