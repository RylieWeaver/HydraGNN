import os
import pdb
import json
import logging
import argparse
from tqdm import tqdm
import numpy as np

import torch

import torch_geometric
from torch_geometric.data import Data
# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

import hydragnn
from hydragnn.utils.print.print_utils import log
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.utils.model import load_existing_model, save_model
from hydragnn.utils.distributed import get_device

torch.backends.cudnn.enabled = False


def compute_scaled_error(true_values, pred_values):
    """
    Compute the Scaled Error as the ratio of MAE to MAD.
    Scaled Error = MAE / MAD
    """
    true_values = torch.tensor(true_values)
    pred_values = torch.tensor(pred_values)

    # Compute Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(true_values - pred_values)).item()

    # Compute Mean Absolute Deviation (MAD)
    mean_value = true_values.mean()
    mad = torch.mean(torch.abs(true_values - mean_value)).item()

    # Handle edge case where MAD is 0
    if mad == 0:
        return float("inf")  # Or define another fallback value

    # Compute Scaled Error
    scaled_error = mae / mad
    return scaled_error


log_name = "matbench_hpo_trials"


# Configurable run choices (JSON file that accompanies this example script).
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matbench.json")
with open(filename, "r") as f:
    config = json.load(f)
verbosity = config["Verbosity"]["level"]

# Preprocess configurations for edge computation
compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)


dataset = torch.load(f"processed_{dataset_name}_train_val.pt")
train, val, test = hydragnn.preprocess.split_dataset(
    dataset, config["NeuralNetwork"]["Training"]["perc_train"]
)



def run(trial):

    global config

    trial_config = config

    # Always initialize for multi-rank training.
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

    # Update the config dictionary with the suggested hyperparameters
    trial_config["NeuralNetwork"]["Architecture"]["mpnn_type"] = trial.parameters[
        "mpnn_type"
    ]
    trial_config["NeuralNetwork"]["Architecture"]["hidden_dim"] = trial.parameters["hidden_dim"]
    trial_config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = trial.parameters[
        "num_conv_layers"
    ]

    dim_headlayers = [
        trial.parameters["dim_headlayers"]
        for i in range(trial.parameters["num_headlayers"])
    ]

    for head_type in trial_config["NeuralNetwork"]["Architecture"]["output_heads"]:
        trial_config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "num_headlayers"
        ] = trial.parameters["num_headlayers"]
        trial_config["NeuralNetwork"]["Architecture"]["output_heads"][head_type][
            "dim_headlayers"
        ] = dim_headlayers

    trial_config = hydragnn.utils.input_config_parsing.update_config(
        trial_config, train_loader, val_loader, test_loader
    )

    hydragnn.utils.input_config_parsing.save_config(trial_config, trial_log_name)

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

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
        optimizer, mode="min", factor=0.5, patience=5, min_lr=(learning_rate/1e3)
    )

    hydragnn.utils.model.model.load_existing_model_config(
        model, trial_config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################
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
        compute_grad_energy=False,
    )

    hydragnn.utils.model.model.save_model(model, optimizer, trial_log_name)
    hydragnn.utils.print.print_distributed(verbosity)

    # Return the metric to minimize (e.g., validation loss)
    validation_loss, tasks_loss = hydragnn.train.validate(
        val_loader, model, verbosity, reduce_ranks=True
    )

    # Move validation_loss to the CPU and convert to NumPy object
    validation_loss = validation_loss.cpu().detach().numpy()

    # Return the metric to minimize (e.g., validation loss)
    # By default, DeepHyper maximized the objective function, so we need to flip the sign of the validation loss function
    print("validation_loss.item()", validation_loss.item())
    return -validation_loss.item()
   

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=(learning_rate/1e3)
    )

    # Run training with the given model and md17 dataset.
    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        compute_grad_energy=False,
    )


if __name__ == "__main__":
    
    # Specify the list of datasets
    structure_datasets = [
        "matbench_mp_e_form",
        "matbench_mp_gap",
        "matbench_perovskites",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
        "matbench_dielectric",
        "matbench_phonons",
        "matbench_jdft2d",
    ]
    composition_datasets = [
        "matbench_glass",
        "matbench_expt_is_metal",
        "matbench_expt_gap",
        "matbench_steels",
    ]
    mixed_datasets = []
    classification_structure_datasets = ["matbench_mp_is_metal"]

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator

    # define the variable you want to optimize
    problem = HpProblem()

    # Define the search space for hyperparameters
    problem.add_hyperparameter((1, 4), "num_conv_layers")  # discrete parameter
    problem.add_hyperparameter((1, 100), "hidden_dim")  # discrete parameter
    problem.add_hyperparameter((1, 3), "num_headlayers")  # discrete parameter
    problem.add_hyperparameter((1, 3), "dim_headlayers")  # discrete parameter

    # Define the search space for hyperparameters
    # define the evaluator to distribute the computation
    parallel_evaluator = Evaluator.create(
        run,
        method="process",
        method_kwargs={
            "num_workers": 1,
        },
    )

    # define your search and execute it
    search = CBO(problem, parallel_evaluator, random_state=42, log_dir=log_name)

    timeout = 1200
    results = search.search(max_evals=10, timeout=timeout)
    print(results)

    sys.exit(0)
    
    
    
    
    
    
def matbench_name_info():
    matbench_ranking_plot_mapping = {
        # Structure Datasets
        "Ef Steel alloys": "matbench_steels",  # Steel alloy properties
        "Eg 2D Materials": "matbench_jdft2d",  # 2D exfoliation energies
        "ωmax Phonons": "matbench_phonons",  # Phonon density of states
        "Eg Experimental": "matbench_dielectric",  # Dielectric properties
        "log10 Gvrh": "matbench_log_gvrh",  # Logarithm of shear modulus
        "log10 Kvrh": "matbench_log_kvrh",  # Logarithm of bulk modulus
        "Ef Perovskites, DFT": "matbench_perovskites",  # Stability of perovskites
        "Eg DFT": "matbench_mp_gap",  # Band gap
        "Ef DFT": "matbench_mp_e_form",  # Formation energy
        # Composition Datasets
        "Expt. Metallicity Classification": "matbench_expt_is_metal",  # Experimental metallicity classification
        "Metallicity DFT": "matbench_glass",  # Metallic glass classification
        "Eg Experimental": "matbench_expt_gap",  # Experimental band gap
        "Ef Steel alloys": "matbench_steels",  # Steel alloys dataset
    }
    print(matbench_ranking_plot_mapping)
    return matbench_ranking_plot_mapping



# Questions
## Where does it get config?