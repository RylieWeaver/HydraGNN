# General
import os, sys, json
import logging
import time
from datetime import datetime

# Torch
import torch
import torch_geometric

# Deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

# HydraGNN
import hydragnn
from hydragnn.utils.scaling_utils import (
    get_minmax_scaling_parameters,
    minmax_scale_dataset,
    save_dataset,
    save_scaling,
    load_scaling,
)

# HPO
import optuna
import pandas as pd

# Disable cuDNN when running on ROCm
torch.backends.cudnn.enabled = False

# Fix random seed for reproducibility
random_state = 0
torch.manual_seed(random_state)


def md17_pre_transform(data):
    """Update each sample prior to loading."""
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    data.forces = data.force
    # Only predict energy (index 0 of 2 properties) for this run.
    data.y = torch.cat(
        [data.energy.view(-1, 1), data.force.flatten().unsqueeze(-1)], dim=0
    )
    data.y_loc = torch.tensor([[0, 1, 1 + data.num_nodes * 3]])
    data = compute_edges(data)
    return data


def md17_pre_filter(data):
    """Optional filter for dataset samples."""
    # return torch.rand(1) < 0.01
    return True


def objective(trial):
    """
    Objective function for Optuna. Sets up a HydraGNN model with hyperparameters,
    performs training/validation, and returns validation loss.
    """
    global config, best_trial_id, best_validation_loss

    # Extract the unique trial ID
    trial_id = trial.number

    # Log the start of the trial
    start_time = datetime.now()
    logging.info(
        f"=== Trial {trial_id} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ==="
    )

    log_name = f"md17_optuna_{trial_id}"
    hydragnn.utils.print.setup_log(log_name)

    # Additional logging config (to ensure logs are shown even if HydraGNN redirects)
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    # --------------------------------------------------------------------------------
    # Define the search space for hyperparameters
    # --------------------------------------------------------------------------------
    hidden_dim = trial.suggest_int("hidden_dim", 20, 300)
    int_emb_size = trial.suggest_int("int_emb_size", 20, 80)
    out_emb_size = trial.suggest_int("out_emb_size", 10, 40)
    num_before_skip = trial.suggest_int("num_before_skip", 1, 3)
    num_after_skip = trial.suggest_int("num_after_skip", 1, 3)
    basis_emb_size = trial.suggest_int("basis_emb_size", 5, 20)
    num_radial = trial.suggest_int("num_radial", 3, 12)
    num_spherical = trial.suggest_int("num_spherical", 2, 8)
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 5)
    num_headlayers = trial.suggest_int("num_headlayers", 1, 3)
    dim_headlayers = [
        trial.suggest_int(f"dim_headlayer_{i}", 20, 200) for i in range(num_headlayers)
    ]
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 8, 256)

    # --------------------------------------------------------------------------------
    # Update the config dictionary with the suggested hyperparameters
    # --------------------------------------------------------------------------------
    config["NeuralNetwork"]["Architecture"]["hidden_dim"] = hidden_dim
    config["NeuralNetwork"]["Architecture"]["int_emb_size"] = int_emb_size
    config["NeuralNetwork"]["Architecture"]["out_emb_size"] = out_emb_size
    config["NeuralNetwork"]["Architecture"]["num_before_skip"] = num_before_skip
    config["NeuralNetwork"]["Architecture"]["num_after_skip"] = num_after_skip
    config["NeuralNetwork"]["Architecture"]["basis_emb_size"] = basis_emb_size
    config["NeuralNetwork"]["Architecture"]["num_radial"] = num_radial
    config["NeuralNetwork"]["Architecture"]["num_spherical"] = num_spherical
    config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = num_conv_layers
    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"][
        "num_headlayers"
    ] = num_headlayers
    config["NeuralNetwork"]["Architecture"]["output_heads"]["node"][
        "dim_headlayers"
    ] = dim_headlayers

    config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"] = learning_rate
    config["NeuralNetwork"]["Training"]["batch_size"] = batch_size

    # Create loaders
    train_loader, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    # Update config with loader info
    hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    # Create and distribute model
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    # Define optimizer, scheduler
    compute_grad_energy = config["NeuralNetwork"]["Training"]["compute_grad_energy"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # Save updated config
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    # --------------------------------------------------------------------------------
    # Train / Validate / Test
    # --------------------------------------------------------------------------------
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        hydragnn.utils.model.get_summary_writer(log_name),
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
        compute_grad_energy=compute_grad_energy,
    )

    # Save the model and log timings
    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)

    # Evaluate validation loss
    validation_loss, tasks_loss = hydragnn.train.validate(
        val_loader,
        model,
        verbosity,
        reduce_ranks=True,
        compute_grad_energy=compute_grad_energy,
    )
    validation_loss = validation_loss.cpu().detach().numpy()

    # Store trial results in our DataFrame
    trial_results.loc[trial_id] = [
        trial_id,
        config["NeuralNetwork"]["Architecture"]["model_type"],
        hidden_dim,
        int_emb_size,
        out_emb_size,
        num_before_skip,
        num_after_skip,
        basis_emb_size,
        num_radial,
        num_spherical,
        num_conv_layers,
        num_headlayers,
        dim_headlayers,
        learning_rate,
        batch_size,
        validation_loss,
    ]

    # Update best trial tracking
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_trial_id = trial_id

    # Log the end of the trial
    end_time = datetime.now()
    logging.info(
        f"=== Trial {trial_id} finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"with validation_loss={validation_loss:.6f} ==="
    )

    return validation_loss


if __name__ == "__main__":
    # Prepare environment
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except KeyError:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    path = os.environ["SERIALIZED_DATA_PATH"]

    filename = os.path.join(os.path.dirname(__file__), "md17.json")
    with open(filename, "r") as f:
        config = json.load(f)

    verbosity = config["Verbosity"]["level"]
    arch_config = config["NeuralNetwork"]["Architecture"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Initialize distributed environment (for multi-rank training of each trial)
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()

    log_name = "md17_test"
    hydragnn.utils.print.setup_log(log_name)

    # Prepare pre_transform function
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

    # --------------------------------------------------------------------------------
    # Load or create dataset splits
    # --------------------------------------------------------------------------------
    if not (
        os.path.exists(os.path.join(path, "train.pt"))
        and os.path.exists(os.path.join(path, "val.pt"))
        and os.path.exists(os.path.join(path, "test.pt"))
    ):
        # Fix for MD17 datasets naming
        torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"

        dataset = torch_geometric.datasets.MD17(
            root="dataset/md17",
            name="uracil",
            pre_transform=md17_pre_transform,
            pre_filter=md17_pre_filter,
        )

        # Split dataset
        train, val, test = hydragnn.preprocess.split_dataset(
            dataset, config["NeuralNetwork"]["Training"]["perc_train"]
        )
        # Save the raw splits
        save_dataset(path, train, "train")
        save_dataset(path, val, "val")
        save_dataset(path, test, "test")

        # Compute and save scaling parameters
        train_energy_min, train_energy_max = get_minmax_scaling_parameters(train)
        save_scaling(
            os.path.join(path, "scaling.pt"), train_energy_min, train_energy_max
        )

        # Scale all subsets
        train, val, test = minmax_scale_dataset(
            train, val, test, train_energy_min, train_energy_max
        )
    else:
        # Load existing splits
        train = torch.load(os.path.join(path, "train.pt"))
        val = torch.load(os.path.join(path, "val.pt"))
        test = torch.load(os.path.join(path, "test.pt"))

        # Load existing scaling
        train_energy_min, train_energy_max = load_scaling(
            os.path.join(path, "scaling.pt")
        )

        # Scale all subsets
        train, val, test = minmax_scale_dataset(
            train, val, test, train_energy_min, train_energy_max
        )

    # --------------------------------------------------------------------------------
    # Set up Optuna
    # --------------------------------------------------------------------------------
    sampler = optuna.samplers.TPESampler(consider_prior=True, consider_magic_clip=False)

    # Create a global DataFrame and best-trial trackers
    trial_results = pd.DataFrame(
        columns=[
            "Trial_ID",
            "Model_Type",
            "Hidden_Dim",
            "Int_Emb_Size",
            "Out_Emb_Size",
            "Num_Before_Skip",
            "Num_After_Skip",
            "Basis_Emb_Size",
            "Num_Radial",
            "Num_Spherical",
            "Num_Conv_Layers",
            "Num_Headlayers",
            "Dim_Headlayers",
            "Learning_Rate",
            "Batch_Size",
            "Validation_Loss",
        ]
    )

    best_trial_id = None
    best_validation_loss = float("inf")

    # Create a study object and run the optimization
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # You can adjust n_jobs for parallel trials (local concurrency)
    study.optimize(objective, n_trials=50, n_jobs=4)

    # Store info about the best trial
    best_trial_info = pd.Series(
        {"Trial_ID": best_trial_id, "Best_Validation_Loss": best_validation_loss}
    )
    trial_results = pd.concat(
        [trial_results, best_trial_info.to_frame().T], ignore_index=True
    )

    # Save results to CSV
    trial_results.to_csv("hpo_results.csv", index=False)

    # Print best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    best_trial_id = study.best_trial.number
    print("Best Trial ID:", best_trial_id)
    print("Best Validation Loss:", best_validation_loss)

    sys.exit(0)
