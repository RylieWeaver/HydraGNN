# General
import os, sys, json
import logging

# Torch
import torch
import torch_geometric

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

# HydraGNN
import hydragnn
from hydragnn.utils.uncertainty_utils import (
    get_minmax_scaling_parameters,
    minmax_scale_data,
    minmax_scale_dataset,
    save_dataset,
    save_scaling,
    load_scaling,
    rotate_data,
    rotate_dataset,
)

# HPO
import optuna
import pandas as pd

# FIX random seed
random_state = 0
torch.manual_seed(random_state)


# Update each sample prior to loading.
def md17_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    data.forces = data.force
    # Only predict energy (index 0 of 2 properties) for this run.
    # data.y = data.energy / len(data.x)
    data.y = torch.cat(
        [data.energy.view(-1, 1), data.forces.flatten().unsqueeze(-1)], dim=0
    )
    data.y_loc = torch.tensor([[0, 1, 1 + data.num_nodes * 3]])
    # graph_features_dim = [1]
    # node_feature_dim = [1]
    data = compute_edges(data)
    data = rotate_data(data)
    # NOTE  It's important to have data from various orientations in order to truly represent energy/force predictions.
    #       I was getting a R2 score of 0.6 with invariant prediction, which indicates that the data wasn't representatice
    #       of the true distribution.
    return data


def md17_pre_filter(data):
    # return torch.rand(1) < 0.01
    return True


def objective(trial):
    global config, best_trial_id, best_validation_loss

    # Extract the unique trial ID
    trial_id = trial.number  # or trial_id = trial.trial_id

    log_name = "md17_optuna"
    log_name = log_name + "_" + str(trial_id)
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    # log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    # Define the search space for hyperparameters
    ## Model Parameters
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
    ## Learning Parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_int("batch_size", 8, 256)

    # Update the config dictionary with the suggested hyperparameters
    # Model Parameters
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
    # config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"][
    #     "num_headlayers"
    # ] = num_headlayers
    # config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"][
    #     "dim_headlayers"
    # ] = dim_headlayers
    ## Learning Parameters
    config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"] = learning_rate
    config["NeuralNetwork"]["Training"]["batch_size"] = batch_size

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        train, val, test, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    compute_grad_energy = config["NeuralNetwork"]["Training"]["compute_grad_energy"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001
    )

    writer = hydragnn.utils.model.model.get_summary_writer(log_name)
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    ##################################################################################################################

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
        create_plots=False,
        compute_grad_energy=compute_grad_energy,
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)

    """
    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()
    """

    # Return the metric to minimize (e.g., validation loss)
    validation_loss, tasks_loss = hydragnn.train.validate(
        val_loader,
        model,
        verbosity,
        reduce_ranks=True,
        compute_grad_energy=compute_grad_energy,
    )

    # Move validation_loss to the CPU and convert to NumPy object
    validation_loss = validation_loss.cpu().detach().numpy()

    # Append trial results to the DataFrame
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

    # Update information about the best trial
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_trial_id = trial_id

    # Return the metric to minimize (e.g., validation loss)
    return validation_loss


if __name__ == "__main__":
    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    path = os.environ.get("SERIALIZED_DATA_PATH")

    # Configurable run choices (JSON file that accompanies this example script).
    filename = os.path.join(os.path.dirname(__file__), "md17_uncertainty.json")
    with open(filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    arch_config = config["NeuralNetwork"]["Architecture"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()

    log_name = "md17_test"
    # Enable print to log file.
    hydragnn.utils.print.setup_log(log_name)

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

    # Check if split is already done
    if not (
        os.path.exists(os.path.join(path, "train.pt"))
        and os.path.exists(os.path.join(path, "val.pt"))
        and os.path.exists(os.path.join(path, "test.pt"))
    ):
        # Fix for MD17 datasets
        torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"

        dataset = torch_geometric.datasets.MD17(
            root="dataset/md17",
            name="uracil",
            pre_transform=md17_pre_transform,
            pre_filter=md17_pre_filter,
        )
        # dataset = dataset[:20000]
        train, val, test = hydragnn.preprocess.split_dataset(
            dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
        )
        # NOTE that we're saving BEFORE scaling
        save_dataset(path, train, "train")
        save_dataset(path, val, "val")
        save_dataset(path, test, "test")

        train_energy_min, train_energy_max = get_minmax_scaling_parameters(train)
        save_scaling(
            os.path.join(path, "scaling.pt"), train_energy_min, train_energy_max
        )
        train, val, test = minmax_scale_dataset(
            train, val, test, train_energy_min, train_energy_max
        )
    # Else load the data (this will keep all our experiments with the same datasets)
    else:
        train = torch.load(os.path.join(path, "train.pt"))
        val = torch.load(os.path.join(path, "val.pt"))
        test = torch.load(os.path.join(path, "test.pt"))
        train_energy_min, train_energy_max = load_scaling(
            os.path.join(path, "scaling.pt")
        )
        train, val, test = minmax_scale_dataset(
            train, val, test, train_energy_min, train_energy_max
        )

    # Choose the sampler (e.g., TPESampler or RandomSampler)
    sampler = optuna.samplers.TPESampler(consider_prior=True, consider_magic_clip=False)
    # sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.CmaEsSampler(cma_stds=[1.0, 1.0, 1.0], consider_pruned_trials=True, consider_prior=False)
    # sampler = optuna.samplers.GridSampler(consider_prior=False)
    # sampler = optuna.samplers.NSGAIISampler(pop_size=100, crossover_prob=0.9, mutation_prob=0.1)

    # Create an empty DataFrame to store trial results
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

    # Variables to store information about the best trial
    best_trial_id = None
    best_validation_loss = float("inf")

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, n_jobs=4)

    # Update the best trial information directly within the DataFrame
    best_trial_info = pd.Series(
        {"Trial_ID": best_trial_id, "Best_Validation_Loss": best_validation_loss}
    )
    # trial_results = trial_results.append(best_trial_info, ignore_index=True)  # Deprecated
    trial_results = pd.concat(
        [trial_results, best_trial_info.to_frame().T], ignore_index=True
    )

    # Save the trial results to a CSV file
    trial_results.to_csv("hpo_results.csv", index=False)

    # Get the best hyperparameters and corresponding trial ID
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    best_trial_id = (
        study.best_trial.number
    )  # or best_trial_id = study.best_trial.trial_id

    # Print information about the best trial
    print("Best Trial ID:", best_trial_id)
    print("Best Validation Loss:", best_validation_loss)

    sys.exit(0)
