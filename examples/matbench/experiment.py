from torch_geometric.data import Data
import torch
from torch_geometric.transforms import RadiusGraph
import json
import os
from tqdm import tqdm
import numpy as np

import hydragnn
from hydragnn.utils.print.print_utils import log
from hydragnn.preprocess.load_data import split_dataset

# from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg
from hydragnn.utils.model import load_existing_model, save_model
from hydragnn.utils.distributed import get_device


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


def run_model(model_type, dataset_name, config, train, val, test):
    verbosity = config["Verbosity"]["level"]
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type

    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    log_name = f"matbench_test_{dataset_name}_{model_type}"
    hydragnn.utils.print.print_utils.setup_log(log_name)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Run training with the given model and qm9 datasets.
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
    )

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    load_existing_model(model, log_name, path="./logs/")
    model.eval()
    device = get_device()

    test_true = []
    test_pred = []
    # Iterate over the test set
    for data_id, data in enumerate(tqdm(test)):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)[0]
        test_pred.append(pred.item())
        test_true.append(data.y.item())

    # Calculate Error
    scaled_error = compute_scaled_error(test_true, test_pred)
    print("Scaled Error:", scaled_error)

    return scaled_error


def run_experiment():
    # Make sure to clean logs directory
    os.system("rm -rf logs")

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

    num_repeats = 1
    model_types = [
        "CGCNN",
        "EGNN",
        "GIN",
        "GAT",
        "MFC",
        "PNA",
        "SAGE",
        "SchNet",
        "DimeNet",
        "PNAPlus",
        "PNAEq",
        "PAINN",
        "MACE",
    ]
    results = {model: [] for model in model_types}

    # Read Config
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matbench.json")
    with open(filename, "r") as f:
        config = json.load(f)

    # Run for all possible tasks
    for dataset_name in structure_datasets:
        print(f"\n--- Dataset: {dataset_name} ---")
        # Get data
        # Run Experiment
        for repeat in range(num_repeats):
            print(f"\n--- Repetition {repeat + 1} ---")
            dataset = torch.load(f"processed_{dataset_name}_train_val.pt")
            train, val, test = hydragnn.preprocess.split_dataset(
                dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
            )
            batch_size = int(np.sqrt(len(train)) - 16)
            learning_rate = 0.00001 * np.sqrt(len(train))
            config["NeuralNetwork"]["Training"]["batch_size"] = batch_size
            config["NeuralNetwork"]["Training"]["Optimizer"][
                "learning_rate"
            ] = learning_rate
            # Run dataset for all models
            for model_type in model_types:
                print(f"\nRunning model: {model_type}")
                # hidden_dim = 40 if model_type in ["PNAEq", "PAINN", "MACE"] else 120
                # config["NeuralNetwork"]["Architecture"]["hidden_dim"] = hidden_dim
                scaled_error = run_model(
                    model_type, dataset_name, config, train, val, test
                )
                results[model_type].append(
                    {
                        "dataset": dataset_name,
                        "repetition": repeat + 1,
                        "Scaled Error": scaled_error,
                    }
                )

    # Save results to file
    output_file = "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    run_experiment()
