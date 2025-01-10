import os
import torch
import numpy as np
import math
from torch_geometric.data import Data
from scipy.stats import pearsonr, spearmanr


def get_minmax_scaling_parameters(train):
    # Do minmax scaling from just the training set
    train_energy = torch.cat([data.energy for data in train])
    train_energy_min = train_energy.min()
    train_energy_max = train_energy.max()

    return train_energy_min, train_energy_max


def minmax_scale_data(data, train_energy_min, train_energy_max):
    data.energy = (data.energy - train_energy_min) / (
        train_energy_max - train_energy_min
    )
    data.forces = (data.forces) / (train_energy_max - train_energy_min)
    return data


def minmax_scale_dataset(
    train, val, test, train_energy_min=None, train_energy_max=None
):
    # Do minmax scaling from just the training set
    if train_energy_min is None or train_energy_max is None:
        train_energy_min, train_energy_max = get_scaling_parameters(train)

    # Scale the energy values
    for dataset in [train, val, test]:
        for data in dataset:
            data = minmax_scale_data(data, train_energy_min, train_energy_max)

    return train, val, test


def save_dataset(path, split, split_name):
    save_path = os.path.join(path, f"{split_name}_dataset.pt")
    torch.save(split, save_path)


def save_scaling(path, train_energy_min, train_energy_max):
    torch.save(
        {"train_energy_min": train_energy_min, "train_energy_max": train_energy_max},
        path,
    )


def load_scaling(path):
    scaling = torch.load(path)
    return scaling["train_energy_min"], scaling["train_energy_max"]


def reverse_minmax_scale_data(
    energy_pred,
    energy_true,
    forces_pred_direct,
    forces_pred_grad,
    forces_true,
    train_energy_min,
    train_energy_max,
):
    energy_pred = energy_pred * (train_energy_max - train_energy_min) + train_energy_min
    energy_true = energy_true * (train_energy_max - train_energy_min) + train_energy_min
    forces_pred_direct = forces_pred_direct * (train_energy_max - train_energy_min)
    forces_pred_grad = forces_pred_grad * (train_energy_max - train_energy_min)
    forces_true = forces_true * (train_energy_max - train_energy_min)
    return energy_pred, energy_true, forces_pred_direct, forces_pred_grad, forces_true


def rotate_data(data):
    """
    Rotate the positions and forces in data.pos by a random angle/axis.
    """
    # Convert angle to radians
    angle_degrees = np.random.uniform(0, 360)
    angle_radians = math.radians(angle_degrees)

    # Define rotation matrix based on the axis
    axis = np.random.choice(["x", "y", "z"])
    if axis == "x":
        rotation_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, math.cos(angle_radians), -math.sin(angle_radians)],
                [0, math.sin(angle_radians), math.cos(angle_radians)],
            ]
        )
    elif axis == "y":
        rotation_matrix = torch.tensor(
            [
                [math.cos(angle_radians), 0, math.sin(angle_radians)],
                [0, 1, 0],
                [-math.sin(angle_radians), 0, math.cos(angle_radians)],
            ]
        )
    elif axis == "z":
        rotation_matrix = torch.tensor(
            [
                [math.cos(angle_radians), -math.sin(angle_radians), 0],
                [math.sin(angle_radians), math.cos(angle_radians), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # Rotate positions
    data.pos = torch.matmul(data.pos, rotation_matrix.T)
    data.forces = torch.matmul(data.forces, rotation_matrix.T)

    return data


def rotate_dataset(dataset):
    """
    Rotate the positions in a dataset of PyTorch Geometric data objects by a specified angle around a given axis.

    Args:
        dataset: A PyTorch Geometric dataset.
        angle_degrees (float): The rotation angle in degrees.
        axis (str): The axis of rotation ('x', 'y', or 'z').

    Returns:
        rotated_dataset: A new PyTorch Geometric dataset with rotated positions.
    """
    for data in dataset:
        data = rotate_data(data)

    return dataset


def calculate_metrics(dataset):
    energy_mse_list = []
    force_direct_mse_list = []
    force_grad_mse_list = []
    uncertainty_min_list = []
    uncertainty_max_list = []
    uncertainty_ma_list = []
    uncertainty_ms_list = []
    uncertainty_mc_list = []
    uncertainty_log_list = []
    uncertainty_exp_list = []

    # Get metrics for each data point
    for data in dataset:
        # Extract values
        energy_true = data.energy.cpu()
        energy_pred = data.energy_pred.cpu()
        forces_true = data.forces.cpu()
        forces_pred_direct = data.forces_pred_direct.cpu()
        forces_pred_grad = data.forces_pred_grad.cpu()

        # Convert to flat numpy arrays
        energy_true = np.array(energy_true)
        energy_pred = np.array(energy_pred)
        forces_true = np.array(forces_true.flatten().tolist())
        forces_pred_direct = np.array(forces_pred_direct.flatten().tolist())
        forces_pred_grad = np.array(forces_pred_grad.flatten().tolist())

        # Metrics
        energy_mse = np.mean((energy_pred - energy_true) ** 2)
        force_direct_mse = np.mean((forces_pred_direct - forces_true) ** 2)
        force_grad_mse = np.mean((forces_pred_grad - forces_true) ** 2)
        uncertainty_min = np.min(np.abs(forces_pred_direct - forces_pred_grad))
        uncertainty_max = np.max(np.abs(forces_pred_direct - forces_pred_grad))
        uncertainty_ma = np.mean(np.abs(forces_pred_direct - forces_pred_grad))
        uncertainty_ms = np.mean((forces_pred_direct - forces_pred_grad) ** 2)
        uncertainty_mc = np.mean(np.abs(forces_pred_direct - forces_pred_grad) ** 3)
        uncertainty_log = np.mean(np.log(np.abs(forces_pred_direct - forces_pred_grad)))
        uncertainty_exp = np.mean(np.exp(np.abs(forces_pred_direct - forces_pred_grad)))

        # Append to lists
        energy_mse_list.append(energy_mse)
        force_direct_mse_list.append(force_direct_mse)
        force_grad_mse_list.append(force_grad_mse)
        uncertainty_min_list.append(uncertainty_min)
        uncertainty_max_list.append(uncertainty_max)
        uncertainty_ma_list.append(uncertainty_ma)
        uncertainty_ms_list.append(uncertainty_ms)
        uncertainty_mc_list.append(uncertainty_mc)
        uncertainty_log_list.append(uncertainty_log)
        uncertainty_exp_list.append(uncertainty_exp)

    return (
        energy_mse_list,
        force_direct_mse_list,
        force_grad_mse_list,
        uncertainty_min_list,
        uncertainty_max_list,
        uncertainty_ma_list,
        uncertainty_ms_list,
        uncertainty_mc_list,
        uncertainty_log_list,
        uncertainty_exp_list,
    )


def analyze_correlations(
    energy_mse_list,
    force_direct_mse_list,
    force_grad_mse_list,
    uncertainty_min_list,
    uncertainty_max_list,
    uncertainty_ma_list,
    uncertainty_ms_list,
    uncertainty_mc_list,
    uncertainty_log_list,
    uncertainty_exp_list,
):
    """
    Perform Pearson and Spearman correlation analysis between MSE and uncertainty metrics.
    """

    def calculate_and_print_correlations(corr_func, corr_type):
        print(f"=== {corr_type} Correlations ===")
        print_corr = lambda name, x, y: print(
            f"{name} {corr_type} Correlation: {corr_func(x, y)[0]:.4f}"
        )

        print_corr("Energy Uncertainty Min", energy_mse_list, uncertainty_min_list)
        print_corr(
            "Force Direct Uncertainty Min", force_direct_mse_list, uncertainty_min_list
        )
        print_corr(
            "Force Grad Uncertainty Min", force_grad_mse_list, uncertainty_min_list
        )

        print_corr("Energy Uncertainty Max", energy_mse_list, uncertainty_max_list)
        print_corr(
            "Force Direct Uncertainty Max", force_direct_mse_list, uncertainty_max_list
        )
        print_corr(
            "Force Grad Uncertainty Max", force_grad_mse_list, uncertainty_max_list
        )

        print_corr(
            "Energy Uncertainty Mean Absolute", energy_mse_list, uncertainty_ma_list
        )
        print_corr(
            "Force Direct Uncertainty Mean Absolute",
            force_direct_mse_list,
            uncertainty_ma_list,
        )
        print_corr(
            "Force Grad Uncertainty Mean Absolute",
            force_grad_mse_list,
            uncertainty_ma_list,
        )

        print_corr(
            "Energy Uncertainty Mean Squared", energy_mse_list, uncertainty_ms_list
        )
        print_corr(
            "Force Direct Uncertainty Mean Squared",
            force_direct_mse_list,
            uncertainty_ms_list,
        )
        print_corr(
            "Force Grad Uncertainty Mean Squared",
            force_grad_mse_list,
            uncertainty_ms_list,
        )

        print_corr(
            "Energy Uncertainty Mean Cubic", energy_mse_list, uncertainty_mc_list
        )
        print_corr(
            "Force Direct Uncertainty Mean Cubic",
            force_direct_mse_list,
            uncertainty_mc_list,
        )
        print_corr(
            "Force Grad Uncertainty Mean Cubic",
            force_grad_mse_list,
            uncertainty_mc_list,
        )

        print_corr("Energy Uncertainty Log", energy_mse_list, uncertainty_log_list)
        print_corr(
            "Force Direct Uncertainty Log", force_direct_mse_list, uncertainty_log_list
        )
        print_corr(
            "Force Grad Uncertainty Log", force_grad_mse_list, uncertainty_log_list
        )

        print_corr("Energy Uncertainty Exp", energy_mse_list, uncertainty_exp_list)
        print_corr(
            "Force Direct Uncertainty Exp", force_direct_mse_list, uncertainty_exp_list
        )
        print_corr(
            "Force Grad Uncertainty Exp", force_grad_mse_list, uncertainty_exp_list
        )

    # Pearson correlations
    calculate_and_print_correlations(pearsonr, "Pearson")

    # Spearman correlations
    calculate_and_print_correlations(spearmanr, "Spearman")
