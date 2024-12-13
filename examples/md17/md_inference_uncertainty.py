##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory
# All rights reserved.
#
# This file is part of HydraGNN and is distributed under a BSD 3-clause
# license. For the licensing terms see the LICENSE file in the top-level
# directory.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

import json
import os
import sys
from tqdm import tqdm
import argparse
from mpi4py import MPI

import torch
import numpy as np
from torch_geometric.data import Data

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.distributed import get_device, setup_ddp
from hydragnn.utils.model import load_existing_model
from hydragnn.models.create import create_model_config
from hydragnn.utils.uncertainty_utils import (
    minmax_scale_data,
    reverse_minmax_scale_data,
)

from scipy.interpolate import griddata

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})


def get_log_name_config(config):
    return (
        config["NeuralNetwork"]["Architecture"]["model_type"]
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-ncl-"
        + str(config["NeuralNetwork"]["Architecture"]["num_conv_layers"])
        + "-hd-"
        + str(config["NeuralNetwork"]["Architecture"]["hidden_dim"])
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-node_ft-"
        + "".join(
            str(x)
            for x in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        )
        + "-task_weights-"
        + "".join(
            str(weigh) + "-"
            for weigh in config["NeuralNetwork"]["Architecture"]["task_weights"]
        )
    )


def getcolordensity(xdata, ydata):
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(f"Maximum normalized histogram value: {np.amax(hist2d)}")

    bctx1d = BCTX.flatten()
    bcty1d = BCTY.flatten()
    loc_pts = np.vstack((bctx1d, bcty1d)).T
    hist2d_norm = griddata(
        loc_pts,
        hist2d.flatten(),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )
    return hist2d_norm


def plot_scatter(x, y, hist2d_norm, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, s=8, c=hist2d_norm, vmin=0, vmax=1, cmap="viridis")
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
    plt.colorbar(scatter, ax=ax, label="Density")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close(fig)
    print(f"Saved plot: {filename}")


if __name__ == "__main__":

    modelname = "md17"

    parser = argparse.ArgumentParser(
        description="Evaluate HydraGNN Model on Test Dataset"
    )
    parser.add_argument(
        "--inputfile",
        help="Path to the config JSON file",
        type=str,
        default="./logs/md17/config.json",
    )
    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    if not os.path.isfile(input_filename):
        print(f"Config file not found: {input_filename}")
        sys.exit(1)

    with open(input_filename, "r") as f:
        config = json.load(f)

    hydragnn.utils.print.setup_log(get_log_name_config(config))
    print("Configuration loaded and logging setup.")

    ##################################################################################################################
    # Initialize Distributed Data Parallel (DDP) if applicable
    comm_size, rank = setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD  # Assuming MPI is being used elsewhere as per original script

    datasetname = "MD17"

    comm.Barrier()

    # Define the paths to the serialized datasets
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    serialized_data_path = os.environ.get("SERIALIZED_DATA_PATH")
    train_path = os.path.join(serialized_data_path, "train_dataset.pt")
    val_path = os.path.join(serialized_data_path, "val_dataset.pt")
    test_path = os.path.join(serialized_data_path, "test_dataset.pt")

    # Check if the files exist
    if not all(os.path.isfile(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"One or more dataset files not found in {serialized_data_path}. "
            "Ensure that 'train_dataset.pt', 'val_dataset.pt', and 'test_dataset.pt' exist."
        )

    # Load the datasets
    trainset = torch.load(train_path)
    valset = torch.load(val_path)
    testset = torch.load(test_path)
    print(f"Loaded train set with {len(trainset)} samples.")
    print(f"Loaded val set with {len(valset)} samples.")
    print(f"Loaded test set with {len(testset)} samples.")

    # Scale the data
    trainset, valset, testset, train_energy_min, train_energy_max = minmax_scale_data(
        trainset, valset, testset
    )

    # Initialize the model
    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )
    model = torch.nn.parallel.DistributedDataParallel(model)
    load_existing_model(model, modelname, path="./logs/")

    # Prepare model
    device = get_device()
    model.to(device)
    model.eval()

    # Initialize lists to collect predictions and true values
    energy_true_list = []
    energy_pred_list = []
    forces_true_list = []
    forces_pred_direct_list = []
    forces_pred_grad_list = []
    dataset_predictions = []

    # Disable gradient computation for evaluation
    for data_id, data in enumerate(tqdm(testset, desc="Evaluating")):
        # Prepare
        data = data.to(device)
        data.pos.requires_grad = True

        # Predict
        pred = model(data)
        node_energy_pred, node_forces_pred_direct = pred[0], pred[1]
        energy_pred = torch.sum(node_energy_pred, dim=0).float()
        node_forces_pred_grad = -torch.autograd.grad(
            outputs=energy_pred,
            inputs=data.pos,
            grad_outputs=torch.ones_like(energy_pred),
            retain_graph=False,
            create_graph=False,
        )[0]

        # True
        energy_true = data.energy
        node_forces_true = data.forces

        # Descale
        (
            energy_pred,
            energy_true,
            node_forces_pred_direct,
            node_forces_pred_grad,
            node_forces_true,
        ) = reverse_minmax_scale_data(
            energy_pred,
            energy_true,
            node_forces_pred_direct,
            node_forces_pred_grad,
            node_forces_true,
            train_energy_min,
            train_energy_max,
        )

        # Collect lists
        energy_pred_list.extend(energy_pred.tolist())
        energy_true_list.extend(energy_true.tolist())
        forces_pred_direct_list.extend((node_forces_pred_direct).flatten().tolist())
        forces_pred_grad_list.extend((node_forces_pred_grad).flatten().tolist())
        forces_true_list.extend(node_forces_true.flatten().tolist())

        # Collect dataset with predictions
        data.energy = torch.tensor(energy_true)
        data.energy_pred = torch.tensor(energy_pred)
        data.forces = torch.tensor(node_forces_true)
        data.forces_pred_direct = torch.tensor(node_forces_pred_direct)
        data.forces_pred_grad = torch.tensor(node_forces_pred_grad)
        dataset_predictions.append(data)

    # Compute R² scores
    energy_r2 = r2_score(np.array(energy_true_list), np.array(energy_pred_list))
    forces_direct_r2 = r2_score(
        np.array(forces_true_list), np.array(forces_pred_direct_list)
    )
    forces_grad_r2 = r2_score(
        np.array(forces_true_list), np.array(forces_pred_grad_list)
    )
    print(f"Energy R²: {energy_r2:.4f}")
    print(f"Forces Direct R²: {forces_direct_r2:.4f}")
    print(f"Forces Grad R²: {forces_grad_r2:.4f}")

    # Save predicted test dataset
    save_file = os.path.join(serialized_data_path, "test_dataset_predictions.pt")
    torch.save(dataset_predictions, save_file)

    # Plotting Energy Scatter Plot
    hist2d_norm = getcolordensity(energy_true_list, energy_pred_list)
    plot_scatter(
        x=energy_true_list,
        y=energy_pred_list,
        hist2d_norm=hist2d_norm,
        xlabel="True Energy",
        ylabel="Predicted Energy",
        title="Energy Prediction Scatter Plot",
        filename="./energy_Scatterplot.png",
    )

    # Plotting Forces Direct Scatter Plot
    hist2d_norm = getcolordensity(forces_pred_direct_list, forces_true_list)
    plot_scatter(
        x=forces_pred_direct_list,
        y=forces_true_list,
        hist2d_norm=hist2d_norm,
        xlabel="Predicted Forces (Direct)",
        ylabel="True Forces",
        title="Forces Direct Prediction Scatter Plot",
        filename="./Forces_Scatterplot_direct.png",
    )

    # Plotting Forces Gradient Scatter Plot
    hist2d_norm = getcolordensity(forces_pred_grad_list, forces_true_list)
    plot_scatter(
        x=forces_pred_grad_list,
        y=forces_true_list,
        hist2d_norm=hist2d_norm,
        xlabel="Predicted Forces (Grad)",
        ylabel="True Forces",
        title="Forces Gradient Prediction Scatter Plot",
        filename="./Forces_Scatterplot_grad.png",
    )

    print("Evaluation and plotting completed successfully.")
