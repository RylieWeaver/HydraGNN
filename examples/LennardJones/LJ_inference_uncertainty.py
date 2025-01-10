##############################################################################
# Copyright (c) 2024, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import json, os
import sys
import logging
import pickle
from tqdm import tqdm
from mpi4py import MPI
import argparse

import torch
import torch_scatter
import numpy as np
from torch_geometric.data import Data

import hydragnn
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
from hydragnn.utils.distributed import get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.utils.input_config_parsing.config_utils import (
    update_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

from scipy.interpolate import griddata

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from LJ_data import info
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
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
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

    modelname = "LJ_optuna_40"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="./logs/LJ_optuna_40/config.json"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios gan_dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle gan_dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.print.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    datasetname = "LJ"

    comm.Barrier()

    # Define the paths to the serialized datasets
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    serialized_data_path = os.environ.get("SERIALIZED_DATA_PATH")

    timer = Timer("load_data")
    timer.start()
    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % datasetname
        )
        trainset = SimplePickleDataset(
            basedir=basedir,
            label="trainset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        valset = SimplePickleDataset(
            basedir=basedir,
            label="valset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        testset = SimplePickleDataset(
            basedir=basedir,
            label="testset",
            var_config=config["NeuralNetwork"]["Variables_of_interest"],
        )
        pna_deg = trainset.pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(model)

    load_existing_model(model, modelname, path="./logs/")
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
