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
import torchmetrics
import numpy as np

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


def evaluate(modeltype):
    modelname = "LJ"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="./logs/LJ/config.json"
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
    config["NeuralNetwork"]["Architecture"]["model_type"] = modeltype
    # config["Dataset"]["primitive_bravais_constant"] = primitive_bravais_constant
    hydragnn.utils.print.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    datasetname = "LJ"

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()
    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
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
    model.eval()

    variable_index = 0
    # for output_name, output_type, output_dim in zip(config["NeuralNetwork"]["Variables_of_interest"]["output_names"], config["NeuralNetwork"]["Variables_of_interest"]["type"], config["NeuralNetwork"]["Variables_of_interest"]["output_dim"]):

    # Global variables
    num_samples = len(testset)
    energy_min = testset.energy_min
    energy_max = testset.energy_max
    forces_max = testset.forces_max
    forces_min = testset.forces_min
    
    energy_true_all = torch.empty(0, device=get_device())
    energy_pred_all = torch.empty(0, device=get_device())
    forces_true_all = torch.empty(0, 3, device=get_device())  # Assuming forces are 3D vectors
    forces_pred_all = torch.empty(0, 3, device=get_device())

    for data_id, data in enumerate(tqdm(testset)):
        data.pos.requires_grad = True
        node_energy_pred = model(data.to(get_device()))[
            0
        ]  # Note that this is sensitive to energy and forces prediction being single-task (current requirement)
        energy_pred = torch.sum(node_energy_pred, dim=0).float()
        # predicted.backward(retain_graph=True)
        # gradients = data.pos.grad
        grads_energy = torch.autograd.grad(
            outputs=energy_pred,
            inputs=data.pos,
            grad_outputs=torch.ones_like(energy_pred),
            retain_graph=False,
            create_graph=True,
        )[0]
        
        # Convert to tensors
        energy_pred = torch.tensor(energy_pred, device=get_device())
        grads_energy = torch.tensor(grads_energy, device=get_device())
        
        # De-Scale Logarithmic Energy
        # energy_pred = torch.sign(energy_pred) * (torch.exp(energy_pred.abs()) - 1.0)
        # energy_true = torch.sign(data.energy) * (torch.exp(data.energy.abs()) - 1.0)
        # forces_pred = -grads_energy * (torch.abs(energy_pred) + 1.0)
        # forces_true = data.forces * (torch.abs(energy_true) + 1.0)
        # De-Scale MinMax [0,1] Energies
        energy_pred = (energy_pred * (energy_max - energy_min)) + energy_min
        energy_true = (data.energy * (energy_max - energy_min)) + energy_min
        forces_pred = (-grads_energy * (energy_max - energy_min)) + energy_min
        forces_true = (data.forces * (energy_max - energy_min)) + energy_min
        # # De-Scale MinMax [0,1] Forces
        # energy_pred = (energy_pred * (forces_max - forces_min)) + forces_min
        # energy_true = (data.energy * (forces_max - forces_min)) + forces_min
        # forces_pred = (-grads_energy * (forces_max - forces_min)) + forces_min
        # forces_true = (data.forces * (forces_max - forces_min)) + forces_min
        # No De-Scaling
        # energy_pred = energy_pred
        # energy_true = data.energy
        # forces_pred = -grads_energy
        # forces_true = data.forces
        
        # Concatenate predictions and true values to initialized tensors
        energy_pred_all = torch.cat((energy_pred_all, energy_pred.unsqueeze(0)))
        energy_true_all = torch.cat((energy_true_all, energy_true.unsqueeze(0)))
        forces_pred_all = torch.cat((forces_pred_all, forces_pred))
        forces_true_all = torch.cat((forces_true_all, forces_true))
        
    # Flatten forces
    forces_pred_all = forces_pred_all.flatten()
    forces_true_all = forces_true_all.flatten()
    
    # Get MSE losses
    energy_loss = torch.nn.MSELoss()(energy_pred_all, energy_true_all)
    forces_loss = torch.nn.MSELoss()(forces_pred_all, forces_true_all)
    tasks_loss = [energy_loss, forces_loss]

    # Get R2 values
    energy_r2 = torchmetrics.R2Score()(energy_pred_all, energy_true_all)
    forces_r2 = torchmetrics.R2Score()(forces_pred_all, forces_true_all)
    tasks_r2 = [energy_r2, forces_r2]
    
    # Convert to lists for plotting
    energy_true_list = energy_true_all.squeeze().tolist()
    energy_pred_list = energy_pred_all.squeeze().tolist()
    forces_true_list = forces_true_all.squeeze().tolist()
    forces_pred_list = forces_pred_all.squeeze().tolist()
    
    # Find the common range for x and y axes based on data
    energy_min = min(min(energy_true_list), min(energy_pred_list))
    energy_max = max(max(energy_true_list), max(energy_pred_list))
    forces_min = min(min(forces_true_list), min(forces_pred_list))
    forces_max = max(max(forces_true_list), max(forces_pred_list))

    # Set limits based on the maximum range
    energy_range = (energy_min, energy_max)
    forces_range = (forces_min, forces_max)

    # Plotting energy predictions
    hist2d_norm = getcolordensity(energy_true_list, energy_pred_list)
    fig, ax = plt.subplots()
    plt.scatter(energy_true_list, energy_pred_list, s=8, c=hist2d_norm, vmin=0, vmax=1)
    plt.clim(0, 1)
    ax.plot(energy_range, energy_range, ls="--", color="red")  # Diagonal line with same scale
    plt.colorbar()
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Energy")
    ax.set_xlim(energy_range)
    ax.set_ylim(energy_range)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(f"./energy_Scatterplot.png", dpi=400)

    # Plotting forces predictions
    hist2d_norm = getcolordensity(forces_true_list, forces_pred_list)
    fig, ax = plt.subplots()
    plt.scatter(forces_true_list, forces_pred_list, s=8, c=hist2d_norm, vmin=0, vmax=1)
    plt.clim(0, 1)
    ax.plot(forces_range, forces_range, ls="--", color="red")  # Diagonal line with same scale
    plt.colorbar()
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Forces")
    ax.set_xlim(forces_range)
    ax.set_ylim(forces_range)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(f"./Forces_Scatterplot.png", dpi=400)
    
    return [tasks_loss, tasks_r2]