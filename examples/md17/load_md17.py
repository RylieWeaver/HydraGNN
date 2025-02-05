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

# General
import os, json
import logging
import sys
import argparse
import random

# Torch
import torch
import torch_geometric
import numpy as np

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float64)

# Distributed
import mpi4py
from mpi4py import MPI

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

# HydraGNN
import hydragnn
from hydragnn.utils.print.print_utils import log
from hydragnn.utils.profiling_and_tracing.time_utils import Timer
import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.utils.model import print_model
from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)
from hydragnn.preprocess.graph_samples_checks_and_updates import gather_deg

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from scipy.interpolate import BSpline, make_interp_spline
import adios2 as ad2

torch.backends.cudnn.enabled = False


##################################################################################################################


# Update each sample prior to loading.
def md17_pre_transform(data, compute_edges):
    data.x = data.z.float().view(-1, 1)
    data.y = data.energy
    data.forces = data.force
    data = compute_edges(data)
    return data


# Take all samples
def md17_pre_filter(data):
    return True


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def stable_split_dataset(dataset, perc_train=0.8, seed=42):
    total = list(dataset)  # Ensure dataset is a list for slicing
    random.seed(seed)
    random.shuffle(total)
    n = len(total)
    train_size = int(n * perc_train)
    remaining = n - train_size
    # Split the remainder equally into validation and test sets
    val_size = remaining // 2
    test_size = remaining - val_size
    trainset = total[:train_size]
    valset = total[train_size : train_size + val_size]
    testset = total[train_size + val_size : train_size + val_size + test_size]
    return trainset, valset, testset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="md17_PNAEq.json"
    )
    parser.add_argument("--mpnn_type", help="mpnn_type", default="PNAEq")
    parser.add_argument("--hidden_dim", type=int, help="hidden_dim", default=5)
    parser.add_argument(
        "--num_conv_layers", type=int, help="num_conv_layers", default=6
    )
    parser.add_argument(
        "--num_sharedlayers", type=int, help="num_sharedlayers", default=2
    )
    parser.add_argument(
        "--dim_sharedlayers", type=int, help="dim_sharedlayers", default=10
    )
    parser.add_argument("--num_headlayers", type=int, help="num_headlayers", default=2)
    parser.add_argument(
        "--dim_headlayers_graph", type=int, help="dim_headlayers_graph", default=10
    )
    parser.add_argument(
        "--dim_headlayers_node", type=int, help="dim_headlayers_node", default=10
    )

    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name", default="MD17_test")
    parser.add_argument("--num_epoch", type=int, help="num_epoch", default=None)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument(
        "--multi_model_list", help="multidataset list", default="OC2020"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="set num samples per process for weak-scaling test",
        default=None,
    )
    parser.add_argument(
        "--compute_grad_energy",
        action="store_true",
        help="use automatic differentiation to compute gradients of energy",
        default=False,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    group.add_argument(
        "--multi",
        help="Multi dataset",
        action="store_const",
        dest="format",
        const="multi",
    )
    parser.set_defaults(format="adios")
    args = parser.parse_args()
    args.parameters = vars(args)

    graph_feature_names = ["energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number"]
    node_feature_dims = [1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    config["NeuralNetwork"]["Architecture"]["mpnn_type"] = (
        args.mpnn_type
        if args.mpnn_type
        else config["NeuralNetwork"]["Architecture"]["mpnn_type"]
    )
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size

    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "MD17_test" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "MD17" if args.modelname is None else args.modelname
    # Make path for each format
    if args.format == "pickle":
        basedir = os.path.join(dirpwd, "dataset", "%s.pickle" % modelname)
    if args.format == "adios":
        fname = os.path.join(dirpwd, "./dataset/%s.bp" % modelname)

    tr.initialize()
    tr.disable()
    timer = Timer("load_data")
    timer.start()

    # Preprocess configurations for edge computation
    arch_config = config["NeuralNetwork"]["Architecture"]
    compute_edges = hydragnn.preprocess.get_radius_graph_config(arch_config)

    # Fix for MD17 datasets
    torch_geometric.datasets.MD17.file_names["uracil"] = "md17_uracil.npz"
    dataset = torch_geometric.datasets.MD17(
        root="dataset/md17",
        name="uracil",
        pre_transform=lambda data: md17_pre_transform(data, compute_edges),
        pre_filter=md17_pre_filter,
    )
    trainset, valset, testset = stable_split_dataset(
        dataset, config["NeuralNetwork"]["Training"]["perc_train"], 42
    )  # Consistent seed for same splitting

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    deg = gather_deg(trainset)
    config["pna_deg"] = deg.tolist()

    if args.format == "pickle":

        ## pickle
        attrs = dict()
        attrs["pna_deg"] = deg
        SimplePickleWriter(
            trainset,
            basedir,
            "trainset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
            attrs=attrs,
        )
        SimplePickleWriter(
            valset,
            basedir,
            "valset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )
        SimplePickleWriter(
            testset,
            basedir,
            "testset",
            # minmax_node_feature=total.minmax_node_feature,
            # minmax_graph_feature=total.minmax_graph_feature,
            use_subdir=True,
        )

    if args.format == "adios":
        ## adios
        adwriter = AdiosWriter(fname, comm)
        adwriter.add("trainset", trainset)
        adwriter.add("valset", valset)
        adwriter.add("testset", testset)
        # adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
        # adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
        adwriter.add_global("pna_deg", deg)
        adwriter.save()

    timer.stop()


if __name__ == "__main__":
    main()
