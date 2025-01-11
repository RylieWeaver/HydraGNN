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
from tqdm import tqdm

# Torch
import torch

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
from hydragnn.preprocess.load_data import split_dataset
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

from hydragnn.utils.model import load_existing_model, save_model
from hydragnn.utils.distributed import get_device

# Lennard Jones
from LJ_data import create_dataset, LJDataset, info


def run_model(model_type, num_samples, config, trainset, valset, testset):
    verbosity = config["Verbosity"]["level"]
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    
    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    log_name = f"LJ_test_{model_type}_{num_samples}"
    hydragnn.utils.print.print_utils.setup_log(log_name)
    
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
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
        create_plots=False,
        compute_grad_energy=False,
    )
    
    hydragnn.utils.model.save_model(model, optimizer, log_name)
    load_existing_model(model, log_name, path="./logs/")
    model.eval()
    
    test_MAE = 0.0
    test_MSE = 0.0
    for data_id, data in enumerate(tqdm(testset)):
        # data.pos.requires_grad = True
        # node_energy_pred = model(data.to(get_device()))[
        #     0
        # ]  # Note that this is sensitive to energy and forces prediction being single-task (current requirement)
        # energy_pred = torch.sum(node_energy_pred, dim=0).float()
        # # predicted.backward(retain_graph=True)
        # # gradients = data.pos.grad
        # grads_energy = torch.autograd.grad(
        #     outputs=energy_pred,
        #     inputs=data.pos,
        #     grad_outputs=torch.ones_like(energy_pred),
        #     retain_graph=False,
        #     create_graph=True,
        # )[0]
        # forces_pred = -grads_energy
        energy_pred, forces_pred = model(data.to(get_device()))
        test_MAE += torch.norm(energy_pred - data.energy, p=1).item()
        test_MAE += torch.norm(forces_pred + data.forces, p=1).item()
        test_MSE += torch.norm(energy_pred - data.energy, p=2).item() ** 2
        test_MSE += torch.norm(forces_pred + data.forces, p=2).item() ** 2

    
    return test_MAE / len(testset), test_MSE / len(testset)


def run_experiment():
    # Make sure to clean logs directories
    os.system("rm -rf logs")
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument("--inputfile", help="input file", type=str, default="LJ_multitask.json")
    parser.add_argument("--model_type", help="model type", type=str, default=None)
    parser.add_argument("--mae", action="store_true", help="do mae calculation")
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")

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
    parser.set_defaults(format="pickle")  # Changed this for my PC
    args = parser.parse_args()

    graph_feature_names = ["total_energy"]
    graph_feature_dims = [1]
    node_feature_names = ["atomic_number", "pos", "forces"]
    node_feature_dims = [1, 3, 3]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, args.inputfile)
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    config["NeuralNetwork"]["Architecture"]["model_type"] = (
        args.model_type
        if args.model_type
        else config["NeuralNetwork"]["Architecture"]["model_type"]
    )
    verbosity = config["Verbosity"]["level"]
    config["NeuralNetwork"]["Variables_of_interest"][
        "graph_feature_names"
    ] = graph_feature_names
    config["NeuralNetwork"]["Variables_of_interest"][
        "graph_feature_dims"
    ] = graph_feature_dims
    config["NeuralNetwork"]["Variables_of_interest"][
        "node_feature_names"
    ] = node_feature_names
    config["NeuralNetwork"]["Variables_of_interest"][
        "node_feature_dims"
    ] = node_feature_dims

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

    log_name = "LJ" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    modelname = "LJ"
    # Check for dataset for each format
    if args.format == "pickle":
        basedir = os.path.join(dirpwd, "dataset", "%s.pickle" % modelname)
        dataset_exists = os.path.exists(os.path.join(dirpwd, "dataset/LJ.pickle"))
    if args.format == "adios":
        fname = os.path.join(dirpwd, "./dataset/%s.bp" % modelname)
        dataset_exists = os.path.exists(
            os.path.join(dirpwd, "dataset", "%s.bp" % modelname)
        )

    ## Create Dataset
    config["Dataset"]["number_configurations"] = 10000
    if not dataset_exists:
        create_dataset(os.path.join(dirpwd, "dataset/data"), config)
    dataset = LJDataset(
        os.path.join(dirpwd, "dataset/data"),
        config,
        dist=True,
    )

    
    # Define number of samples and models
    num_repeats = 1
    # list(map(int, np.logspace(np.log10(100), np.log10(10000), num=10)))
    num_samples = [100, 166, 278, 464, 774, 1291, 2154, 3593, 5994, 10000]
    # model_types = ["CGCNN", "EGNN", "GIN", "GAT", "MFC", "SAGE", "PNA", "SchNet", "DimeNet", "PNAPlus", "PNAEq", "PAINN", "MACE"]
    # model_types = ["CGCNN", "EGNN", "GIN", "GAT", "MFC", "SAGE", "PNA", "SchNet", "DimeNet", "PNAPlus", "PNAEq", "PAINN"]
    model_types = ["DimeNet"]
    results = {model: [] for model in model_types}
    
    # Run Experiment
    for num_sample in num_samples:
        print(f"\n\n\nRunning experiment with {num_sample} samples")
        for repeat in range(num_repeats):
            print(f"\n--- Repetition {repeat + 1} ---")
            
            # Split dataset
            sample_dataset = dataset[:num_sample]
            
            ## This is a local split
            trainset, valset, testset = split_dataset(
                dataset=sample_dataset,
                perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
                stratify_splitting=False,
            )
            print("Local splitting: ", len(sample_dataset), len(trainset), len(valset), len(testset))

            deg = gather_deg(trainset)
            config["pna_deg"] = deg.tolist()
            
            setnames = ["trainset", "valset", "testset"]
            args.format = "pickle"

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

            tr.initialize()
            tr.disable()
            timer = Timer("load_data")
            timer.start()
            if args.format == "adios":
                info("Adios load")
                assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
                opt = {
                    "preload": False,
                    "shmem": args.shmem,
                    "ddstore": args.ddstore,
                    "ddstore_width": args.ddstore_width,
                }
                trainset = AdiosDataset(fname, "trainset", comm, **opt)
                valset = AdiosDataset(fname, "valset", comm, **opt)
                testset = AdiosDataset(fname, "testset", comm, **opt)
            elif args.format == "pickle":
                info("Pickle load")
                var_config = config["NeuralNetwork"]["Variables_of_interest"]
                trainset = SimplePickleDataset(
                    basedir=basedir, label="trainset", preload=False, var_config=var_config
                )
                valset = SimplePickleDataset(
                    basedir=basedir, label="valset", var_config=var_config
                )
                testset = SimplePickleDataset(
                    basedir=basedir, label="testset", var_config=var_config
                )
                # minmax_node_feature = trainset.minmax_node_feature
                # minmax_graph_feature = trainset.minmax_graph_feature
                pna_deg = trainset.pna_deg
                if args.ddstore:
                    opt = {"ddstore_width": args.ddstore_width}
                    trainset = DistDataset(trainset, "trainset", comm, **opt)
                    valset = DistDataset(valset, "valset", comm, **opt)
                    testset = DistDataset(testset, "testset", comm, **opt)
                    # trainset.minmax_node_feature = minmax_node_feature
                    # trainset.minmax_graph_feature = minmax_graph_feature
                    trainset.pna_deg = pna_deg
            else:
                raise NotImplementedError("No supported format: %s" % (args.format))

            info(
                "trainset,valset,testset size: %d %d %d"
                % (len(trainset), len(valset), len(testset))
            )

            if args.ddstore:
                os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
                os.environ["HYDRAGNN_USE_ddstore"] = "1"
            

            # Run dataset for all models
            for model_type in model_types:
                edge_models = [
                    "PNA",
                    "PNAPlus",
                    "PAINN",
                    "PNAEq",
                    # "CGCNN",
                    "SchNet",
                    "EGNN",
                    "DimeNet",
                    "MACE",
                ]
                if model_type in edge_models:
                    config["NeuralNetwork"]["Architecture"]["edge_features"] = ["a", "b", "c"]
                else:
                    config["NeuralNetwork"]["Architecture"]["edge_features"] = None
                print(f"\nRunning model: {model_type}")
                MAE, MSE = run_model(model_type, num_sample, config, trainset, valset, testset)
                results[model_type].append({
                    "num_samples": num_sample,
                    "repetition": repeat + 1,
                    "MAE": MAE,
                    "MSE": MSE,
                })
            
    # Save results to file
    output_file = "results_direct.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
        

if __name__ == "__main__":
    run_experiment()