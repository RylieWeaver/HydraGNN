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

# Torch
import torch
import torch_geometric
from torch_geometric.data import DataLoader

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

# Lennard Jones
from LJ_data import create_dataset, LJDataset, info

# HPO
import optuna
import pandas as pd


def objective(trial):
    global config, best_trial_id, best_validation_loss

    # Extract the unique trial ID
    trial_id = trial.number  # or trial_id = trial.trial_id

    log_name = "LJ_optuna"
    log_name = log_name + "_" + str(trial_id)
    hydragnn.utils.print.print_utils.setup_log(log_name)
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
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.distributed.get_distributed_model(model, verbosity)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001
    )

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
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
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
        compute_grad_energy=True,
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
        val_loader, model, verbosity, reduce_ranks=True, compute_grad_energy=True
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
    # Read Pickle or Adios
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--shmem", action="store_true", help="shmem")
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
    node_feature_names = ["atomic_number", "potential", "forces"]
    node_feature_dims = [1, 1, 3]

    # Configurable run choices (JSON file that accompanies this example script).
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirpwd, "LJ_uncertainty.json")
    with open(filename, "r") as f:
        config = json.load(f)
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
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    log_name = "LJ_optuna"
    # Enable print to log file.
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

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

    # Create dataset if preonly specified or dataset does not exist
    if not dataset_exists:
        ## local data
        create_dataset(os.path.join(dirpwd, "dataset/data"), config)
        total = LJDataset(
            os.path.join(dirpwd, "dataset/data"),
            config,
            dist=True,
        )
        ## This is a local split
        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        print("Local splitting: ", len(total), len(trainset), len(valset), len(testset))

        deg = gather_deg(trainset)
        config["pna_deg"] = deg.tolist()

        setnames = ["trainset", "valset", "testset"]

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

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    ## Good to sync with everyone right after DDStore setup
    comm.Barrier()

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
