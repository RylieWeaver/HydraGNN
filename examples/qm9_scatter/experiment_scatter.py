import os, json

import torch

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

import torch_geometric
from tqdm import tqdm

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import hydragnn
from hydragnn.utils.model import load_existing_model, save_model
from hydragnn.utils.distributed import get_device

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
from sklearn.metrics import r2_score

# Update each sample prior to loading.
def qm9_pre_transform(data):
    # Set descriptor as element type.
    data.x = data.z.float().view(-1, 1)
    # Only predict free energy (index 10 of 19 properties) for this run.
    # data.y = data.y[:, 10] / len(data.x)
    data.y = data.y[:, 10]
    graph_features_dim = [1]
    node_feature_dim = [1]
    return data


def qm9_pre_filter(data):
    return data


def minmax_scale_data(train, val, test):
    # Do minmax scaling from just the training set
    train_energy = torch.cat([data.y for data in train])
    train_energy_min = train_energy.min()
    train_energy_max = train_energy.max()
    
    # Scale energy and forces by same factor
    for dataset in [train, val, test]:
        for data in dataset:
            data.y = (data.y - train_energy_min) / (train_energy_max - train_energy_min)
            
    return train, val, test, train_energy_min, train_energy_max


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


# Make dataset
def create_dataset(config):
    # Set this path for output.
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

    # Use built-in torch_geometric datasets.
    # Filter function above used to run quick example.
    # NOTE: data is moved to the device in the pre-transform.
    # NOTE: transforms/filters will NOT be re-run unless the qm9/processed/ directory is removed.
    dataset = torch_geometric.datasets.QM9(
        root="dataset/qm9", pre_transform=qm9_pre_transform, pre_filter=qm9_pre_filter
    )
    
    return dataset


def run_model(model_type, num_samples, config, train, val, test):
    verbosity = config["Verbosity"]["level"]
    config["NeuralNetwork"]["Architecture"]["model_type"] = model_type
    
    # Scale Data
    train, val, test, energy_min, energy_max = minmax_scale_data(train, val, test)
    
    # Always initialize for multi-rank training.
    world_size, world_rank = hydragnn.utils.distributed.setup_ddp()
    log_name = f"qm9_test_{model_type}_{num_samples}"
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
    
    test_MAE = 0.0
    test_MSE = 0.0
    pred_list = []
    true_list = []
    for data_id, data in enumerate(tqdm(test)):
        # data.pos.requires_grad = True
        energy_pred = model(data.to(get_device()))[0]
        energy_true = data.y
        
        # De-Scale
        energy_pred = energy_pred * (energy_max - energy_min) + energy_min
        energy_true = energy_true * (energy_max - energy_min) + energy_min
        
        # Calculate MAE and MSE
        test_MAE += torch.norm(energy_pred - energy_true, p=1).item()
        test_MSE += torch.norm(energy_pred - energy_true, p=2).item() ** 2
        pred_list.append(energy_pred)
        true_list.append(energy_true) 
    
    return test_MAE / len(test), test_MSE / len(test), pred_list, true_list
    

def run_experiment():
    # Make sure to clean logs directory
    os.system("rm -rf logs")
    
    # num_repeats = 1
    # list(map(int, np.logspace(np.log10(100), np.log10(10000), num=10)))
    # num_samples = [100, 166, 278, 464, 774, 1291, 2154, 3593, 5994, 10000]
    model_types = ["PAINN"]
    results = {model: [] for model in model_types}

    # Read Config
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.json")
    with open(filename, "r") as f:
        config = json.load(f)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    
    # Run Experiment
    dataset = create_dataset(config)
    num_sample = len(dataset)
    # Split dataset
    sample_dataset = dataset
    train, val, test = hydragnn.preprocess.split_dataset(
        sample_dataset, config["NeuralNetwork"]["Training"]["perc_train"], False
    )
    # Run dataset for all models
    for model_type in model_types:
        print(f"\nRunning model: {model_type}")
        MAE, MSE, pred_list, true_list = run_model(model_type, num_sample, config, train, val, test)
        # results[model_type].append({
        #     "num_samples": num_sample,
        #     "repetition": repeat + 1,
        #     "MAE": MAE,
        #     "MSE": MSE,
        #     "pred_list": pred_list,
        #     "true_list": true_list
        # })
    
        # Use pred and true to create scatter plot and save as png
        print(f"\nCreating scatter plot for {model_type}")
        true_all = torch.cat([t.detach() for t in true_list]).squeeze().cpu().numpy()
        pred_all = torch.cat([t.detach() for t in pred_list]).squeeze().cpu().numpy()
        # Save True and Pred
        np.save(f"{model_type}_true.npy", true_all)
        np.save(f"{model_type}_pred.npy", pred_all)
        # Normalize density for color mapping
        hist2d_norm = getcolordensity(pred_all, true_all)
        # Create scatter plot with color density
        fig, ax = plt.subplots()
        sc = plt.scatter(true_all, pred_all, s=8, c=hist2d_norm, vmin=0, vmax=1)
        plt.clim(0, 1)
        # Add diagonal reference line
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
        ax.tick_params(axis='both', labelsize=8)
        # Add color bar
        cb = plt.colorbar(sc)
        cb.set_label("Density", fontsize=12)
        # Add labels and title
        plt.xlabel("True Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(f"{model_type} QM9 Energy Values", fontsize=16)
        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(f"{model_type}_scatter.png", dpi=400)
        plt.close()
        
        # Save Results
        # Calculate R2 score
        r2_value = r2_score(true_all, pred_all)
        print(f"R2 Score for {model_type}: {r2_value}")
        # Save predicted and true values along with R2 score
        results_file = f"{model_type}_results.json"
        results_data = {
            "r2_score": r2_value,
            "predicted": pred_all.tolist(),
            "true": true_all.tolist(),
        }
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Saved results for {model_type} to {results_file}")


if __name__ == "__main__":
    run_experiment()