import argparse
import os
import sys
import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from hydragnn.utils.uncertainty_utils import calculate_metrics, analyze_correlations


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


def main():
    # Define the paths to the serialized dataset
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    serialized_data_path = os.environ.get("SERIALIZED_DATA_PATH")

    # Load and calculate metrics
    dataset = torch.load(
        os.path.join(serialized_data_path, "test_dataset_predictions.pt")
    )
    (
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
    ) = calculate_metrics(dataset)

    # Plot Energy Error vs Uncertainty Error
    hist2d_norm = getcolordensity(energy_mse_list, uncertainty_ms_list)
    plot_scatter(
        x=energy_mse_list,
        y=uncertainty_ms_list,
        hist2d_norm=hist2d_norm,
        xlabel="Energy Error",
        ylabel="Uncertainty",
        title="Uncertainty vs Energy Error Scatter Plot",
        filename="./energy_uncertainty_Scatterplot.png",
    )

    hist2d_norm = getcolordensity(force_direct_mse_list, uncertainty_ms_list)
    plot_scatter(
        x=force_direct_mse_list,
        y=uncertainty_ms_list,
        hist2d_norm=hist2d_norm,
        xlabel="Force Error",
        ylabel="Uncertainty",
        title="Uncertainty vs Force (Direct) Error Scatter Plot",
        filename="./force_direct_uncertainty_Scatterplot.png",
    )

    hist2d_norm = getcolordensity(force_grad_mse_list, uncertainty_ms_list)
    plot_scatter(
        x=force_grad_mse_list,
        y=uncertainty_ms_list,
        hist2d_norm=hist2d_norm,
        xlabel="Force Error",
        ylabel="Uncertainty",
        title="Uncertainty vs Force (Grad) Error Scatter Plot",
        filename="./force_grad_uncertainty_Scatterplot.png",
    )

    analyze_correlations(
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


if __name__ == "__main__":
    main()
