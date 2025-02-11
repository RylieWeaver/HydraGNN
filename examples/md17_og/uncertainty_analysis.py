import argparse
import os
import sys
import torch
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, auc
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
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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


def do_sensitivity_analysis(
    errors, uncertainties, subset_percentages, error_percentiles
):
    """
    For a given error array (errors) and uncertainty array (uncertainties),
    compute the multiplier c for each subset of top-X% errors and for each
    percentile in error_percentiles.
    Returns a dict of structure:
    {
       subset_size1: { error_pct1: c_value, error_pct2: c_value, ... },
       subset_size2: { error_pct1: c_value, error_pct2: c_value, ... },
       ...
    }
    """
    # Convert to numpy arrays if needed
    errors = np.array(errors)
    uncertainties = np.array(uncertainties)

    # Sort descending by error
    sorted_indices = np.argsort(errors)[::-1]
    errors_sorted = errors[sorted_indices]
    uncertainties_sorted = uncertainties[sorted_indices]

    results = {}
    total_count = len(errors_sorted)

    for sp in subset_percentages:
        # Number of entries to keep
        count_to_keep = max(1, int(np.ceil(total_count * sp / 100.0)))

        # Subset
        e_sub = errors_sorted[:count_to_keep]
        u_sub = uncertainties_sorted[:count_to_keep]

        # Compute ratio
        ratio = e_sub / (u_sub + 1e-15)  # small epsilon to avoid /0
        ratio_sorted = np.sort(ratio)

        # For each percentile p, find the c such that p% of ratio <= c
        # i.e. c = np.percentile(ratio, p)
        results_for_sp = {}
        for p in error_percentiles:
            c_val = np.percentile(ratio_sorted, p)
            results_for_sp[p] = c_val

        results[sp] = results_for_sp

    return results


def plot_sensitivity_analysis(sensitivity_results, descriptor, error_percentiles):
    """
    Plot the multiplier c vs. subset percentage for each percentile line.
    sensitivity_results is the dict returned by do_sensitivity_analysis().
    descriptor is a string like "Direct" or "Grad" to label the plot.
    """
    # X-values are sorted in ascending order, so the largest subset (100) is on the left.
    x_vals = sorted(sensitivity_results.keys())

    plt.figure(figsize=(12, 6))

    for p in error_percentiles:
        y_vals = [sensitivity_results[sp][p] for sp in x_vals]
        plt.plot(x_vals, y_vals, marker="o", label=f"{p}% Coverage")

    # Invert x-axis so it goes from 100 down to lower values on the right
    plt.gca().invert_xaxis()

    plt.title(f"Sensitivity Analysis for {descriptor} MSE and Uncertainty")
    plt.xlabel("Top X% of Structures by Error")
    plt.ylabel("Multiplier c s.t. P(Error <= c * Uncertainty)")

    # Move legend to the right, outside the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.grid(True)
    plt.tight_layout()

    # Save and close
    plt.savefig(
        f"plots/sensitivity_analysis_{descriptor.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved sensitivity plot for {descriptor} as: plots/sensitivity_analysis_{descriptor.lower()}.png"
    )


def plot_roc_like_curve(
    error_list, uncertainty_list, descriptor, percentile, c_values=None
):
    """
    Create an ROC-like curve by:
      1) Defining the "true" label as (error > T), where T is a user-chosen threshold,
         e.g. the 10th percentile of error_list.
      2) For each c in c_values, define the predicted label as (c * uncertainty > T).
      3) Compute TPR = P(pred=1 | true=1), FPR = P(pred=1 | true=0).
      4) Plot TPR vs FPR as we vary c.
    """
    errors = np.array(error_list)
    uncertainties = np.array(uncertainty_list)

    # Dynamically set c_values if not provided
    if c_values is None:
        ratio = errors / (uncertainties + 1e-15)  # small epsilon to avoid div-by-zero
        c_values = np.percentile(ratio, np.linspace(0, 100, 101))

    # Define threshold as the percentile of the error distribution
    threshold = np.percentile(errors, percentile)

    # We define the "true" label as (error > threshold). This is what we want to indentify.
    true_label = (errors > threshold).astype(int)

    pos_count = np.sum(true_label == 1)  # number of errors above threshold
    neg_count = np.sum(true_label == 0)  # # of "small error" items

    tpr_list = []
    fpr_list = []

    for c in c_values:
        # Predicted label: We predict loss above the threshold if c * uncertainty > threshold
        pred_label = (c * uncertainties > threshold).astype(int)

        # True Positives = pred=1 & true=1
        tp = np.sum((pred_label == 1) & (true_label == 1))
        # False Positives = pred=1 & true=0
        fp = np.sum((pred_label == 1) & (true_label == 0))

        # Compute TPR, FPR
        tpr = tp / pos_count if pos_count > 0 else 0.0
        fpr = fp / neg_count if neg_count > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Show AUC
    roc_auc = auc(fpr_list, tpr_list)
    print(f"AUC = {roc_auc:.4f}")

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_list, tpr_list, marker="o", markersize=3, label="ROC curve")
    plt.plot(
        [0, 1], [0, 1], "--", color="gray", label="Random classifier"
    )  # reference line
    plt.title(f"ROC Curve\n(Threshold={percentile}th pct, {descriptor} error)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Assume the TPR, FPR, and c_values lists are in order of ascending c
    target_tprs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Copy so we can pop
    remaining_targets = target_tprs.copy()
    # Use a 'first-pass' labelling system for c-values
    for c, tpr, fpr in zip(c_values, tpr_list, fpr_list):
        # Check if we've crossed any remaining target TPR(s)
        while remaining_targets and (tpr >= remaining_targets[0]):
            # "Pass" the first target
            passed_threshold = remaining_targets.pop(0)
            # Annotate at (fpr, tpr) with c and the threshold
            plt.annotate(
                f"c={c:.1f}",
                xy=(fpr, tpr),
                xytext=(5, 5),  # small offset, if desired
                textcoords="offset points",
                ha="left",
            )
        # If we've labeled all targets, break early
        if not remaining_targets:
            break

    fname = f"plots/roc_{descriptor.lower()}_{percentile}pct.png"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved ROC plot for {descriptor} to {fname}")


def main():
    # Define the paths to the serialized dataset
    try:
        os.environ["SERIALIZED_DATA_PATH"]
    except:
        os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    serialized_data_path = os.environ.get("SERIALIZED_DATA_PATH")

    # Load dataset
    dataset = torch.load(
        os.path.join(serialized_data_path, "test_dataset_predictions.pt")
    )

    # Calculate Metrics for all follow-on tasks
    (
        energy_mse_list,
        force_direct_mse_list,
        force_grad_mse_list,
        force_direct_max_list,
        force_grad_max_list,
        uncertainty_min_list,
        uncertainty_max_list,
        uncertainty_ma_list,
        uncertainty_ms_list,
        uncertainty_mc_list,
        uncertainty_log_list,
        uncertainty_exp_list,
    ) = calculate_metrics(dataset)

    # # Plot Energy Error vs Uncertainty Error in normal scale
    # hist2d_norm = getcolordensity(energy_mse_list, uncertainty_ms_list)
    # plot_scatter(
    #     x=energy_mse_list,
    #     y=uncertainty_ms_list,
    #     hist2d_norm=hist2d_norm,
    #     xlabel="Energy Error",
    #     ylabel="Uncertainty",
    #     title="Uncertainty vs Energy Error Scatter Plot",
    #     filename="plots/energy_uncertainty_Scatterplot.png",
    # )

    # hist2d_norm = getcolordensity(force_direct_mse_list, uncertainty_ms_list)
    # plot_scatter(
    #     x=force_direct_mse_list,
    #     y=uncertainty_ms_list,
    #     hist2d_norm=hist2d_norm,
    #     xlabel="Force Error",
    #     ylabel="Uncertainty",
    #     title="Uncertainty vs Force (Direct) Error Scatter Plot",
    #     filename="plots/force_direct_uncertainty_Scatterplot.png",
    # )

    # hist2d_norm = getcolordensity(force_grad_mse_list, uncertainty_ms_list)
    # plot_scatter(
    #     x=force_grad_mse_list,
    #     y=uncertainty_ms_list,
    #     hist2d_norm=hist2d_norm,
    #     xlabel="Force Error",
    #     ylabel="Uncertainty",
    #     title="Uncertainty vs Force (Grad) Error Scatter Plot",
    #     filename="plots/force_grad_uncertainty_Scatterplot.png",
    # )

    # Make a plot of forces_direct error vs forces_grad error and calculate R2
    # hist2d_norm = getcolordensity(force_direct_mse_list, force_grad_mse_list)
    # plot_scatter(
    #     x=force_direct_mse_list,
    #     y=force_grad_mse_list,
    #     hist2d_norm=hist2d_norm,
    #     xlabel="Force Error (Direct)",
    #     ylabel="Force Error (Grad)",
    #     title="Force (Direct) vs Force (Grad) Error Scatter Plot",
    #     filename="plots/force_direct_vs_force_grad_Scatterplot.png",
    # )
    # forces_direct_grad_r2 = r2_score(force_direct_mse_list, force_grad_mse_list)
    # print(f"R2 Score between forces_direct and forces_grad: {forces_direct_grad_r2}")

    # # Define the subset percentages (top X% by error) to investigate and the error percentiles to consider
    # subset_percentages = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 3, 2, 1]
    # error_percentiles = [99, 95, 90, 75, 50, 25, 10, 5, 1]

    # # Perform the analysis for force_direct
    # direct_sensitivity = do_sensitivity_analysis(
    #     errors=force_direct_mse_list,
    #     uncertainties=uncertainty_ms_list,
    #     subset_percentages=subset_percentages,
    #     error_percentiles=error_percentiles,
    # )
    # plot_sensitivity_analysis(direct_sensitivity, descriptor="Direct", error_percentiles=error_percentiles)

    # # Perform the analysis for force_grad
    # grad_sensitivity = do_sensitivity_analysis(
    #     errors=force_grad_mse_list,
    #     uncertainties=uncertainty_ms_list,
    #     subset_percentages=subset_percentages,
    #     error_percentiles=error_percentiles,
    # )
    # plot_sensitivity_analysis(grad_sensitivity, descriptor="Grad", error_percentiles=error_percentiles)

    # ROC-like plots
    plot_roc_like_curve(
        error_list=force_grad_mse_list,
        uncertainty_list=uncertainty_ms_list,
        descriptor="Grad",
        percentile=95,
    )
    plot_roc_like_curve(
        error_list=force_direct_mse_list,
        uncertainty_list=uncertainty_ms_list,
        descriptor="Direct",
        percentile=95,
    )


if __name__ == "__main__":
    main()


# Do a full sensitivity analysis of uncertainty multiples that cover percentiles of the errors for grad and for direct for various percentiles of the error (elaborated below)
## We have an uncertainty, forces_direct_mse, and forces_grad_mse for each structure. I want to see what the multiple 'c' of uncertainty is such that 95% of the forces_direct_mse
## is less than c*uncertainty. Note that the uncertainty and both force error metrics are all paired together. So, I'm trying the find the 'c' such that 90 percent of the errors are
## less than the c*uncertainty for their same structure. Then, I want to find that 'c' for 90% of the forces_direct_mse, 99%, 75%, 50%, 25%, 10%, 5%, 1%. This will result in a list
## of points that will each be the first point of a line in a plot, where the y-axis is the multiple 'c' and the x-axis is 100. Now, notice that the x-axis is the same and 100 for all
## all of them. This is because we are looking at the same 100% of the data, but I also want to do this analysis for other percentiles of the force error to see if it persists. So, take
## the structures with the top 90% of force_direct errors and do the same analysis... find c for that list of percentiles. This will be the second point for each line of our plot, all
## x-coordinate 90. Do this for 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%, 5%, 4%, 3%, 2%, and 1%. This will give us a plot that shows the 'c' for different percentiles of the force error.
## This will be a good way to see if our trend of the error being probabilistically bounded by the uncertainty holds for different percentiles of the error.
