import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

from pathlib import Path
import logging

from memory_recall import run_recall_hyperbolic
from baseline_recall import run_recall_dam, run_recall_mhn


import warnings
warnings.filterwarnings(
    "error",
    message=".*invalid value encountered in sqrt.*",
    category=RuntimeWarning,
)

def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="outputs")

    ap.add_argument("--dataset", type=str, default="synthetic", 
                    choices=["synthetic", "mnist", "cifar10"],
                    help="Dataset to use: synthetic, mnist, or cifar10")
    ap.add_argument("--d", type=int, default=20,
                    help="Dimension for synthetic, or PCA dimension for images (if specified)")
    ap.add_argument("--pca-dim", type=int, default=None,
                    help="Optional: PCA dimension for image datasets (overrides --d)")
    ap.add_argument("--M-min", type=int, default=100)
    ap.add_argument("--M-max", type=int, default=150)
    ap.add_argument("--M-step", type=int, default=10)
    ap.add_argument("--mem-R", type=int, default=2)

    ap.add_argument("--n-trials", type=int, default=5)

    ap.add_argument("--max-steps", type=int, default=10)

    ap.add_argument("--kappa", type=int, default=-1)
    ap.add_argument("--noise_sigma", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--tol", type=float, default=0.001)

    ap.add_argument("--replot", action="store_true",)
    ap.add_argument("--plot-3x3", action="store_true", help="Generate 3x3 figure with all datasets and dimensions")

    args = ap.parse_args()
    return args


    args = ap.parse_args()
    return args


def set_safe(self, **kwargs):
    if 'xscale' in kwargs and kwargs['xscale'] == 'log':
        raise ValueError('Setting log scale on a categorical axis is '
                         'not valid.')
    return self.set(**kwargs)



def run_replot(args):

    dataset = getattr(args, "dataset", "synthetic")
    if dataset == "synthetic":
        output_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/result.csv"))
    else:
        dim_str = str(args.pca_dim) if args.pca_dim else str(args.d)
        output_dir = Path(args.output_dir + str("/") + dataset + str("/dim") + dim_str + str("/result.csv"))
    df = pd.read_csv(output_dir)

    plt.figure(figsize=(6, 5))

    custom_palette = {'MHN': '#E4572E', 'DAM': '#29335C', 'identity':'#F3A712', 'geo_distance':'#A8C686', 'square_distance': '#669BBC'}

    ax = sns.lineplot(data=df, x="M", y="recall rate", hue="model", errorbar='sd', marker='o',markersize=10, alpha=0.7, palette=custom_palette, err_style='bars')# .scale(x="log") # style="Geometry", 
    
    # set_safe(ax, xscale="log")  # <-- comment/uncomment this line to see the issue

    ax.set_xscale("log")
    # ax.set_xticks("log")

    # Title and labels
    dataset = getattr(args, "dataset", "synthetic")
    dim_str = str(args.pca_dim) if (dataset != "synthetic" and args.pca_dim) else str(args.d)
    title_str = f"Recall Rate vs M ({dataset}, d={dim_str})" if dataset != "synthetic" else f"Recall Rate vs M (d={args.d})"
    ax.set_title(title_str)
    ax.set_xlabel("M")
    ax.set_ylabel("Recall rate")

    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs='all'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.legend(title="Model")

    plt.tight_layout()

    if dataset == "synthetic":
        plot_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/recall_plot.png"))
    else:
        plot_dir = Path(args.output_dir + str("/") + dataset + str("/dim") + dim_str + str("/recall_plot.png")) 
    plot_dir.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_dir, dpi=480)
    plt.close()


def plot_3x3_figure(args):
    """Generate a 3x3 figure with synthetic, MNIST, and CIFAR10 data for dims 10, 20, 100"""
    
    # Set up the figure with 8.5 inches width
    fig_width = 8.5
    fig_height = fig_width * 0.5 # Adjust height proportionally
    fig, axes = plt.subplots(3, 3, figsize=(fig_width, fig_height))
    
    # Set font size to 24
    plt.rcParams.update({'font.size': 20})
    
    # Custom palette matching the existing code
    # custom_palette = {'MHN': '#37373E', 'DAM': '#86928B', 'Karcher-Flow':'#FF6B6B'}
    
    # Define datasets and dimensions
    datasets = ['synthetic', 'mnist', 'cifar10']
    dims = [10, 20, 100]
    
    # Track legend - only show once
    legend_shown = False
    
    for row, dataset in enumerate(datasets):
        for col, dim in enumerate(dims):
            ax = axes[row, col]
            
            # Load CSV file
            if dataset == "synthetic":
                csv_path = Path(args.output_dir) / f"dim{dim}" / f"Radius{args.mem_R}" / "result.csv"
            else:
                csv_path = Path(args.output_dir) / dataset / f"dim{dim}" / "result.csv"
            
            if not csv_path.exists():
                ax.text(0.5, 0.5, f"CSV not found:\n{csv_path}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=16)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            df = pd.read_csv(csv_path)
            
            # Plot the data
            sns.lineplot(data=df, x="M", y="recall rate", hue="model", 
                        errorbar='sd', marker='o', markersize=8, alpha=0.7, 
                        err_style='bars', ax=ax)
            
            # Set log scale
            ax.set_xscale("log")
            
            # Remove title
            ax.set_title("")
            
            # Only rightmost column, middle row has y-label on the right
            if col == 0 and row == 1:
                ax.set_ylabel("Recall rate", fontsize=20)
            else:
                ax.set_ylabel("")
            
            # Only bottom row has x-label
            if row == 2 and col == 1:
                ax.set_xlabel("M", fontsize=20)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            
            # Add legend only on the middle top figure (row 0, col 1)
            if row == 0 and col == 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.73), ncol=3)
            else:
                ax.legend().remove()
            
            # Set tick label sizes
            ax.tick_params(labelsize=20)
    
    # Reduce spacing between subplots
    # plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)
    
    # Save the figure
    output_path =  "./3x3_capacity_figure_r3.png"
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=480, bbox_inches='tight')
    plt.close()
    
    print(f"3x3 figure saved to {output_path}")



def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("getting args")

    args = get_args()
    # plot_3x3_figure(args)
    # raise Exception

    sns.set_theme(rc={
        'axes.titlesize': 20,    # Title font size
        'axes.labelsize': 20,    # X and Y label font size
        'legend.fontsize': 16,   # Legend text font size
        'legend.title_fontsize': 18, # Legend title font size
        'xtick.labelsize': 16,   # X tick label font size
        'ytick.labelsize': 16,    # Y tick label font size
        "font.family": "serif",
        "font.serif": ["Courier New"]
    })
    sns.set_style("whitegrid") 

    if args.replot is True:
        logger.info("Re generating figures...")
        run_replot(args)

    else:

        result_data = {"model": [], "recall rate": [], "M": [], "Geometry":[]}

        logger.info("Start running")

        M_min, M_max = args.M_min, args.M_max
        K = 15  # number of intervals
        r = (M_max / M_min) ** (1 / K)

        M_values = M_min * r ** np.arange(K + 1)
        M_values = np.round(M_values).astype(int)


        for M in M_values:

            for i in range(args.n_trials):

                identity_phi_rate = run_recall_hyperbolic(args, "identity", M, int(i + M))
                result_data["model"].append("Karcher-Flow")
                # result_data["model"].append("geo_distance")
                # result_data["model"].append("square_distance")

                result_data["recall rate"].append(identity_phi_rate)
                # result_data["recall rate"].append(geo_dist_phi_rate)
                # result_data["recall rate"].append(square_dist_phi_rate)

                result_data["M"].append(M)
                # result_data["M"].append(M)
                # result_data["M"].append(M)

                result_data["Geometry"].append("H")
                # result_data["Geometry"].append("H")
                # result_data["Geometry"].append("H")

                dam_rate = run_recall_dam(args, M, int(i + M))
                mhn_rate = run_recall_mhn(args, M, int(i + M))

                result_data["model"].append("DAM")
                result_data["model"].append("MHN")

                result_data["recall rate"].append(dam_rate)
                result_data["recall rate"].append(mhn_rate)

                result_data["M"].append(M)
                result_data["M"].append(M)

                result_data["Geometry"].append("E")
                result_data["Geometry"].append("E")

        df = pd.DataFrame(result_data)
        dataset = getattr(args, "dataset", "synthetic")
        if dataset == "synthetic":
            output_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/result.csv"))
        else:
            dim_str = str(args.pca_dim) if args.pca_dim else str(args.d)
            output_dir = Path(args.output_dir + str("/") + dataset + str("/dim") + dim_str + str("/result.csv"))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir)

        plt.figure(figsize=(8, 5))

        custom_palette = {'MHN': '#37373E', 'DAM': '#86928B', 'Karcher-Flow':'#FF6B6B', 'geo_distance':'#FFAC6B', 'square_distance': '#FBB7C0'}

        ax = sns.lineplot(data=df, x="M", y="recall rate", hue="model", errorbar='sd', marker='o',markersize=10, alpha=0.7, palette=custom_palette, err_style='bars') # style="Geometry", 

        # Title and labels
        dim_str = str(args.pca_dim) if (dataset != "synthetic" and args.pca_dim) else str(args.d)
        title_str = f"Recall Rate vs M ({dataset}, d={dim_str})" if dataset != "synthetic" else f"Recall Rate vs M (d={args.d})"
        ax.set_title(title_str)
        ax.set_xlabel("M")
        ax.set_ylabel("Recall rate")

        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        plt.tight_layout(pad=0.1)

        if dataset == "synthetic":
            plot_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/recall_plot.png"))
        else:
            plot_dir = Path(args.output_dir + str("/") + dataset + str("/dim") + dim_str + str("/recall_plot.png"))
        plot_dir.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(plot_dir, dpi=480)
        plt.close()


main()