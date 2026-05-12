import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

import matplotlib

from pathlib import Path
import logging

#from memory_recall import run_recall_hyperbolic
#from baseline_recall import run_recall_dam, run_recall_mhn
from matplotlib.gridspec import GridSpec
from pathlib import Path

import warnings
warnings.filterwarnings(
    "error",
    message=".*invalid value encountered in sqrt.*",
    category=RuntimeWarning,
)

def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="./hyp_sim/outputs")

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

    # Overall figure
    fig = plt.figure(figsize=(1800/300 * 1.4, 600/300))  # wider figure

    # Outer grid: left (plots) + right (blank)
    outer_gs = GridSpec(
        1, 2,
        width_ratios=[3, 1],   # left big, right reserved
        wspace=0.05
    )

    # Left panel: 3x3 grid
    left_gs = outer_gs[0].subgridspec(3, 3, hspace=0.15, wspace=0.15)
    axes = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            axes[i, j] = fig.add_subplot(left_gs[i, j])

    # Right panel: blank axis
    ax_right = fig.add_subplot(outer_gs[1])
    ax_right.axis("off")   # keep it empty

    plt.rcParams.update({'font.size': 20})

    datasets = ['synthetic', 'mnist', 'cifar10']
    dims = [10, 20, 100]

    for row, dataset in enumerate(datasets):
        for col, dim in enumerate(dims):
            ax = axes[row, col]

            if dataset == "synthetic":
                csv_path = Path(args.output_dir) / f"dim{dim}" / f"Radius{args.mem_R}" / "result.csv"
            else:
                csv_path = Path(args.output_dir) / dataset / f"dim{dim}" / "result.csv"

            if not csv_path.exists():
                ax.text(0.5, 0.5, "CSV not found",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            df = pd.read_csv(csv_path)

            for model, g in df.groupby("model"):
                g = g.sort_values("M")
                x = g["M"].to_numpy()

                y_mean = g.groupby("M")["recall rate"].mean().to_numpy()
                y_sd   = g.groupby("M")["recall rate"].std(ddof=1).to_numpy()

                ax.errorbar(
                    x=np.unique(x),
                    y=y_mean,
                    yerr=y_sd,
                    fmt="-o",
                    markersize=1,
                    linewidth=1,
                    elinewidth=0.5,
                    capsize=1,
                    label=str(model),
                )

            ax.set_xscale("log")
            ax.legend().remove()
            ax.tick_params(labelsize=8)

            if col == 0:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0.0", "", "0.5", "", "1.0"], fontsize=8)
            else:
                ax.set_yticks([])
            
            if row == 2:
                ax.set_xlabel("M", fontsize=8)
            else:
                ax.tick_params(labelbottom=False)

    plt.savefig("./3x3_capacity_with_blank_panel.png",
                dpi=300, bbox_inches="tight")
    plt.close()




def plot_combined_3x3_plus_right(args, data_dir="./dim3", out_path="./combined.png"):
    # ---- Global style (apply once) ----
    font_size = 10
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })

    # ---- Figure + outer layout: [left 3x3] + [right panel] ----
    fig = plt.figure(figsize=(2200/300, 600/300))  # adjust as needed
    outer = GridSpec(1, 2, figure=fig, width_ratios=[3.2, 2.0], wspace=0.2)

    # =========================
    # Left panel: 3x3 grid
    # =========================
    left_gs = outer[0].subgridspec(3, 3, hspace=0.20, wspace=0.20)
    axes_left = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            axes_left[r, c] = fig.add_subplot(left_gs[r, c])

    datasets = ["synthetic", "mnist", "cifar10"]
    dims = [10, 20, 100]

    for row, dataset in enumerate(datasets):
        for col, dim in enumerate(dims):
            ax = axes_left[row, col]

            if dataset == "synthetic":
                csv_path = Path(args.output_dir) / f"dim{dim}" / f"Radius{args.mem_R}" / "result.csv"
            else:
                csv_path = Path(args.output_dir) / dataset / f"dim{dim}" / "result.csv"

            if not csv_path.exists():
                ax.text(0.5, 0.5, f"CSV not found:\n{csv_path}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            df = pd.read_csv(csv_path)

            for model, g in df.groupby("model"):
                g = g.sort_values("M")
                x = g["M"].to_numpy()

                # mean/std over repeats per M
                y_mean = g.groupby("M")["recall rate"].mean().reindex(np.unique(x)).to_numpy()
                y_sd   = g.groupby("M")["recall rate"].std(ddof=1).reindex(np.unique(x)).to_numpy()

                ax.errorbar(
                    x=np.unique(x),
                    y=y_mean,
                    yerr=y_sd,
                    fmt="-o",
                    markersize=1,
                    alpha=1,
                    capsize=1,
                    label=str(model),
                    linewidth=1,
                    elinewidth=0.5,
                )

            ax.set_xscale("log")
            ax.set_title("")

            # y label only at (row==1, col==0) like your original intent
            if col == 0 and row == 1:
                ax.set_ylabel("Recall rate", fontsize=10)
            else:
                ax.set_ylabel("")

            if col == 0:
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0.0", "", "0.5", "", "1.0"], fontsize=10)
            else:
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels([])

            if row == 2:
                ax.set_xlabel("M", fontsize=10)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            ax.legend().remove()
            ax.tick_params(labelsize=10)

    # =========================
    # Right panel: 1x2 (Hyperbolic vs Euclidean)
    # =========================
    right_gs = outer[1].subgridspec(1, 2, wspace=0.15)
    ax1 = fig.add_subplot(right_gs[0, 0])
    ax2 = fig.add_subplot(right_gs[0, 1], sharey=ax1)

    # --- Load + aggregate right-panel data ---
    all_data = []
    for root, dirs, files in os.walk(data_dir):
        if "result.csv" in files:
            folder_name = os.path.basename(root)
            try:
                r_val = int(folder_name.replace("Radius", ""))
            except ValueError:
                continue
            df = pd.read_csv(os.path.join(root, "result.csv"))
            df["R"] = r_val
            all_data.append(df)

    if all_data:
        df_total = pd.concat(all_data, ignore_index=True)
        stats = (
            df_total.groupby(["model", "R", "M"])["recall rate"]
            .agg(["mean", "std"])
            .reset_index()
        )

        radii = sorted(stats["R"].unique())
        colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(radii)))
        model_markers = {"Karcher-Flow": "o", "DAM": "s", "MHN": "^"}

        # --- Left of right-panel: Karcher-Flow (Hyperbolic) ---
        kf = stats[stats["model"] == "Karcher-Flow"]
        for i, r in enumerate(radii):
            sub = kf[kf["R"] == r].sort_values("M")
            if sub.empty:
                continue
            ax1.plot(sub["M"], sub["mean"], color=colors[i],
                     marker=model_markers["Karcher-Flow"], linewidth=1, ms=1)
            ax1.fill_between(sub["M"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                             color=colors[i], alpha=0.2)

        ax1.set_xscale("log")
        #ax1.set_title("Hyperbolic")
        ax1.set_xlabel("M")
        #ax1.set_ylabel("Recall Rate")
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # --- Right of right-panel: DAM & MHN (Euclidean) ---
        for i, r in enumerate(radii):
            dam = stats[(stats["model"] == "DAM") & (stats["R"] == r)].sort_values("M")
            if not dam.empty:
                ax2.plot(dam["M"], dam["mean"], color=colors[i],
                         marker=model_markers["DAM"], linestyle="-", alpha=0.8, linewidth=1, ms=1)

            mhn = stats[(stats["model"] == "MHN") & (stats["R"] == r)].sort_values("M")
            if not mhn.empty:
                ax2.plot(mhn["M"], mhn["mean"], color=colors[i],
                         marker=model_markers["MHN"], linestyle="--", alpha=0.8, linewidth=1, ms=1)

        ax2.set_xscale("log")
        #ax2.set_title("Euclidean")
        ax2.set_xlabel("M")
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        # If you want right-panel y tick labels only on ax1:
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.tick_params(axis="y", length=0)

        ax1.set_yticks([0.0, 0.25, 0.5, 0.75])
        ax1.set_yticklabels(["0.0", "", "", "0.75"], fontsize=10)

    else:
        ax1.axis("off")
        ax2.axis("off")
        ax1.text(0.5, 0.5, f"No data found in {data_dir}", ha="center", va="center",
                 transform=ax1.transAxes)

    # ---- Save ----
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("getting args")

    args = get_args()
    plot_combined_3x3_plus_right(args)
    exit()

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