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

    ap.add_argument("--d", type=int, default=20)
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

    args = ap.parse_args()
    return args


def set_safe(self, **kwargs):
    if 'xscale' in kwargs and kwargs['xscale'] == 'log':
        raise ValueError('Setting log scale on a categorical axis is '
                         'not valid.')
    return self.set(**kwargs)


def run_replot(args):

    output_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/result.csv") ) 
    df = pd.read_csv(output_dir)

    plt.figure(figsize=(6, 5))

    custom_palette = {'MHN': '#E4572E', 'DAM': '#29335C', 'identity':'#F3A712', 'geo_distance':'#A8C686', 'square_distance': '#669BBC'}

    ax = sns.lineplot(data=df, x="M", y="recall rate", hue="model", errorbar='sd', marker='o',markersize=10, alpha=0.7, palette=custom_palette, err_style='bars')# .scale(x="log") # style="Geometry", 
    
    # set_safe(ax, xscale="log")  # <-- comment/uncomment this line to see the issue

    ax.set_xscale("log")
    # ax.set_xticks("log")

    # Title and labels
    ax.set_title(f"Recall Rate vs M (d={args.d})")
    ax.set_xlabel("M")
    ax.set_ylabel("Recall rate")

    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs='all'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(title="Model")

    plt.tight_layout()

    plot_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/recall_plot.png") ) 
    plot_dir.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_dir, dpi=480)
    plt.close()





def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("getting args")

    args = get_args()


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
                geo_dist_phi_rate = run_recall_hyperbolic(
                    args, "geo_distance", M, int(i + M)
                )
                square_dist_phi_rate = run_recall_hyperbolic(
                    args, "square_distance", M, int(i + M)
                )

                result_data["model"].append("identity")
                result_data["model"].append("geo_distance")
                result_data["model"].append("square_distance")

                result_data["recall rate"].append(identity_phi_rate)
                result_data["recall rate"].append(geo_dist_phi_rate)
                result_data["recall rate"].append(square_dist_phi_rate)

                result_data["M"].append(M)
                result_data["M"].append(M)
                result_data["M"].append(M)

                result_data["Geometry"].append("H")
                result_data["Geometry"].append("H")
                result_data["Geometry"].append("H")


                dam_rate = run_recall_dam(args, M, int(i + M))
                mhn_rate = run_recall_dam(args, M, int(i + M))

                result_data["model"].append("DAM")
                result_data["model"].append("MHN")

                result_data["recall rate"].append(dam_rate)
                result_data["recall rate"].append(mhn_rate)

                result_data["M"].append(M)
                result_data["M"].append(M)

                result_data["Geometry"].append("E")
                result_data["Geometry"].append("E")

        df = pd.DataFrame(result_data)
        output_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/result.csv") ) 
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir)

        plt.figure(figsize=(8, 5))

        custom_palette = {'MHN': '#37373E', 'DAM': '#86928B', 'identity':'#FF6B6B', 'geo_distance':'#FFAC6B', 'square_distance': '#FBB7C0'}

        ax = sns.lineplot(data=df, x="M", y="recall rate", hue="model", errorbar='sd', marker='o',markersize=10, alpha=0.7, palette=custom_palette, err_style='bars', style="Geometry", ) # style="Geometry", 

        # Title and labels
        ax.set_title("Recall Rate vs M")
        ax.set_xlabel("M")
        ax.set_ylabel("Recall rate")

        ax.set_xscale("log")

        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(title="Model")

        plt.title(f"Recall Rate vs M (d={args.d})")

        plt.tight_layout()

        plot_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/recall_plot.png") ) 
        plot_dir.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(plot_dir, dpi=480)
        plt.close()


main()