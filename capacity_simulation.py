import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import logging

from memory_recall import run_recall_hyperbolic
from baseline_recall import run_recall_dam, run_recall_mhn


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


    args = ap.parse_args()
    return args


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("getting args")

    args = get_args()

    result_data = {"model": [], "recall rate": [], "M": []}

    logger.info("Start running")

    for M in range(args.M_min, args.M_max, args.M_step):

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

            dam_rate = run_recall_dam(args, M, int(i + M))
            mhn_rate = run_recall_dam(args, M, int(i + M))

            result_data["model"].append("DAM")
            result_data["model"].append("MHN")

            result_data["recall rate"].append(dam_rate)
            result_data["recall rate"].append(mhn_rate)

            result_data["M"].append(M)
            result_data["M"].append(M)

    df = pd.DataFrame(result_data)
    output_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/result.csv") ) 
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir)

    plt.figure(figsize=(8, 5))

    ax = sns.lineplot(
        data=df,
        x="M",
        y="recall rate",
        hue="model",
        errorbar="sd",
        alpha=0.7
    )

    # Title and labels
    ax.set_title("Recall Rate vs M")
    ax.set_xlabel("M")
    ax.set_ylabel("Recall rate")

    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(title="Model")

    plt.tight_layout()

    plot_dir = Path(args.output_dir + str("/dim") + str(args.d) + str("/Radius") + str(args.mem_R) + str("/recall_plot.png") ) 
    plot_dir.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_dir, dpi=480)
    plt.close()


main()