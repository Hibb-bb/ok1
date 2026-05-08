import recall_config

recall_config.early_set_geomstats_backend_from_argv()

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from recall_config import image_feature_dir, image_title_dim

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

    ap.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "mnist", "cifar10"],
        help="Dataset to use: synthetic, mnist, or cifar10",
    )
    ap.add_argument(
        "--d",
        type=int,
        default=20,
        help="Dimension for synthetic, or default PCA dim for images if --pca-dim omitted",
    )
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="PCA dimension for images; ignored if --no-pca",
    )
    ap.add_argument(
        "--no-pca",
        action="store_true",
        help="MNIST/CIFAR: use raw pixels (784 / 3072) after R scaling, no PCA",
    )
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument(
        "--no-batch",
        action="store_true",
        help="Euclidean recall: scalar loop instead of batched torch (debug)",
    )
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

    ap.add_argument("--replot", action="store_true")
    ap.add_argument(
        "--plot-3x3",
        action="store_true",
        help="Generate 3x3 figure with all datasets and dimensions",
    )

    args = ap.parse_args()
    args.use_batch = not args.no_batch
    return args


def _image_csv_plot_paths(args):
    feat = image_feature_dir(args)
    base = Path(args.output_dir) / args.dataset / feat / f"Radius{args.mem_R}"
    return base / "result.csv", base / "recall_plot.png"


def run_replot(args):

    dataset = getattr(args, "dataset", "synthetic")
    if dataset == "synthetic":
        output_dir = Path(
            args.output_dir
            + str("/dim")
            + str(args.d)
            + str("/Radius")
            + str(args.mem_R)
            + str("/result.csv")
        )
    else:
        output_dir, _ = _image_csv_plot_paths(args)
    df = pd.read_csv(output_dir)

    plt.figure(figsize=(6, 5))

    custom_palette = {
        "MHN": "#E4572E",
        "DAM": "#29335C",
        "identity": "#F3A712",
        "geo_distance": "#A8C686",
        "square_distance": "#669BBC",
        "Karcher-Flow": "#FF6B6B",
    }

    ax = sns.lineplot(
        data=df,
        x="M",
        y="recall rate",
        hue="model",
        errorbar="sd",
        marker="o",
        markersize=10,
        alpha=0.7,
        palette=custom_palette,
        err_style="bars",
    )

    ax.set_xscale("log")

    if dataset != "synthetic":
        dim_str = image_title_dim(args)
        title_str = f"Recall Rate vs M ({dataset}, d={dim_str})"
    else:
        title_str = f"Recall Rate vs M (d={args.d})"
    ax.set_title(title_str)
    ax.set_xlabel("M")
    ax.set_ylabel("Recall rate")

    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs="all"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.legend(title="Model")

    plt.tight_layout()

    if dataset == "synthetic":
        plot_dir = Path(
            args.output_dir
            + str("/dim")
            + str(args.d)
            + str("/Radius")
            + str(args.mem_R)
            + str("/recall_plot.png")
        )
    else:
        _, plot_dir = _image_csv_plot_paths(args)
    plot_dir.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_dir, dpi=480)
    plt.close()


def plot_3x3_figure(args):
    fig_width = 8.5
    fig_height = fig_width * 0.5
    fig, axes = plt.subplots(3, 3, figsize=(fig_width, fig_height))

    plt.rcParams.update({"font.size": 20})

    datasets = ["synthetic", "mnist", "cifar10"]
    dims = [10, 20, 100]

    for row, dataset in enumerate(datasets):
        for col, dim in enumerate(dims):
            ax = axes[row, col]

            if dataset == "synthetic":
                csv_path = (
                    Path(args.output_dir) / f"dim{dim}" / f"Radius{args.mem_R}" / "result.csv"
                )
            else:
                csv_path = (
                    Path(args.output_dir)
                    / dataset
                    / f"pca{dim}"
                    / f"Radius{args.mem_R}"
                    / "result.csv"
                )

            if not csv_path.exists():
                ax.text(
                    0.5,
                    0.5,
                    f"CSV not found:\n{csv_path}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            df = pd.read_csv(csv_path)

            sns.lineplot(
                data=df,
                x="M",
                y="recall rate",
                hue="model",
                errorbar="sd",
                marker="o",
                markersize=8,
                alpha=0.7,
                err_style="bars",
                ax=ax,
            )

            ax.set_xscale("log")
            ax.set_title("")

            if col == 0 and row == 1:
                ax.set_ylabel("Recall rate", fontsize=20)
            else:
                ax.set_ylabel("")

            if row == 2 and col == 1:
                ax.set_xlabel("M", fontsize=20)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if row == 0 and col == 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    labels,
                    fontsize=20,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.73),
                    ncol=3,
                )
            else:
                ax.legend().remove()

            ax.tick_params(labelsize=20)

    output_path = "./3x3_capacity_figure_r3.png"
    plt.savefig(output_path, dpi=480, bbox_inches="tight")
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

    import torch

    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    sns.set_theme(
        rc={
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 16,
            "legend.title_fontsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "font.family": "serif",
            "font.serif": ["Courier New"],
        }
    )
    sns.set_style("whitegrid")

    if args.replot is True:
        logger.info("Re generating figures...")
        run_replot(args)

    else:

        result_data = {"model": [], "recall rate": [], "M": [], "Geometry": []}

        logger.info("Start running")

        M_min, M_max = args.M_min, args.M_max
        K = 15
        r = (M_max / M_min) ** (1 / K)

        M_values = M_min * r ** np.arange(K + 1)
        M_values = np.round(M_values).astype(int)

        for M in M_values:

            for i in range(args.n_trials):

                identity_phi_rate = run_recall_hyperbolic(
                    args, "identity", M, int(i + M)
                )
                result_data["model"].append("Karcher-Flow")

                result_data["recall rate"].append(identity_phi_rate)

                result_data["M"].append(M)

                result_data["Geometry"].append("H")

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
            output_dir = Path(
                args.output_dir
                + str("/dim")
                + str(args.d)
                + str("/Radius")
                + str(args.mem_R)
                + str("/result.csv")
            )
        else:
            output_dir, _ = _image_csv_plot_paths(args)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir)

        plt.figure(figsize=(8, 5))

        custom_palette = {
            "MHN": "#37373E",
            "DAM": "#86928B",
            "Karcher-Flow": "#FF6B6B",
            "geo_distance": "#FFAC6B",
            "square_distance": "#FBB7C0",
        }

        ax = sns.lineplot(
            data=df,
            x="M",
            y="recall rate",
            hue="model",
            errorbar="sd",
            marker="o",
            markersize=10,
            alpha=0.7,
            palette=custom_palette,
            err_style="bars",
        )

        if dataset != "synthetic":
            dim_str = image_title_dim(args)
            title_str = f"Recall Rate vs M ({dataset}, d={dim_str})"
        else:
            title_str = f"Recall Rate vs M (d={args.d})"
        ax.set_title(title_str)
        ax.set_xlabel("M")
        ax.set_ylabel("Recall rate")

        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        plt.tight_layout(pad=0.1)

        if dataset == "synthetic":
            plot_dir = Path(
                args.output_dir
                + str("/dim")
                + str(args.d)
                + str("/Radius")
                + str(args.mem_R)
                + str("/recall_plot.png")
            )
        else:
            _, plot_dir = _image_csv_plot_paths(args)
        plot_dir.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(plot_dir, dpi=480)
        plt.close()


if __name__ == "__main__":
    main()
