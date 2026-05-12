from icml_hyp.config import recall_config

recall_config.early_set_geomstats_backend_from_argv()

import argparse
import logging
import os
from pathlib import Path

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icml_hyp.config.recall_config import (
    beta_path_segment,
    image_feature_dir,
    image_title_dim,
    image_csv_plot_paths,
    synthetic_csv_plot_paths,
)

from icml_hyp.recall.memory_recall import run_recall_hyperbolic
from icml_hyp.recall.baseline_recall import run_recall_dam, run_recall_mhn
from icml_hyp.integrations.wandb_utils import init_wandb_run
from icml_hyp.recall_plot_combined_style import (
    apply_combined_figure_rcparams,
    decorate_combined_grid_cell,
    plot_recall_df_errorbars,
    save_single_panel_recall_plot,
)

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
    ap.add_argument("--beta", type=float, default=10)

    ap.add_argument("--tol", type=float, default=0.001)

    ap.add_argument("--replot", action="store_true")
    ap.add_argument(
        "--plot-3x3",
        action="store_true",
        help="Generate 3x3 figure with all datasets and dimensions",
    )
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", None),
        help="W&B project (or env WANDB_PROJECT)",
    )
    ap.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", None),
        help="W&B entity/team (or env WANDB_ENTITY)",
    )
    ap.add_argument(
        "--wandb-group",
        type=str,
        default=os.environ.get("WANDB_GROUP", None),
        help="W&B group for grouping multiple runs (or env WANDB_GROUP)",
    )
    ap.add_argument(
        "--wandb-name",
        type=str,
        default=os.environ.get("WANDB_NAME", None),
        help="W&B run name (or env WANDB_NAME). Defaults to an auto-derived name.",
    )
    ap.add_argument(
        "--wandb-tags",
        type=str,
        default=os.environ.get("WANDB_TAGS", None),
        help="Comma-separated W&B tags (or env WANDB_TAGS)",
    )

    args = ap.parse_args()
    args.use_batch = not args.no_batch
    return args


def run_replot(args):

    dataset = getattr(args, "dataset", "synthetic")
    if dataset == "synthetic":
        csv_path, plot_path = synthetic_csv_plot_paths(args)
    else:
        csv_path, plot_path = image_csv_plot_paths(args)
    df = pd.read_csv(csv_path)

    if dataset != "synthetic":
        dim_str = image_title_dim(args)
        title_str = f"Recall Rate vs M ({dataset}, d={dim_str}, β={args.beta:g})"
    else:
        title_str = f"Recall Rate vs M (d={args.d}, β={args.beta:g})"
    save_single_panel_recall_plot(df, plot_path, title=title_str)


def plot_3x3_figure(args):
    apply_combined_figure_rcparams()
    left_frac = 3.2 / (3.2 + 2.0)
    fig_w = (2200 / 300) * left_frac * 1.15
    fig_h = (600 / 300) * 1.15
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = GridSpec(1, 1, figure=fig)
    left_gs = outer[0].subgridspec(3, 3, hspace=0.20, wspace=0.20)
    axes_left = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            axes_left[r, c] = fig.add_subplot(left_gs[r, c])

    datasets = ["synthetic", "mnist", "cifar10"]
    dims = [10, 20, 100]
    bseg = beta_path_segment(args)

    for row, dataset in enumerate(datasets):
        for col, dim in enumerate(dims):
            ax = axes_left[row, col]

            if dataset == "synthetic":
                csv_path = (
                    Path(args.output_dir)
                    / f"dim{dim}"
                    / f"Radius{args.mem_R}"
                    / bseg
                    / "result.csv"
                )
            else:
                csv_path = (
                    Path(args.output_dir)
                    / dataset
                    / f"pca{dim}"
                    / f"Radius{args.mem_R}"
                    / bseg
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
                    fontsize=10,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            df = pd.read_csv(csv_path)
            plot_recall_df_errorbars(ax, df, font_size=10)
            decorate_combined_grid_cell(ax, row=row, col=col, font_size=10)

    output_path = f"./3x3_capacity_figure_R{args.mem_R}_{beta_path_segment(args)}.png"
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"3x3 figure saved to {output_path}")


def _wandb_init_capacity(args):
    dataset = getattr(args, "dataset", "synthetic")
    feat = image_feature_dir(args) if dataset != "synthetic" else f"dim{args.d}"
    run_name = args.wandb_name or f"capacity/{dataset}/{feat}/R{args.mem_R}/b{args.beta:g}"
    extra_tags = [
        "sim:capacity",
        f"dataset:{dataset}",
        f"feat:{feat}",
        f"R:{args.mem_R}",
        f"beta:{args.beta:g}",
        f"device:{args.device}",
    ]
    return init_wandb_run(
        args,
        project=f"ICML-Hyperbolic-{args.dataset}",
        run_name=run_name,
        extra_tags=extra_tags,
    )


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

    if args.replot is True:
        logger.info("Re generating figures...")
        run_replot(args)

    elif args.plot_3x3:
        plot_3x3_figure(args)

    else:
        wandb_run = _wandb_init_capacity(args)
        try:

            result_data = {"model": [], "recall rate": [], "M": [], "Geometry": []}

            logger.info("Start running")

            M_min, M_max = args.M_min, args.M_max
            K = 15
            r = (M_max / M_min) ** (1 / K)

            M_values = M_min * r ** np.arange(K + 1)
            M_values = np.round(M_values).astype(int)

            for m_idx, M in enumerate(M_values):

                trial_rates = {"MHN": [], "DAM": [], "Karcher-Flow": []}

                for i in range(args.n_trials):
                    identity_phi_rate = run_recall_hyperbolic(
                        args, "identity", M, int(i + M)
                    )
                    dam_rate = run_recall_dam(args, M, int(i + M))
                    mhn_rate = run_recall_mhn(args, M, int(i + M))

                    trial_rates["Karcher-Flow"].append(identity_phi_rate)
                    trial_rates["DAM"].append(dam_rate)
                    trial_rates["MHN"].append(mhn_rate)

                    result_data["model"].append("Karcher-Flow")
                    result_data["recall rate"].append(identity_phi_rate)
                    result_data["M"].append(M)
                    result_data["Geometry"].append("H")

                    result_data["model"].append("DAM")
                    result_data["recall rate"].append(dam_rate)
                    result_data["M"].append(M)
                    result_data["Geometry"].append("E")

                    result_data["model"].append("MHN")
                    result_data["recall rate"].append(mhn_rate)
                    result_data["M"].append(M)
                    result_data["Geometry"].append("E")

                    if wandb_run is not None:
                        step = int(m_idx * args.n_trials + i)
                        wandb_run.log(
                            {
                                "step": step,
                                "M": int(M),
                                "trial": int(i),
                                "trial/recall_KarcherFlow": float(identity_phi_rate),
                                "trial/recall_DAM": float(dam_rate),
                                "trial/recall_MHN": float(mhn_rate),
                            }
                        )

                if wandb_run is not None:

                    def _mean_sd(vals: list[float]) -> tuple[float, float]:
                        if not vals:
                            return 0.0, 0.0
                        mu = float(np.mean(vals))
                        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        return mu, sd

                    kf_mu, kf_sd = _mean_sd(trial_rates["Karcher-Flow"])
                    dam_mu, dam_sd = _mean_sd(trial_rates["DAM"])
                    mhn_mu, mhn_sd = _mean_sd(trial_rates["MHN"])
                    wandb_run.log(
                        {
                            "M": int(M),
                            "agg/recall_KarcherFlow": kf_mu,
                            "agg/recall_sd_KarcherFlow": kf_sd,
                            "agg/recall_DAM": dam_mu,
                            "agg/recall_sd_DAM": dam_sd,
                            "agg/recall_MHN": mhn_mu,
                            "agg/recall_sd_MHN": mhn_sd,
                        }
                    )

            df = pd.DataFrame(result_data)
            dataset = getattr(args, "dataset", "synthetic")
            if dataset == "synthetic":
                csv_path, plot_path = synthetic_csv_plot_paths(args)
            else:
                csv_path, plot_path = image_csv_plot_paths(args)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path)

            if dataset != "synthetic":
                dim_str = image_title_dim(args)
                title_str = f"Recall Rate vs M ({dataset}, d={dim_str}, β={args.beta:g})"
            else:
                title_str = f"Recall Rate vs M (d={args.d}, β={args.beta:g})"
            save_single_panel_recall_plot(df, plot_path, title=title_str)

        finally:
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()
